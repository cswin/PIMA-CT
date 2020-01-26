# -*- coding: utf-8 -*-
# @Time    : 2020/1/6 20:01
# @Author  : xls56i

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator_bg as dg
from data_generator_bg import DenoisingDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNND_L1_pha_20mAs', type=str, help='choose a type of model')
parser.add_argument('--loss_type', default=1, type=int, help='loss type')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')
parser.add_argument('--from_does', default='../dataset/phantom_noise_20mAs', type=str, help='path of low-does data')
parser.add_argument('--to_does', default='../dataset/phantom_noise_free', type=str, help='path of high-does data')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
aug_times = args.aug_times
patch_size = args.patch_size
stride = args.stride

cuda = torch.cuda.is_available()
n_epoch = args.epoch

save_dir = os.path.join('models', args.model+'_' + args.from_does.split('_')[-1] + '-' + args.to_does.split('_')[-1])

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    print(model)
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()
    if args.loss_type == 2:
        criterion = nn.MSELoss(reduction='sum')  # PyTorch 0.4.1
    else:
        criterion = nn.L1Loss()
    if cuda:
        model = model.cuda()
        device_ids = [0]
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[15, 30, 45], gamma=0.2)  # learning rates
    for epoch in range(initial_epoch, n_epoch):
        xs = dg.datagenerator(args.from_does, args.to_does, batch_size, aug_times, patch_size, stride, 0.1)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0, 1, 4, 2, 3)))  # tensor of the clean patches, N X C X H X W
        DDataset = DenoisingDataset(xs)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        for n_count, batch in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                batch_x, batch_y = batch[0].cuda(), batch[1].cuda()
            else:
                batch_x, batch_y = batch[0], batch[1]
            loss = criterion(model(batch_x), batch_y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
