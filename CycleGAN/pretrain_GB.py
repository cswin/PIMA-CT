# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 16:39
# @Author  : xls56i

from __future__ import print_function
import argparse
import os
import random
import glob
import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data_generator_bg as dg
from data_generator_bg import DenoisingDataset
from torch.optim.lr_scheduler import MultiStepLR
import networks
import myloss
from util import findLastCheckpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

parser = argparse.ArgumentParser(description='PyTorch GAN')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--model', default='pretrain_GB', type=str, help='choose a type of model')
parser.add_argument('--lambda1', default=5.0, type=float, help='rec loss')
parser.add_argument('--lambda2', default=5.0, type=float, help='TV loss')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')
parser.add_argument('--from_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str,
                    help='path of low-does data')
parser.add_argument('--to_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str,
                    help='path of high-does data')
parser.add_argument('--threshold', default=0.1, type=float, help='background threshold')
parser.add_argument('--epoch', default=60, type=int, help='number of train epoches')

args = parser.parse_args()

gen_save_dir = os.path.join('models',
                            args.model + '_l1_' + args.from_does.split('_')[-1] + '-' + args.to_does.split('_')[-1])
cuda = torch.cuda.is_available()

if not os.path.exists(gen_save_dir):
    os.mkdir(gen_save_dir)

if __name__ == '__main__':

    netG = networks.define_G(1, 1, 64, 'unet_64', 'batch', False, 'normal', 0.02)

    # criterion1 = nn.BCELoss()
    criterion = nn.MSELoss()
    criterion1 = nn.L1Loss()

    if cuda:
        netG = netG.cuda()
        device_ids = [0]
        netG = nn.DataParallel(netG, device_ids=device_ids).cuda()
        criterion = criterion.cuda()
        criterion1 = criterion1.cuda()

    initial_epoch = findLastCheckpoint(save_dir=gen_save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        netG.load_state_dict(torch.load(os.path.join(gen_save_dir, 'model_%03d.pth' % initial_epoch)))

    xs = dg.datagenerator(args.from_does, args.to_does, args.batch_size, args.aug_times, args.patch_size, args.stride,
                          args.threshold)
    xs = xs.astype('float32') / 255.0
    xs = torch.from_numpy(xs.transpose((0, 1, 4, 2, 3)))  # tensor of the clean patches, N X C X H X W
    DDataset = DenoisingDataset(xs)
    DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=args.batch_size, shuffle=True)

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    schedulerG = MultiStepLR(optimizerG, milestones=[5, 10, 15, 20, 25, 30, 40, 50], gamma=0.2)

    for epoch in range(initial_epoch, args.epoch):
        for i, data in enumerate(DLoader, 0):
            if cuda:
                noise, clean = data[0].cuda(), data[1].cuda()
            else:
                noise, clean = data[0], data[1]
            output = netG(noise)
            loss_G = criterion1(output, clean)
            # errG2 = criterion1(output)
            # loss_G = args.lambda1*errG1 + args.lambda2*errG2
            # loss_G = errG1(output, clean)
            loss_G.backward()
            optimizerG.step()
            print('[%d/%d][%d/%d] MSE-Loss: %.4f' % (epoch, args.epoch, i, len(DLoader), loss_G))
        schedulerG.step(epoch)
        # do checkpointing
        torch.save(netG.state_dict(), os.path.join(gen_save_dir, 'model_%03d.pth' % (epoch + 1)))
