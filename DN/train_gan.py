# -*- coding: utf-8 -*-
# @Time    : 2019-11-03 23:09
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
from torch.optim.lr_scheduler import MultiStepLR
from data_generator_bg import DenoisingDataset
import networks
import myloss
from util import findLastCheckpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='PyTorch GAN')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--model', default='unet64_basic_only_advloss', type=str, help='choose a type of model')
parser.add_argument('--lambda1', default=1, type=float, help='adver loss')
parser.add_argument('--lambda2', default=0.0, type=float, help='rec loss')
parser.add_argument('--lambda3', default=0.0, type=float, help='TV loss')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')
parser.add_argument('--from_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str, help='path of high-does data')
parser.add_argument('--to_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str, help='path of low-does data')
parser.add_argument('--threshold', default=0.1, type=float, help='background threshold')
parser.add_argument('--epoch', default=90, type=int, help='number of train epoches')

args = parser.parse_args()

gen_save_dir = os.path.join('models', args.model+'_G_' + args.from_does.split('_')[-1] + '-' + args.to_does.split('_')[-1])
dis_save_dir = os.path.join('models', args.model+'_D_' + args.from_does.split('_')[-1] + '-' + args.to_does.split('_')[-1])
cuda = torch.cuda.is_available()

if not os.path.exists(gen_save_dir):
    os.mkdir(gen_save_dir)
if not os.path.exists(dis_save_dir):
    os.mkdir(dis_save_dir)


if __name__ == '__main__':

    netG = networks.define_G(1, 1, 64, 'unet_64')
    print(netG)

    netD = networks.define_D(1, 64, 'basic')
    print(netD)

    # criterion1 = nn.BCELoss()
    criterion = nn.MSELoss(reduction='sum')
    criterion1 = myloss.TVLoss()

    if cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        device_ids = [0]
        netG = nn.DataParallel(netG, device_ids=device_ids).cuda()
        netD = nn.DataParallel(netD, device_ids=device_ids).cuda()
        criterion = criterion.cuda()
        criterion1 = criterion1.cuda()

    initial_epoch = findLastCheckpoint(save_dir=gen_save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        netG.load_state_dict(torch.load(os.path.join(gen_save_dir, 'model_%03d.pth' % initial_epoch)))

    initial_epoch = findLastCheckpoint(save_dir=dis_save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        netD.load_state_dict(torch.load(os.path.join(dis_save_dir, 'model_%03d.pth' % initial_epoch)))


    xs = dg.datagenerator(args.from_does, args.to_does, args.batch_size, args.aug_times, args.patch_size, args.stride, args.threshold)
    xs = xs.astype('float32') / 255.0
    xs = torch.from_numpy(xs.transpose((0, 1, 4, 2, 3)))  # tensor of the clean patches, N X C X H X W
    DDataset = DenoisingDataset(xs)
    DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=args.batch_size, shuffle=True)

    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    schedulerD = MultiStepLR(optimizerD, milestones=[30, 60, 90], gamma=0.2)
    schedulerG = MultiStepLR(optimizerG, milestones=[30, 60, 90], gamma=0.2)
    
    for epoch in range(initial_epoch, args.epoch):
        schedulerD.step(epoch)
        schedulerG.step(epoch)
        for i, data in enumerate(DLoader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0]

            if cuda:
                real_cpu = real_cpu.cuda()

            output = netD(real_cpu)
            label = torch.full(output.shape, real_label)
            
            if cuda:
                label = label.cuda()
            errD_real = criterion(output, label)
            errD_real.backward()
        
            D_x = output.mean().item()

            # train with fake
            noise = data[1]
            
            if cuda:
                noise = noise.cuda()
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)

            errG1 = criterion(output, label)
            errG2 = criterion(fake, real_cpu)/((fake.shape[0]/output[0].shape[0])**2)
            errG3 = criterion1(fake)/(2*(fake.shape[0]/output[0].shape[0])**2)

            loss_G = args.lambda1*errG1 + args.lambda2*errG2 + args.lambda3*errG3
            loss_G.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epoch, i, len(DLoader),
                     errD.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

        # do checkpointing
        torch.save(netG.state_dict(), os.path.join(gen_save_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(netD.state_dict(), os.path.join(dis_save_dir, 'model_%03d.pth' % (epoch + 1)))

