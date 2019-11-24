# -*- coding: utf-8 -*-
# @Time    : 2019-11-19 14:03
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
import info_data_generator as dg
from info_data_generator import DenoisingDataset
from torch.optim.lr_scheduler import MultiStepLR
import networks
import myloss
from util import findLastCheckpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch GAN')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--model', default='unet64_basic_acgan', type=str, help='choose a type of model')
parser.add_argument('--lambda0', default=0.1, type=float, help='noise_level loss')
parser.add_argument('--lambda1', default=0.1, type=float, help='adver loss')
parser.add_argument('--lambda2', default=0.8, type=float, help='rec loss')
parser.add_argument('--lambda3', default=0.0, type=float, help='TV loss')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')
parser.add_argument('--from_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str,
                    help='path of high-does data')
parser.add_argument('--to_60does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str,
                    help='path of low-does data')
parser.add_argument('--to_30does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_30mAs', type=str,
                    help='path of low-does data')
parser.add_argument('--threshold', default=0.1, type=float, help='background threshold')
parser.add_argument('--epoch', default=90, type=int, help='number of train epoches')

args = parser.parse_args()

gen_save_dir = os.path.join('models',
                            args.model + '_G')
dis_save_dir = os.path.join('models',
                            args.model + '_D')
cuda = torch.cuda.is_available()

if not os.path.exists(gen_save_dir):
    os.mkdir(gen_save_dir)
if not os.path.exists(dis_save_dir):
    os.mkdir(dis_save_dir)

if __name__ == '__main__':

    netG = networks.define_G(2, 1, 64, 'unet_64')
    print(netG)

    netD = networks.define_D(1, 64, 'basic_condition')
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

    xs, levels = dg.datagenerator(args.from_does, args.to_60does, args.to_30does, args.batch_size, args.aug_times,
                          args.patch_size, args.stride, args.threshold)
    xs = xs.astype('float32') / 255.0
    levels = levels.astype('float32')
    xs = torch.from_numpy(xs.transpose((0, 1, 4, 2, 3)))  # tensor of the clean patches, N X C X H X W
    DDataset = DenoisingDataset(xs, levels)
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
            noise_level = data[2]

            if cuda:
                real_cpu = real_cpu.cuda()

            output = netD(real_cpu)
            label = torch.full(output[0].shape, real_label)
            N, C, H, W = output[1].shape
            level_map = noise_level / 175.0
            level_map = level_map.view(N,1,1,1).repeat(1,C,H,W)

            if cuda:
                label = label.cuda()
                level_map = level_map.cuda()
            errD_real = criterion(output[0], label)
            errD_level = criterion(output[1], level_map)
            errD1 = errD_real + errD_level
            errD1.backward()
            D_x = output[0].mean().item()

            # train with fake
            noise = data[1]
            N, C, H, W = noise.size()

            noise_sigma = noise_level / 175.0
            noise_sigma = torch.FloatTensor(noise_sigma)
            noise_map = noise_sigma.view(N, 1, 1, 1).repeat(1, C, H, W)
            noise = torch.cat((noise, noise_map), 1)

            if cuda:
                noise = noise.cuda()
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output[0], label)
            errD_fake_level = criterion(output[1], level_map)
            errD_fake_loss = errD_fake + errD_fake_level
            errD_fake_loss.backward()
            D_G_z1 = output[0].mean().item()
            errD = errD1 + errD_fake_loss
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG0 = criterion(output[1], level_map)
            errG1 = criterion(output[0], label)
            errG2 = criterion(fake, real_cpu) / ((fake.shape[0] / output[0].shape[0]) ** 2)
            errG3 = criterion1(fake) / (2 * (fake.shape[0] / output[0].shape[0]) ** 2)
            loss_G = args.lambda0 * errG0 + args.lambda1 * errG1 + args.lambda2 * errG2 + args.lambda3 * errG3
            loss_G.backward()
            D_G_z2 = output[0].mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epoch, i, len(DLoader),
                     errD.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

        # do checkpointing
        torch.save(netG.state_dict(), os.path.join(gen_save_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(netD.state_dict(), os.path.join(dis_save_dir, 'model_%03d.pth' % (epoch + 1)))
