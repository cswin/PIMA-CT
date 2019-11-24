# -*- coding: utf-8 -*-
# @Time    : 2019-11-03 23:09
# @Author  : xls56i

from __future__ import print_function

import matplotlib.pyplot as plt
import time
import random
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import info_data_generator as dg
from info_data_generator import DenoisingDataset
import networks
import myloss
from torch.optim.lr_scheduler import MultiStepLR
from util import findLastCheckpoint, noise_sample


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch GAN')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--model', default='infoGAN', type=str, help='choose a type of model')
parser.add_argument('--modelQ_type', default='classification', type=str, help='choose a type of model, classification or regression')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')
parser.add_argument('--from_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str, help='path of high-does data')
parser.add_argument('--to60_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str, help='path of low-does data')
parser.add_argument('--to30_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_30mAs', type=str, help='path of low-does data')
parser.add_argument('--threshold', default=0.05, type=float, help='background threshold')
parser.add_argument('--epoch', default=90, type=int, help='number of train epoches')

args = parser.parse_args()
cuda = torch.cuda.is_available()

netG_dir = os.path.join('models', args.modelQ_type + args.model+'_netG')
netFE_dir = os.path.join('models', args.modelQ_type + args.model+'_netFE')
netD_dir = os.path.join('models', args.modelQ_type + args.model+'_netD')
netQ_dir = os.path.join('models', args.modelQ_type + args.model+'_netQ')

if not os.path.exists(netG_dir):
    os.mkdir(netG_dir)
if not os.path.exists(netFE_dir):
    os.mkdir(netFE_dir)
if not os.path.exists(netD_dir):
    os.mkdir(netD_dir)
if not os.path.exists(netQ_dir):
    os.mkdir(netQ_dir)

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

if __name__ == '__main__':

    # Initialise the network.
    netG = networks.define_G(2, 1, 64, 'unet_64')
    # print(netG)

    netFE = networks.define_D(0, 0, 'FE_infogan')
    # print(netFE)

    netD = networks.define_D(0, 0, 'DHead')
    # print(netD)

    netQ = networks.define_D(0, 0, 'QHead')
    # print(netQ)

    # Loss for recover between fake image and real image
    criterion = nn.MSELoss()
    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()
    # Loss for discrete latent code.
    criterionQ_dis = nn.CrossEntropyLoss()
    # Loss for continuous latent code.
    criterionQ_con = myloss.NormalNLLLoss()

    if cuda:
        netG = netG.cuda()
        netFE = netFE.cuda()
        netD = netD.cuda()
        netQ = netQ.cuda()
        device_ids = [0]
        netG = nn.DataParallel(netG, device_ids=device_ids).cuda()
        netFE = nn.DataParallel(netFE, device_ids=device_ids).cuda()
        netD = nn.DataParallel(netD, device_ids=device_ids).cuda()
        netQ = nn.DataParallel(netQ, device_ids=device_ids).cuda()
        criterionD = criterionD.cuda()
        criterionQ_dis = criterionQ_dis.cuda()
        criterionQ_con = criterionQ_con.cuda()
        criterion = criterion.cuda()

    # Adam optimiser is used.
    optimD = optim.Adam([{'params': netFE.parameters()}, {'params': netD.parameters()}], lr=args.lr, betas=(args.beta1, args.beta2))
    optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=args.lr, betas=(args.beta1, args.beta2))
    schedulerD = MultiStepLR(optimD, milestones=[30, 60, 90], gamma=0.2)
    schedulerG = MultiStepLR(optimG, milestones=[30, 60, 90], gamma=0.2)

    initial_epoch = findLastCheckpoint(save_dir=netG_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        netG.load_state_dict(torch.load(os.path.join(netG_dir, 'model_%03d.pth' % initial_epoch)))

    initial_epoch = findLastCheckpoint(save_dir=netFE_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        netFE.load_state_dict(torch.load(os.path.join(netFE_dir, 'model_%03d.pth' % initial_epoch)))

    initial_epoch = findLastCheckpoint(save_dir=netD_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        netD.load_state_dict(torch.load(os.path.join(netD_dir, 'model_%03d.pth' % initial_epoch)))

    initial_epoch = findLastCheckpoint(save_dir=netQ_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        netQ.load_state_dict(torch.load(os.path.join(netQ_dir, 'model_%03d.pth' % initial_epoch)))


    xs, levels = dg.datagenerator(args.from_does, args.to60_does, args.to30_does, args.batch_size, args.aug_times, args.patch_size, args.stride,
                          args.threshold)
    xs = xs.astype('float32') / 255.0
    levels = levels.astype('float32')
    xs = torch.from_numpy(xs.transpose((0, 1, 4, 2, 3)))  # tensor of the clean patches, N X C X H X W
    DDataset = DenoisingDataset(xs, levels)
    DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=args.batch_size, shuffle=True)

    real_label = 1
    fake_label = 0

    # List variables to store results pf training.
    G_losses = []
    D_losses = []

    print("-"*25)
    print("Starting Training Loop...\n")
    print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format('175-60-30') % (args.epoch, args.batch_size, len(DLoader)))
    print("-"*25)

    start_time = time.time()

    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        schedulerD.step(epoch)
        schedulerG.step(epoch)
        for i, data in enumerate(DLoader, 0):

            high_dose = data[0] # 175mAs
            low_dose = data[1] # 60mAs or 30 mAs
            noise_level = data[2] # 60 or 30
        
            # Get batch size
            b_size = high_dose.size(0)

            # Transfer data tensor to GPU/CPU (device)
            if cuda:
                low_dose = low_dose.cuda()

            # Updating discriminator and DHead
            optimD.zero_grad()
            # Real data
            label = torch.full((b_size, ), real_label)
            if cuda:
                label = label.cuda()
            output1 = netFE(low_dose)
            probs_real = netD(output1).view(-1)
            loss_real = criterionD(probs_real, label)
            # Calculate gradients.
            loss_real.backward()

            # Fake data
            label.fill_(fake_label)
            noise = noise_sample(high_dose, noise_level)
            if cuda:
                noise = noise.cuda()
            fake_data = netG(noise)
            output2 = netFE(fake_data.detach())
            probs_fake = netD(output2).view(-1)
            loss_fake = criterionD(probs_fake, label)
            # Calculate gradients.
            loss_fake.backward()

            # Net Loss for the discriminator
            D_loss = loss_real + loss_fake
            # Update parameters
            optimD.step()

            # Updating Generator and QHead
            optimG.zero_grad()

            # Fake data treated as real.
            output = netFE(fake_data)
            label.fill_(real_label)
            probs_fake = netD(output).view(-1)
            gen_loss = criterionD(probs_fake, label)

            q_logits, q_mu, q_var = netQ(output) # q_logits:[128*2], q_mu:[128*1], q_var:[128*1]
            target = torch.LongTensor([0 if level == 60 else 1 for level in noise_level]) # 60mAs第一类，30mAs第二类
            if cuda:
                target = target.cuda()
                noise_level = noise_level.cuda()
            # Calculating recover loss
            recover_loss = criterion(fake_data, low_dose)
            # print(recover_loss)
            # Calculating loss for discrete latent code.
            dis_loss = criterionQ_dis(q_logits, target)
            # print(dis_loss)
            # Calculating loss for continuous latent code.
            con_loss = criterionQ_con(noise_level/175., q_mu, q_var) * 0.1
            # print(con_loss)

            # Net loss for generator.
            if args.modelQ_type == 'classification': # 当作分类问题
                G_loss = recover_loss + gen_loss + dis_loss
            elif args.modelQ_type == 'regression': # 当作回归问题
                G_loss = recover_loss + gen_loss + con_loss
            else: # 既分类又回归
                G_loss = recover_loss + gen_loss + dis_loss + con_loss

            # Calculate gradients.
            G_loss.backward()
            # Update parameters.
            optimG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, args.epoch, i, len(DLoader), D_loss.item(), G_loss.item()))

            # Save the losses for plotting.
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())


        epoch_time = time.time() - epoch_start_time
        print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
        # Save network weights.
        torch.save(netG.state_dict(), os.path.join(netG_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(netFE.state_dict(), os.path.join(netFE_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(netD.state_dict(), os.path.join(netD_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(netQ.state_dict(), os.path.join(netQ_dir, 'model_%03d.pth' % (epoch + 1)))

    training_time = time.time() - start_time
    print("-"*50)
    print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
    print("-"*50)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Loss Curve {}".format('175-60-30'))

