# -*- coding: utf-8 -*-
# @Time    : 2019/11/26 13:54
# @Author  : xls56i

from __future__ import print_function
import cycle_data_generator as dg
from cycle_data_generator import AlignedDenoisingDataset, UnAlignedDenoisingDataset
import torch
from torch.utils.data import DataLoader
from cyclegan_model import CycleGANModel
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch CycleGAN')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--model', default='unet64_basic', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')
parser.add_argument('--datasetA', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str, help='path of high-dose data')
parser.add_argument('--datasetB', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str, help='path of low-dose data')
parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
parser.add_argument('--threshold', default=0.1, type=float, help='background threshold')
parser.add_argument('--epoch', default=90, type=int, help='number of train epoches')
parser.add_argument('--datatype', default='aligned', type=str, help='datatype: aligned or unaligned')
parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', default=True,action='store_true', help='no dropout for the generator')
parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
parser.add_argument('--lambda_DN', type=float, default=10.0, help='weight for DN(A) and fakeB')
parser.add_argument('--noise_level', type=float, default=0.34, help='noise level, default:60/175')
parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. n_layers allows you to specify the layers in the discriminator')
parser.add_argument('--netG', type=str, default='unet_64', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_64 | unet_256 | unet_128]')
parser.add_argument('--isTrain', default=True, help='Train or Test')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--model_dir', default='models', type=str)
parser.add_argument('--DNmodel_dir', default='../DN/models/unet64_basic_condition_G/model_001.pth', type=str)
args = parser.parse_args()

if args.datatype == 'aligned':
    xs = dg.aligned_datagenerator(args.datasetA, args.datasetB, args.batch_size, args.aug_times, args.patch_size, args.stride, args.threshold)
    xs = xs.astype('float32') / 255.0
    xs = torch.from_numpy(xs.transpose((0, 1, 4, 2, 3)))  # tensor of the clean patches, N X C X H X W
    DDataset = AlignedDenoisingDataset(xs)
    DLoader = DataLoader(dataset=DDataset, num_workers=0, drop_last=True, batch_size=args.batch_size, shuffle=True)

else:
    dataA, dataB = dg.unaligned_datagenerator(args.from_does, args.to_60does, args.batch_size, args.aug_times, args.patch_size, args.stride, args.threshold)
    dataA = dataA.astype('float32') / 255.0
    dataB = dataB.astype('float32') / 255.0
    dataA = torch.from_numpy(dataA.transpose((0, 3, 1, 2)))
    dataB = torch.from_numpy(dataB.transpose((0, 3, 1, 2)))
    DDataset = UnAlignedDenoisingDataset(dataA, dataB)
    DLoader = DataLoader(dataset=DDataset, num_workers=0, drop_last=True, batch_size=args.batch_size, shuffle=True)

model = CycleGANModel(args)
model.load_networks(args)
for epoch in range(args.epoch):
    for i, data in enumerate(DLoader):
        model.set_input(data)
        model.optimize_parameters()
        print('[%d/%d][%d/%d] Loss_DA: %.4f Loss_DB: %.4f Loss_G: %.4f' % (epoch, args.epoch, i, len(DLoader), model.loss_D_A, model.loss_D_B, model.loss_G))
    model.save_networks(args, epoch)
    model.update_learning_rate(epoch)
