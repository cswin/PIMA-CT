# -*- coding: utf-8 -*-
# @Time    : 2019/12/3 15:06
# @Author  : xls56i

from __future__ import print_function
import cycle_data_generator as dg
from cycle_data_generator import AlignedDenoisingDataset, UnAlignedDenoisingDataset
import torch
from torch.utils.data import DataLoader
from cyclegan_model import CycleGANModel
import os, time
import cv2
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import torch.nn as nn
import networks
from util import log
from util import save_result
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch CycleGAN')

parser.add_argument('--test_dataset', default='../dataset/CT_Data_All_Patients/test/', type=str, help='path of test dataset')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', default=True,action='store_true', help='no dropout for the generator')
parser.add_argument('--netG', type=str, default='unet_64', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_64 | unet_256 | unet_128]')
parser.add_argument('--modelGA_dir', default='models/unet64_basic_origincyclegan_noDN_l1loss_G_A/model_060.pth', type=str)
parser.add_argument('--modelGB_dir', default='models/unet64_basic_origincyclegan_noDN_l1loss_G_B/model_060.pth', type=str)
parser.add_argument('--GA_results', default='results/GAl1loss', type=str)
parser.add_argument('--GB_results', default='results/GBl1loss', type=str)

args = parser.parse_args()

cuda = torch.cuda.is_available()
# netG_A
netG_A = networks.define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.norm, not args.no_dropout, args.init_type, args.init_gain)
netG_B = networks.define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.norm, not args.no_dropout, args.init_type, args.init_gain)
if cuda:
    netG_A = netG_A.cuda()
    netG_B = netG_B.cuda()
    device_ids = [0]
    netG_A = nn.DataParallel(netG_A, device_ids=device_ids).cuda()
    netG_B = nn.DataParallel(netG_B, device_ids=device_ids).cuda()

netG_A.load_state_dict(torch.load(args.modelGA_dir))
netG_B.load_state_dict(torch.load(args.modelGB_dir))


netG_A.eval()  # evaluation mode
netG_B.eval()

if not os.path.exists(args.GA_results):
    os.makedirs(args.GA_results)

if not os.path.exists(args.GB_results):
    os.makedirs(args.GB_results)

psnrs = []
ssims = []
    
for im in os.listdir(args.test_dataset):
    if im.endswith(".tif") or im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
        x = cv2.imread(os.path.join(args.test_dataset, im), 0)
        pre_img = np.array(x, dtype=np.float32) / 255.0
        height, weight = x.shape
        resize_h = round(height/64)*64
        resize_w = round(weight/64)*64
        x = cv2.resize(x, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        x = np.array(x, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])

        x_ = netG_A(x)  # inference
        x_ = x_.view(x_.shape[2], x_.shape[3])
        x_ = x_.cpu()
        x_ = x_.detach().numpy().astype(np.float32)
        x_ = cv2.resize(x_, (weight, height), interpolation=cv2.INTER_LINEAR)

        x_[np.where(pre_img == 0)] = 0

        name, ext = os.path.splitext(im)
        save_result(x_, path=os.path.join(args.GA_results, name + ext))  # save the denoised image
        
        x_ = cv2.resize(x_, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        x_ = torch.from_numpy(x_).view(1, -1, x_.shape[0], x_.shape[1])
        y_ = netG_B(x_)    
        y_ = y_.view(y_.shape[2], y_.shape[3])
        y_ = y_.cpu()
        y_ = y_.detach().numpy().astype(np.float32)
        y_ = cv2.resize(y_, (weight, height), interpolation=cv2.INTER_LINEAR)

        y_[np.where(pre_img == 0)] = 0

        name, ext = os.path.splitext(im)
        save_result(y_, path=os.path.join(args.GB_results, name + ext))  # save the denoised image
        
        psnr_x_ = compare_psnr(pre_img, y_)
        ssim_x_ = compare_ssim(pre_img, y_)
        
        print('%10s  PSNR: %2.2f SSIM: %1.4f' % (im, psnr_x_, ssim_x_))

        psnrs.append(psnr_x_)
        ssims.append(ssim_x_)

psnr_avg = np.mean(psnrs)
ssim_avg = np.mean(ssims)
print('PSNR = %2.2f dB, SSIM = %1.4f' % (psnr_avg, ssim_avg))
