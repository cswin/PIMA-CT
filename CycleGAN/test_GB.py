# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 15:15
# @Author  : xls56i

from __future__ import print_function
import networks
import argparse
import os, time
import numpy as np
import torch.nn as nn
import torch
import cv2
from util import log
from util import save_result
from skimage.measure import compare_psnr, compare_ssim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

parser = argparse.ArgumentParser(description='PyTorch CycleGAN')
parser.add_argument('--set_dir', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str,
                    help='directory of test dataset')
parser.add_argument('--model_dir', default=os.path.join('models', 'teacher_stu_unet64_basic_60_unaligned_l1loss_G_B'),
                    help='directory of the model:G_A==>high2low,simulate, G_B==>low2high,denoising')
parser.add_argument('--model_name', default='model_003_010.pth', type=str, help='the model name')
parser.add_argument('--isTrain', default=False, help='Train or Test')
parser.add_argument('--result_dir', default='l60to175', type=str, help='directory of test dataset')

args = parser.parse_args()

if __name__ == '__main__':

    cuda = torch.cuda.is_available()

    netG = networks.define_G(1, 1, 64, 'unet_64', 'batch', False, 'normal', 0.02)

    if cuda:
        netG = netG.cuda()
        device_ids = [0]
        netG = nn.DataParallel(netG, device_ids=device_ids).cuda()

    netG.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    log('load trained model')

    netG.eval()  # evaluation mode

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    psnrs = []
    ssims = []

    for im in os.listdir(args.set_dir):
        if im.endswith(".tif") or im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            num = im.split('_')[1]
            clean_im = 'IM-0002-0320_' + num + '_gt_Adj_brain_crop_175mAs.tif'
            clean = cv2.imread(os.path.join('../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', clean_im), 0)
            clean = np.array(clean, dtype=np.float32) / 255.0
            x = cv2.imread(os.path.join(args.set_dir, im), 0)
            pre_img = x
            x = cv2.resize(x, (320, 384), interpolation=cv2.INTER_LINEAR)
            x = np.array(x, dtype=np.float32) / 255.0
            x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])

            start_time = time.time()
            x_ = netG(x)  # inference
            x_ = x_.view(x_.shape[2], x_.shape[3])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float32)
            x_ = cv2.resize(x_, (300, 410), interpolation=cv2.INTER_LINEAR)

            x_[np.where(pre_img == 0)] = 0

            name, ext = os.path.splitext(im)
            save_result(x_, path=os.path.join(args.result_dir, name + '_unet64' + ext))  # save the denoised image
            psnr_x_ = compare_psnr(clean, x_)
            ssim_x_ = compare_ssim(clean, x_)

            print('%10s  PSNR: %2.2f SSIM: %1.4f' % (im, psnr_x_, ssim_x_))

            psnrs.append(psnr_x_)
            ssims.append(ssim_x_)

    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)
    print('PSNR = %2.2f dB, SSIM = %1.4f' % (psnr_avg, ssim_avg))
