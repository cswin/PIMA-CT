# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 17:39
# @Author  : xls56i


from __future__ import print_function
import networks
import argparse
import numpy as np
import torch.nn as nn
import torch
import cv2
import os, datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser(description='PyTorch CycleGAN')
parser.add_argument('--set_dir', default='../dataset/Real_Low_Dose_10images/clean', type=str,
                    help='directory of test dataset')
parser.add_argument('--model_dir', default='./models/cyclegan_simulation_20mAs_4_2_Gs',
                    help='directory of the model:G_A==>high2low,simulate, G_B==>low2high,denoising')
parser.add_argument('--result_dir', default='../dataset/Real_Low_Dose_10images/pseudo20mAs', type=str, help='directory of test dataset')

args = parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

if __name__ == '__main__':

    cuda = torch.cuda.is_available()

    netG = networks.define_G(1, 1, 64, 'unet_64', 'batch', False, 'normal', 0.02)

    if cuda:
        netG = netG.cuda()
        device_ids = [0]
        netG = nn.DataParallel(netG, device_ids=device_ids).cuda()

    model_pth = os.listdir(args.model_dir)

    for pth in model_pth:

        netG.load_state_dict(torch.load(os.path.join(args.model_dir, pth)))
        log('load trained model')

        netG.eval()  # evaluation mode

        save_dir = os.path.join(args.result_dir, pth[:-4])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for im in os.listdir(args.set_dir):
            if im.endswith(".tif") or im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                x = cv2.imread(os.path.join(args.set_dir, im), 0)
                pre_img = x

                height, weight = x.shape
                resize_h = round(height / 64) * 64
                resize_w = round(weight / 64) * 64
                x = cv2.resize(x, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                x = np.array(x, dtype=np.float32) / 255.0
                x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])

                x_ = netG(x)  # inference
                x_ = x_.view(x_.shape[2], x_.shape[3])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                x_ = cv2.resize(x_, (weight, height), interpolation=cv2.INTER_LINEAR)

                x_[np.where(pre_img == 0)] = 0
                x_ = x_ * 255.0
                x_ = np.array(x_, dtype='uint8')
                cv2.imwrite(os.path.join(save_dir, im), x_)
