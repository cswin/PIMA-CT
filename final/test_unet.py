# -*- coding: utf-8 -*-
# @Time    : 2020/2/26 15:39
# @Author  : xls56i

import argparse
import os, datetime
import numpy as np
import torch.nn as nn
import torch
import cv2
from skimage.measure import compare_psnr, compare_ssim
import networks

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='../dataset/patients_noisy/Gua_sim_LD_30mAs/test', type=str,
                        help='directory of test dataset')
    parser.add_argument('--model_dir', default='./models/Cyclegan-both-unet-combinedloss-simulation-30mAs_Gd',
                        help='directory of the model')
    parser.add_argument('--result_dir', default='results/Cyclegan_denoising_results_gaussian30mAs', type=str, help='directory of test dataset')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = networks.define_G(1, 1, 64, 'unet_64', 'batch', False, 'normal', 0.02)
    if torch.cuda.is_available():
        model = model.cuda()
        device_ids = [0]
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    psnr_res = 0
    ssim_res = 0
    res_model = ''

    model_pth = os.listdir(args.model_dir)

    for pth in model_pth:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, pth)))
        log('load trained model')

        model.eval()  # evaluation mode

        psnrs = []
        ssims = []
        for im in os.listdir(args.set_dir):
            if im.endswith(".tif") or im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                clean = cv2.imread(os.path.join('../dataset/CT_Data_All_Patients/test', im), 0)
                clean = np.array(clean, dtype=np.float32) / 255.0

                x = cv2.imread(os.path.join(args.set_dir, im), 0)
                height, weight = x.shape
                resize_h = round(height / 64) * 64
                resize_w = round(weight / 64) * 64
                x = cv2.resize(x, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                x = np.array(x, dtype=np.float32) / 255.0
                x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])

                if torch.cuda.is_available():
                    x = x.cuda()
                x_ = model(x)  # inference
                x_ = x_.view(x.shape[2], x.shape[3])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                x_ = cv2.resize(x_, (weight, height), interpolation=cv2.INTER_LINEAR)
                x_[np.where(clean == 0)] = 0
                x_[np.where(clean == 1)] = 1

                psnr_x_ = compare_psnr(clean, x_)
                ssim_x_ = compare_ssim(clean, x_)

                # print('%10s  PSNR: %2.2f SSIM: %1.4f' % (im, psnr_x_, ssim_x_))

                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        print('PSNR = %2.2f dB, SSIM = %1.4f' % (psnr_avg, ssim_avg))

        if psnr_avg > psnr_res:
            psnr_res = psnr_avg
            ssim_res = ssim_avg
            res_model = pth
    print('Finally model = %s PSNR = %2.2f dB, SSIM = %1.4f' % (res_model, psnr_res, ssim_res))

    model.load_state_dict(torch.load(os.path.join(args.model_dir, res_model)))
    log('load trained model')
    model.eval()  # evaluation mode

    for im in os.listdir(args.set_dir):
        if im.endswith(".tif") or im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = cv2.imread(os.path.join(args.set_dir, im), 0)
            height, weight = x.shape
            resize_h = round(height / 64) * 64
            resize_w = round(weight / 64) * 64
            x = cv2.resize(x, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            x = np.array(x, dtype=np.float32) / 255.0
            x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])

            clean = np.array(cv2.imread(os.path.join(args.set_dir, im)), dtype=np.float32) / 255.0

            if torch.cuda.is_available():
                x = x.cuda()
            x_ = model(x)  # inference
            x_ = x_.view(x.shape[2], x.shape[3])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float32)
            x_ = cv2.resize(x_, (weight, height), interpolation=cv2.INTER_LINEAR)
            x_[np.where(clean == 0)] = 0
            x_[np.where(clean == 1)] = 1

            x_ = x_ * 255.0
            x_ = np.array(x_, dtype='uint8')
            cv2.imwrite(os.path.join(args.result_dir, im), x_)

