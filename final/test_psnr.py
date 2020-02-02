# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 20:58
# @Author  : xls56i

import argparse
import os
import numpy as np
import cv2
from skimage.measure import compare_psnr, compare_ssim



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', default='../dataset/CT_Data_All_Patients/train', type=str, help='directory of test dataset')
    parser.add_argument('--noisy_dir', default='../dataset/CT_Data_All_Patients/train002030_simulate60mAs', type=str, help='directory of test dataset')
    # parser.add_argument('--noisy_dir', default='../dataset/patients_noisy/Gua_sim_LD_60mAs/train', type=str, help='directory of test dataset')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    
    psnrs = []
    ssims = []
    for im in os.listdir(args.noisy_dir):
        if im.endswith(".tif") or im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            clean = np.array(cv2.imread(os.path.join(args.clean_dir, im), 0), dtype=np.float32)/255.0
            x = np.array(cv2.imread(os.path.join(args.noisy_dir, im), 0), dtype=np.float32)/255.0
            psnr_x_ = compare_psnr(clean, x)
            ssim_x_ = compare_ssim(clean, x)
            psnrs.append(psnr_x_)
            ssims.append(ssim_x_)

    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)
    print('PSNR = %2.2f dB, SSIM = %1.4f' % (psnr_avg, ssim_avg))
