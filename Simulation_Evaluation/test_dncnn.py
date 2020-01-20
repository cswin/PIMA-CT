# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 20:58
# @Author  : xls56i

import argparse
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
import cv2
from skimage.io import imread, imsave
from skimage.measure import compare_psnr, compare_ssim

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='./test_real20mAs', type=str,
                        help='directory of test dataset')
    parser.add_argument('--model_dir', default='model',
                        help='directory of the model')
    parser.add_argument('--model_name', default='DnCNNB_30mAs.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results/DnCNN-B-L1_onreal20mAsdata', type=str, help='directory of test dataset')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == '__main__':

    args = parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = DnCNN()
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
                clean = cv2.imread(os.path.join('../dataset/CT_Data_All_Patients/real_high_dose', im), 0)
                clean = np.array(clean, dtype=np.float32) / 255.0

                x = np.array(imread(os.path.join(args.set_dir, im)), dtype=np.float32)/255.0
                x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])
                if torch.cuda.is_available():
                    x = x.cuda()
                x_ = model(x)  # inference
                x_ = x_.view(x.shape[2], x.shape[3])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                x_[np.where(clean == 0)] = 0

                psnr_x_ = compare_psnr(clean, x_)
                ssim_x_ = compare_ssim(clean, x_)

                # print('%10s  PSNR: %2.2f SSIM: %1.4f' % (im, psnr_x_, ssim_x_))

                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        print('model = %s  PSNR = %2.2f dB, SSIM = %1.4f' % (pth, psnr_avg, ssim_avg))

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
            x = np.array(imread(os.path.join(args.set_dir, im)), dtype=np.float32) / 255.0
            clean = x
            x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])

            x_ = model(x)  # inference
            x_ = x_.view(x.shape[2], x.shape[3])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float32)
            x_[np.where(clean == 0)] = 0

            name, ext = os.path.splitext(im)
            save_result(x_, path=os.path.join(args.result_dir, name + ext))  # save the denoised image
