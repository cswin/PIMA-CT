# -*- coding: utf-8 -*-
# @Time    : 2019-11-03 23:09
# @Author  : xls56i

from __future__ import print_function
import networks
import argparse
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.io import imread, imsave
import cv2
from util import log
from util import save_result

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch GAN')
parser.add_argument('--set_dir', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str, help='directory of test dataset')
parser.add_argument('--model_dir', default=os.path.join('models', 'unet64_basic_G_175mAs-60mAs'), help='directory of the model')
parser.add_argument('--model_name', default='model_090.pth', type=str, help='the model name')
parser.add_argument('--result_dir', default='visuals', type=str, help='directory of test dataset')

args = parser.parse_args()

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    
    netG = networks.define_G(1, 1, 64, 'unet64_visual')

    if cuda:
        netG = netG.cuda()
        device_ids = [0]
        netG = nn.DataParallel(netG, device_ids=device_ids).cuda()
        
    netG.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    log('load trained model')

    netG.eval()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    for im in os.listdir(args.set_dir):
        if im.endswith(".tif") or im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = cv2.imread(os.path.join(args.set_dir, im), 0)
            x = cv2.resize(x,(320,384),interpolation=cv2.INTER_LINEAR)
            x = np.array(x, dtype=np.float32)/255.0
            x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])
            visual_dir = os.path.join(args.result_dir, os.path.splitext(im)[0])
            if not os.path.exists(visual_dir):
                os.makedirs(visual_dir)
            start_time = time.time()
            x_ = netG(x)  # inference
            for i in range(len(x_)):
                tmp = x_[i].cpu()
                tmp = tmp.detach().numpy().astype(np.float32)
                np.save(os.path.join(visual_dir, str(i)+'.npy'), tmp)
            elapsed_time = time.time() - start_time
            print('%10s : %2.4f second' % (im, elapsed_time))
