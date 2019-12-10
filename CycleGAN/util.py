# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 13:39
# @Author  : xls56i

from __future__ import print_function
import glob
import os
import re
import os, time, datetime
import numpy as np
from skimage.io import imread, imsave
import cv2
import torch

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def findLastCheckpoint_time_epoch(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        times_exist = []
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*)_(.*).pth.*", file_)
            times_exist.append(int(result[0][0]))
        initial_time = max(times_exist)
        for file_ in file_list:
            result = re.findall(".*model_(.*)_(.*).pth.*", file_)
            if int(result[0][0]) == initial_time:
                epochs_exist.append(int(result[0][1]))
        initial_epoch = max(epochs_exist)
    else:
        initial_time = 1
        initial_epoch = 0
    return initial_time-1, initial_epoch

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        result = result * 255.0
        result = np.array(result, dtype='uint8')
        imsave(path, result)
        
def noise_sample(high_dose, noise_level):
    N, C, H, W = high_dose.size()
    noise_sigma = noise_level / 175.0
    noise_sigma = torch.FloatTensor(noise_sigma)
    noise_map = noise_sigma.view(N, 1, 1, 1).repeat(1, C, H, W)
    noise = torch.cat((high_dose, noise_map), 1)
    return noise