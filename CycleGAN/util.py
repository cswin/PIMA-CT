# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 13:39
# @Author  : xls56i

from __future__ import print_function
import glob
import re
import os, datetime


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
