# -*- coding: utf-8 -*-
# @Time    : 2019-10-25 13:06
# @Author  : xls56i

import glob
import cv2
import numpy as np
from torch.utils.data import Dataset


class DenoisingDataset(Dataset):

    def __init__(self, xs):
        super(DenoisingDataset, self).__init__()
        self.xs = xs

    def __getitem__(self, index):
        batch = self.xs[index]
        return batch[0], batch[1]

    def __len__(self):
        return self.xs.size(0)


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img) #翻转
    elif mode == 2:
        return np.rot90(img) #旋转90
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2) #旋转180
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name1, file_name2, aug_times, patch_size, stride, threshold):
    # get multiscale patches from a single image
    img1 = cv2.imread(file_name1, 0)  # gray scale
    img2 = cv2.imread(file_name2, 0)
    h, w = img1.shape
    patches = []
    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img1[i:i+patch_size, j:j+patch_size]
            y = img2[i:i+patch_size, j:j+patch_size]
            count_bg = (x==0).sum()
            if count_bg > patch_size*patch_size*threshold:
                continue
            for k in range(0, aug_times):
                mode = np.random.randint(0, 8)
                x_aug = data_aug(x, mode)
                y_aug = data_aug(y, mode)
                patches.append((x_aug,y_aug))
    return patches


def datagenerator(from_dir, to_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(from_dir+'/*.tif') + glob.glob(from_dir+'/*.bmp')
    file_list2 = glob.glob(to_dir+'/*.tif') + glob.glob(to_dir+'/*.bmp')
    file_list1 = sorted(file_list1)
    file_list2 = sorted(file_list2)
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list1)):
        patches = gen_patches(file_list1[i], file_list2[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list1)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=4)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data

