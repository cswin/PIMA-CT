# -*- coding: utf-8 -*-
# @Time    : 2019/11/25 21:19
# @Author  : xls56i

import glob
import cv2
import numpy as np
import random
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


class BoseDenoisingDataset(Dataset):
    def __init__(self, data_noise_free, real_noise, data_align):
        super(BoseDenoisingDataset, self).__init__()
        self.data_noise_free = data_noise_free
        self.real_noise = real_noise
        self.data_align = data_align
        self.data_noise_free_size = data_noise_free.size(0)
        self.real_noise_size = real_noise.size(0)
        self.data_align_size = data_align.size(0)

    def __getitem__(self, index):
        noise_free = self.data_noise_free[index % self.data_noise_free_size]
        real_noise_idx = random.randint(0, self.real_noise_size-1)
        real_noise = self.real_noise[real_noise_idx % self.real_noise_size]
        index_data_align = random.randint(0, self.data_align_size-1)
        align = self.data_align[index_data_align % self.data_align_size]
        return {'noise_free': noise_free, 'real_noise': real_noise, 'noise': align[0], 'no-noise': align[1]}

    def __len__(self):
        return max(max(self.data_noise_free_size, self.real_noise_size), self.data_align_size)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


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


def realA_datagenerator(dir1='../dataset/noise_free', batch_size=128, aug_times=3, patch_size=64, stride=20, threshold=0.1, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(dir1 + '/*.bmp') + glob.glob(dir1 + '/*.tif')
    file_list1 = sorted(file_list1)
    # initrialize
    dataA = []
    # generate A patches
    for i in range(len(file_list1)):
        patches = unaligned_gen_patches(file_list1[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            dataA.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list1)) + ' is done ^_^')

    dataA = np.array(dataA, dtype='uint8')
    #print(dataA.shape)
    dataA = np.expand_dims(dataA, axis=3)
    discard_n = len(dataA) - len(dataA) // batch_size * batch_size  # because of batch namalization
    dataA = np.delete(dataA, range(discard_n), axis=0)
    print('^_^-training dataA finished-^_^')
    return dataA


def realB_datagenerator(dir1='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', batch_size=128, aug_times=3, patch_size=64, stride=20, threshold=0.1, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(dir1 + '/*.tif')
    file_list1 = sorted(file_list1)
    # initrialize
    dataA = []
    # generate A patches
    for i in range(len(file_list1)):
        patches = unaligned_gen_patches(file_list1[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            dataA.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list1)) + ' is done ^_^')

    dataA = np.array(dataA, dtype='uint8')
    dataA = np.expand_dims(dataA, axis=3)
    discard_n = len(dataA) - len(dataA) // batch_size * batch_size  # because of batch namalization
    dataA = np.delete(dataA, range(discard_n), axis=0)
    print('^_^-training dataB finished-^_^')
    return dataA


def aligned_datagenerator(from_dir, to_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(from_dir+'/*.tif') + glob.glob(from_dir+'/*.bmp')
    file_list2 = glob.glob(to_dir+'/*.tif') + glob.glob(from_dir+'/*.bmp')
    file_list1 = sorted(file_list1)
    file_list2 = sorted(file_list2)
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list1)):
        patches = aligned_gen_patches(file_list1[i], file_list2[i], aug_times, patch_size, stride, threshold)
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


def aligned_gen_patches(file_name1, file_name2, aug_times, patch_size, stride, threshold):
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


def unaligned_gen_patches(file_name, aug_times, patch_size, stride, threshold):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img[i:i+patch_size, j:j+patch_size]
            count_bg = (x==0).sum()
            if count_bg > patch_size*patch_size*threshold:
                continue
            for k in range(0, aug_times):
                mode = np.random.randint(0, 8)
                x_aug = data_aug(x, mode)
                patches.append(x_aug)
    return patches
