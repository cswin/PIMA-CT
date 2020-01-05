# -*- coding: utf-8 -*-
# @Time    : 2019/11/25 21:19
# @Author  : xls56i

import glob
import cv2
import numpy as np
import random
from torch.utils.data import Dataset


class AlignedDenoisingDataset(Dataset):

    def __init__(self, xs):
        super(AlignedDenoisingDataset, self).__init__()
        self.xs = xs

    def __getitem__(self, index):
        batch = self.xs[index]
        return {'A': batch[0], 'B': batch[1]}

    def __len__(self):
        return self.xs.size(0)

class UnAlignedDenoisingDataset(Dataset):

    def __init__(self, dataA, dataB):
        super(UnAlignedDenoisingDataset, self).__init__()
        self.dataA = dataA
        self.dataB = dataB
        self.A_size = dataA.size(0)
        self.B_size = dataB.size(0)

    def __getitem__(self, index):
        A = self.dataA[index % self.A_size]
        index_B = random.randint(0, self.B_size - 1)
        B = self.dataB[index_B % self.B_size]
        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)

# 重写dataloader 有对应的图片来求denoising时的loss
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
    dataA = np.expand_dims(dataA, axis=3)
    discard_n = len(dataA) - len(dataA) // batch_size * batch_size  # because of batch namalization
    dataA = np.delete(dataA, range(discard_n), axis=0)
    print('^_^-training dataA finished-^_^')
    return dataA

def realB_datagenerator(dir1='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', batch_size=128, aug_times=3, patch_size=64, stride=20, threshold=0.1, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(dir1 + '/*.tif') + glob.glob(dir1 + '/*.bmp')
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


def bose_datagenerator(dir1='../dataset/noise_free',  dir2='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', dir3='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', batch_size=128, aug_times=3, patch_size=64, stride=20, threshold=0.1, verbose=False):
    file_list1 = glob.glob(dir1+'/*.tif') + glob.glob(dir1+'/*.bmp')
    file_list2 = glob.glob(dir2+'/*.tif')
    file_list3 = glob.glob(dir3+'/*.tif')
    file_list2 = sorted(file_list2)
    file_list3 = sorted(file_list3)

    data_noise_free = []
    data_align = []
    for i in range(len(file_list1)):
        patches = unaligned_gen_patches(file_list1[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            data_noise_free.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list1)) + ' is done ^_^')

    for i in range(len(file_list2)):
        patches = aligned_gen_patches(file_list2[i], file_list3[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            data_align.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list1)) + ' is done ^_^')

    data_noise_free = np.array(data_noise_free, dtype='uint8')
    data_noise_free = np.expand_dims(data_noise_free, axis=4)
    discard_n = len(data_noise_free) - len(data_noise_free) // batch_size * batch_size  # because of batch namalization
    data_noise_free = np.delete(data_noise_free, range(discard_n), axis=0)
    print('^_^-training data_noise_free finished-^_^')

    data_align = np.array(data_align, dtype='uint8')
    data_align = np.expand_dims(data_align, axis=4)
    discard_n = len(data_align) - len(data_align) // batch_size * batch_size  # because of batch namalization
    data_align = np.delete(data_align, range(discard_n), axis=0)
    print('^_^-training data_align finished-^_^')

    return data_noise_free, data_align

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

def aligned_datagenerator_bmp(from_dir, to_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(from_dir+'/*.bmp')
    file_list2 = glob.glob(to_dir+'/*.bmp')
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

def unaligned_datagenerator(from_dir, to_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(from_dir+'/*.bmp') + glob.glob(from_dir+'/*.tif')
    file_list2 = glob.glob(to_dir+'/*.tif')
    file_list1 = sorted(file_list1)
    file_list2 = sorted(file_list2)
    # initrialize
    dataA = []
    dataB = []
    # generate A patches
    for i in range(len(file_list1)):
        patches = unaligned_gen_patches(file_list1[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            dataA.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list1)) + ' is done ^_^')

    # generate A patches
    for i in range(len(file_list2)):
        patches = unaligned_gen_patches(file_list2[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            dataB.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list1)) + ' is done ^_^')

    dataA = np.array(dataA, dtype='uint8')
    dataA = np.expand_dims(dataA, axis=4)
    discard_n = len(dataA)-len(dataA)//batch_size*batch_size  # because of batch namalization
    dataA = np.delete(dataA, range(discard_n), axis=0)
    print('^_^-training dataA finished-^_^')

    dataB = np.array(dataB, dtype='uint8')
    dataB = np.expand_dims(dataB, axis=4)
    discard_n = len(dataB) - len(dataB) // batch_size * batch_size  # because of batch namalization
    dataB = np.delete(dataB, range(discard_n), axis=0)
    print('^_^-training dataB finished-^_^')

    return dataA, dataB

def unaligned_datagenerator_bmp(from_dir, to_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(from_dir+'/*.bmp')
    file_list2 = glob.glob(to_dir+'/*.bmp')
    file_list1 = sorted(file_list1)
    file_list2 = sorted(file_list2)
    # initrialize
    dataA = []
    dataB = []
    # generate A patches
    for i in range(len(file_list1)):
        patches = unaligned_gen_patches(file_list1[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            dataA.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list1)) + ' is done ^_^')

    # generate A patches
    for i in range(len(file_list2)):
        patches = unaligned_gen_patches(file_list2[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            dataB.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list1)) + ' is done ^_^')

    dataA = np.array(dataA, dtype='uint8')
    dataA = np.expand_dims(dataA, axis=4)
    discard_n = len(dataA)-len(dataA)//batch_size*batch_size  # because of batch namalization
    dataA = np.delete(dataA, range(discard_n), axis=0)
    print('^_^-training dataA finished-^_^')

    dataB = np.array(dataB, dtype='uint8')
    dataB = np.expand_dims(dataB, axis=4)
    discard_n = len(dataB) - len(dataB) // batch_size * batch_size  # because of batch namalization
    dataB = np.delete(dataB, range(discard_n), axis=0)
    print('^_^-training dataB finished-^_^')

    return dataA, dataB


if __name__ == '__main__':
    # data = unaligned_datagenerator('../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', '../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', 128, 3, 64, 20, 0.1)
    dataA, dataB = bose_datagenerator('../dataset/noise_free','../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs','../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', 128, 3, 64, 20, 0.1)
