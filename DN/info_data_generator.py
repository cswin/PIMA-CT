# -*- coding: utf-8 -*-
# @Time    : 2019-11-17 23:24
# @Author  : xls56i

import glob
import cv2
import numpy as np
from torch.utils.data import Dataset


class DenoisingDataset(Dataset):

    def __init__(self, xs, level):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.level = level

    def __getitem__(self, index):
        batch = self.xs[index]
        return batch[0], batch[1], self.level[index]

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


def gen_patches(file_name1, file_name2, file_name3, aug_times, patch_size, stride, threshold):
    # get multiscale patches from a single image
    img1 = cv2.imread(file_name1, 0)  # gray scale
    img2 = cv2.imread(file_name2, 0)
    img3 = cv2.imread(file_name3, 0)
    h, w = img1.shape
    patches = []
    level = []
    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img1[i:i+patch_size, j:j+patch_size]
            y = img2[i:i+patch_size, j:j+patch_size]
            z = img3[i:i+patch_size, j:j+patch_size]
            count_bg = (x==0).sum()
            if count_bg > patch_size*patch_size*threshold:
                continue
            for k in range(0, aug_times):
                mode = np.random.randint(0, 8)
                x_aug = data_aug(x, mode)
                y_aug = data_aug(y, mode)
                z_aug = data_aug(z, mode)
                patches.append((x_aug,y_aug))
                level.append(60)
                patches.append((x_aug,z_aug))
                level.append(30)
    return patches, level


def datagenerator(from_dir, to60_dir, to30_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False):
    # generate clean patches from a dataset
    file_list1 = glob.glob(from_dir+'/*.tif')
    file_list1 = sorted(file_list1)
    file_list2 = glob.glob(to60_dir + '/*.tif')
    file_list2 = sorted(file_list2)
    file_list3 = glob.glob(to30_dir + '/*.tif')
    file_list3 = sorted(file_list3)
    # initrialize
    data = []
    levels = []
    # generate patches
    for i in range(len(file_list1)):
        patches, level = gen_patches(file_list1[i], file_list2[i], file_list3[i], aug_times, patch_size, stride, threshold)
        for patch in patches:
            data.append(patch)
        for l in level:
            levels.append(l)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list1)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    levels = np.array(levels)
    data = np.expand_dims(data, axis=4)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    levels = np.delete(levels, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data, levels

if __name__ == '__main__':
    data = datagenerator('../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', '../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', '../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_30mAs', 128, 3, 64, 20, 0.1)

