# -*- coding: utf-8 -*-
# @Time    : 2019/12/29 13:06
# @Author  : xls56i

# 重写dataloader 和 loss 的训练

from __future__ import print_function
import cycle_data_generator as dg
from cycle_data_generator import BoseDenoisingDataset, UnAlignedDenoisingDataset
import torch
from torch.utils.data import DataLoader
from cyclegan_model1 import CycleGANModel
import os, time
import cv2
import numpy as np
import torch.nn as nn
import networks
from util import log
from util import save_result
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

parser = argparse.ArgumentParser(description='PyTorch CycleGAN')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--model', default='teacher_stu_unet64_basic_60_unaligned_oneDiscriminator', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')

parser.add_argument('--dataset_unlabeled', default='../dataset/CT_Data_All_Patients/train', type=str, help='noise free for patient')
parser.add_argument('--dataset_noise_free', default='../dataset/noise_free', type=str,
                    help='noise_free patient and phantom high')
parser.add_argument('--dataset_noises', default='../dataset/noises', type=str, help='phantom low and patient simulation')
parser.add_argument('--phantom_low', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str,
                    help='path of low-dose data')

parser.add_argument('--phantom_high', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str,
                    help='path of high-dose data')

parser.add_argument('--result_pseudo', default='../dataset/CT_Data_All_Patients/pseudo', type=str,
                    help='path of pseudo dictionary')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')
parser.add_argument('--threshold', default=0.1, type=float, help='background threshold')
parser.add_argument('--epoch', default=60, type=int, help='number of train epoches')
parser.add_argument('--teach_nums', default=5, type=int, help='number of teach times')
parser.add_argument('--datatype', default='unaligned', type=str, help='datatype: aligned or unaligned')
parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=1,
                    help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='batch',
                    help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal',
                    help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', default=True, action='store_true', help='no dropout for the generator')
parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_identity', type=float, default=0.5,
                    help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
# parser.add_argument('--lambda_DN', type=float, default=10.0, help='weight for DN(A) and fakeB')
# parser.add_argument('--noise_level', type=float, default=0.171, help='noise level, default:60/175')
parser.add_argument('--self_training_loss', type=int, default=2, help='1 for l1loss, 2 for l2loss' )
parser.add_argument('--netD', type=str, default='basic',
                    help='specify discriminator architecture [basic | n_layers | pixel]. n_layers allows you to specify the layers in the discriminator')
parser.add_argument('--netG', type=str, default='unet_64',
                    help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_64 | unet_256 | unet_128]')
parser.add_argument('--isTrain', default=True, help='Train or Test')
parser.add_argument('--gan_mode', type=str, default='lsgan',
                    help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--lr_policy', type=str, default='linear',
                    help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50,
                    help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--model_dir', default='models', type=str)
parser.add_argument('--pretrainGA', default='models/pretrainGA.pth', type=str)
parser.add_argument('--pretrainGB', default='models/pretrainGB.pth', type=str)
parser.add_argument('--DNmodel_dir', default='../DN/models/unet64_basic_acgan_3_2_5_0_G/model_090.pth', type=str)

args = parser.parse_args()

data_realA = dg.realA_datagenerator(args.dataset_noise_free, args.batch_size, args.aug_times,
                                              args.patch_size, args.stride, args.threshold)
data_realB = dg.realB_datagenerator(args.phantom_low, args.batch_size, args.aug_times,
                                              args.patch_size, args.stride, args.threshold)

data_align = dg.aligned_datagenerator(args.phantom_low, args.phantom_high, args.batch_size, args.aug_times,
                                              args.patch_size, args.stride, args.threshold)

data_realA = data_realA.astype('float32') / 255.0
data_realB = data_realB.astype('float32') / 255.0
data_align = data_align.astype('float32') / 255.0
data_realA = torch.from_numpy(data_realA.transpose((0, 3, 1, 2)))
data_realB = torch.from_numpy(data_realB.transpose((0, 3, 1, 2)))
data_align = torch.from_numpy(data_align.transpose((0, 1, 4, 2, 3)))

DDataset = BoseDenoisingDataset(data_realA, data_realB, data_align)
DLoader = DataLoader(dataset=DDataset, num_workers=0, drop_last=True, batch_size=args.batch_size, shuffle=True)


model = CycleGANModel(args)
initial_time, initial_epoch = model.load_networks_teacher_stu(args)

if initial_time != 0:
    data_align = dg.aligned_datagenerator(args.dataset_noises, args.dataset_noise_free, args.batch_size, args.aug_times,
                                          args.patch_size, args.stride, args.threshold)
    data_align = data_align.astype('float32') / 255.0
    data_align = torch.from_numpy(data_align.transpose((0, 1, 4, 2, 3)))

    DDataset = BoseDenoisingDataset(data_realA, data_realB, data_align)
    DLoader = DataLoader(dataset=DDataset, num_workers=0, drop_last=True, batch_size=args.batch_size, shuffle=True)


for teach_time in range(initial_time, args.teach_nums):
    print('---------------- start %d times ----------------' % teach_time)
    if teach_time != initial_time:
        initial_epoch = 0
    print('---------------- Training labeled images ----------------')
    for epoch in range(initial_epoch, args.epoch):
        for i, data in enumerate(DLoader):
            model.set_input(data)
            model.optimize_parameters()
            print('--------------- phantom data ---------------')
            print('[%d/%d][%d/%d] Loss_DA: %.4f Loss_G: %.4f Loss_GA: %.4f Loss_GB: %.4f CycleA: %.4f CycleB: %.4f' % (
                epoch, args.epoch, i, len(DLoader), model.loss_D_A, model.loss_G, model.loss_G_A,
                model.loss_G_B, model.loss_cycle_A, model.loss_cycle_B))

        model.save_networks_teacher_stu(args, teach_time, epoch)
        model.update_learning_rate(epoch)
    print('---------------- Finish training labeled images ----------------')

    print("Generate pseudo labeled images")
    cuda = torch.cuda.is_available()
    # netG_A
    netG = networks.define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.norm, not args.no_dropout,
                             args.init_type, args.init_gain)
    if cuda:
        netG = netG.cuda()
        device_ids = [0]
        netG = nn.DataParallel(netG, device_ids=device_ids).cuda()

    G_A_dir = os.path.join(args.model_dir, args.model + '_G_A')
    netG.load_state_dict(torch.load(os.path.join(G_A_dir, 'model_%03d_%03d.pth' % (teach_time + 1, args.epoch))))
    log('load trained model times %03d epoch %03d' % (teach_time, args.epoch))

    netG.eval()  # evaluation mode

    if not os.path.exists(args.dataset_noises):
        os.makedirs(args.dataset_noises)

    for im in os.listdir(args.dataset_unlabeled):
        if im.endswith(".tif") or im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = cv2.imread(os.path.join(args.dataset_unlabeled, im), 0)
            pre_img = x
            height, weight = x.shape
            resize_h = round(height / 64) * 64
            resize_w = round(weight / 64) * 64
            x = cv2.resize(x, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            x = np.array(x, dtype=np.float32) / 255.0
            x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])

            # start_time = time.time()
            x_ = netG(x)  # inference
            x_ = x_.view(x_.shape[2], x_.shape[3])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float32)
            x_ = cv2.resize(x_, (weight, height), interpolation=cv2.INTER_LINEAR)

            x_[np.where(pre_img == 0)] = 0

            name, ext = os.path.splitext(im)
            save_result(x_, path=os.path.join(args.dataset_noises, name + ext))  # save the denoised image
    print("Finished pseudo labeled images")

    print("Re-prepare training dataset")
    data_align = dg.aligned_datagenerator(args.dataset_noises, args.dataset_noise_free, args.batch_size, args.aug_times,
                                          args.patch_size, args.stride, args.threshold)
    data_align = data_align.astype('float32') / 255.0
    data_align = torch.from_numpy(data_align.transpose((0, 1, 4, 2, 3)))

    DDataset = BoseDenoisingDataset(data_realA, data_realB, data_align)
    DLoader = DataLoader(dataset=DDataset, num_workers=0, drop_last=True, batch_size=args.batch_size, shuffle=True)

    print("Finished Re-prepare training dataset")

    print('---------------- finish %d times ----------------' % teach_time)
