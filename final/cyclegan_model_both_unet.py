# -*- coding: utf-8 -*-
# @Time    : 2020/2/23 15:54
# @Author  : xls56i

# cyclegan model Gs, Gd都是unet, loss一起算

import torch
import itertools
from image_pool import ImagePool
import networks
from torch.optim.lr_scheduler import MultiStepLR
import os
from util import findLastCheckpoint_time_epoch
import torch.nn as nn
import torch.nn.init as init


class CycleGANModel():

    def __init__(self, opt):

        super(CycleGANModel, self).__init__()

        self.opt = opt
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.netGs = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netGd = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netDs = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netDd = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

        self.criterionL1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netGs.parameters(), self.netGd.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netDs.parameters(), self.netDd.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.schedulerD = MultiStepLR(self.optimizer_D, milestones=[10, 20, 30, 40, 50], gamma=0.2)
        self.schedulerG = MultiStepLR(self.optimizer_G, milestones=[10, 20, 30, 40, 50], gamma=0.2)

    def update_learning_rate(self, epoch):
        self.schedulerD.step(epoch)
        self.schedulerG.step(epoch)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, input):
        self.real_A = input['noise_free'].to(self.device)  # 真实无噪声数据
        self.real_B = input['real_noise'].to(self.device)  # 真实有噪声数据
        self.align_noise = input['noise'].to(self.device)
        self.align_free = input['no-noise'].to(self.device)

    def save_networks_teacher_stu(self, opt, time, epoch):
        Gs_dir = os.path.join(opt.model_dir, opt.model + '_Gs')
        Gd_dir = os.path.join(opt.model_dir, opt.model + '_Gd')
        Ds_dir = os.path.join(opt.model_dir, opt.model + '_Ds')
        Dd_dir = os.path.join(opt.model_dir, opt.model + '_Dd')
        if not os.path.exists(Gs_dir):
            os.makedirs(Gs_dir)
        if not os.path.exists(Gd_dir):
            os.makedirs(Gd_dir)
        if not os.path.exists(Ds_dir):
            os.makedirs(Ds_dir)
        if not os.path.exists(Dd_dir):
            os.makedirs(Dd_dir)
        torch.save(self.netGs.state_dict(), os.path.join(Gs_dir, 'model_%03d_%03d.pth' % (time + 1, epoch + 1)))
        torch.save(self.netGd.state_dict(), os.path.join(Gd_dir, 'model_%03d_%03d.pth' % (time + 1, epoch + 1)))
        torch.save(self.netDs.state_dict(), os.path.join(Ds_dir, 'model_%03d_%03d.pth' % (time + 1, epoch + 1)))
        torch.save(self.netDd.state_dict(), os.path.join(Dd_dir, 'model_%03d_%03d.pth' % (time + 1, epoch + 1)))

    def load_networks_teacher_stu(self, opt):
        Gs_dir = os.path.join(opt.model_dir, opt.model + '_Gs')
        Gd_dir = os.path.join(opt.model_dir, opt.model + '_Gd')
        Ds_dir = os.path.join(opt.model_dir, opt.model + '_Ds')
        Dd_dir = os.path.join(opt.model_dir, opt.model + '_Dd')
        initial_time, initial_epoch = findLastCheckpoint_time_epoch(
            save_dir=Gs_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading times %03d epoch %03d' % (initial_time + 1, initial_epoch))
            self.netGs.load_state_dict(
                torch.load(os.path.join(Gs_dir, 'model_%03d_%03d.pth' % (initial_time + 1, initial_epoch))))
        else:
            print('resuming by loading pretrainGA.pth')
            self.netGs.load_state_dict(torch.load(opt.pretrainGA))

        initial_time, initial_epoch = findLastCheckpoint_time_epoch(
            save_dir=Gd_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading times %03d epoch %03d' % (initial_time + 1, initial_epoch))
            self.netGd.load_state_dict(
                torch.load(os.path.join(Gd_dir, 'model_%03d_%03d.pth' % (initial_time + 1, initial_epoch))))
        else:
            print('resuming by loading times pretrainGB.pth')
            self.netGd.load_state_dict(torch.load(opt.pretrainGB))

        initial_time, initial_epoch = findLastCheckpoint_time_epoch(
            save_dir=Ds_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading times %03d epoch %03d' % (initial_time + 1, initial_epoch))
            self.netDs.load_state_dict(
                torch.load(os.path.join(Ds_dir, 'model_%03d_%03d.pth' % (initial_time + 1, initial_epoch))))

        initial_time, initial_epoch = findLastCheckpoint_time_epoch(
            save_dir=Dd_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading times %03d epoch %03d' % (initial_time + 1, initial_epoch))
            self.netDd.load_state_dict(
                torch.load(os.path.join(Dd_dir, 'model_%03d_%03d.pth' % (initial_time + 1, initial_epoch))))
        print('times %03d epoch %03d' % (initial_time, initial_epoch))
        return initial_time, initial_epoch

    def forward(self):

        self.fake_B = self.netGs(self.real_A)  # Gs(A)
        self.rec_A = self.netGd(self.fake_B)  # Gd(Gs(A))
        self.fake_A = self.netGd(self.align_noise)  # Gd(B)
        self.rec_B = self.netGs(self.fake_A)  # Gs(Gd(B))

    def backward_Ddasic(self, netD, real, fake):

        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_Ds(self):
        """Calculate GAN loss for discriminator Ds"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_Ds = self.backward_Ddasic(self.netDs, self.real_B, fake_B)

    def backward_Dd(self):
        """Calculate GAN loss for discriminator Dd"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_Dd = self.backward_Ddasic(self.netDd, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators Gs and Gd"""
        lambda1 = 10
        lambda2 = 1
        lambda3 = 20
        lambda4 = 1
        lambda5 = 1

        self.loss_Gs = self.criterionGAN(self.netDs(self.fake_B), True) * lambda1  # (b)图 Ds Loss

        self.loss_Gd = self.criterionGAN(self.netDd(self.fake_A), True) * lambda2  # (c)图 Dd Loss

        self.loss_Gd_L1 = self.criterionL1(self.fake_A, self.align_free) * lambda3  # (c)图 L1 Loss

        self.loss_cycle_A = self.criterionL1(self.rec_A, self.real_A) * lambda4  # (b)图 L1 Loss

        self.loss_cycle_B = self.criterionGAN(self.netDs(self.rec_B), True) * lambda5  # (c)图 Ds Loss

        # combined loss and calculate gradients
        self.loss_G = self.loss_Gs + self.loss_Gd + self.loss_cycle_A + self.loss_cycle_B + self.loss_Gd_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # Gs and Gd
        self.set_requires_grad([self.netDs, self.netDd], False)
        self.optimizer_G.zero_grad()  # set Gs and Gd's gradients to zero
        self.backward_G()  # calculate gradients for Gs and Gd
        self.optimizer_G.step()  # update Gs and Gd's weights
        # Ds and Dd
        self.set_requires_grad([self.netDs, self.netDd], True)
        self.optimizer_D.zero_grad()  # set Ds and Dd's gradients to zero
        self.backward_Ds()  # calculate gradients for Ds
        self.backward_Dd()  # calculate graidents for Dd
        self.optimizer_D.step()  # update Ds and Dd's weights

