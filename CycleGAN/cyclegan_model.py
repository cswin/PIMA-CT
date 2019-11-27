# -*- coding: utf-8 -*-
# @Time    : 2019/11/26 13:51
# @Author  : xls56i

import torch
import itertools
from image_pool import ImagePool
from DN import networks
from torch.optim.lr_scheduler import MultiStepLR
import os
from DN.util import findLastCheckpoint

class CycleGANModel():

    def __init__(self, opt):

        super(CycleGANModel, self).__init__()

        self.opt = opt
        self.isTrain = opt.isTrain
        self.noise_level = opt.noise_level
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

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netDN = networks.define_G(2, 1, 64, 'unet_64')
        # print(self.netG_A)
        # print(self.netG_B)
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # print(self.netD_A)
            # print(self.netD_B)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionDN = torch.nn.MSELoss()
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.schedulerD = MultiStepLR(self.optimizer_D, milestones=[30, 60, 90], gamma=0.2)
            self.schedulerG = MultiStepLR(self.optimizer_G, milestones=[30, 60, 90], gamma=0.2)

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
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def save_networks(self, opt, epoch):
        G_A_dir = os.path.join(opt.model_dir, opt.model+'_G_A')
        G_B_dir = os.path.join(opt.model_dir, opt.model+'_G_B')
        D_A_dir = os.path.join(opt.model_dir, opt.model+'_D_A')
        D_B_dir = os.path.join(opt.model_dir, opt.model+'_D_A')
        torch.save(self.netG_A.state_dict(), os.path.join(G_A_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(self.netG_B.state_dict(), os.path.join(G_B_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(self.netD_A.state_dict(), os.path.join(D_A_dir, 'model_%03d.pth' % (epoch + 1)))
        torch.save(self.netD_B.state_dict(), os.path.join(D_B_dir, 'model_%03d.pth' % (epoch + 1)))

    def load_networks(self, opt):
        G_A_dir = os.path.join(opt.model_dir, opt.model + '_G_A')
        G_B_dir = os.path.join(opt.model_dir, opt.model + '_G_B')
        D_A_dir = os.path.join(opt.model_dir, opt.model + '_D_A')
        D_B_dir = os.path.join(opt.model_dir, opt.model + '_D_A')
        initial_epoch = findLastCheckpoint(save_dir=G_A_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading epoch %03d' % initial_epoch)
            self.netG_A.load_state_dict(torch.load(os.path.join(G_A_dir, 'model_%03d.pth' % initial_epoch)))

        initial_epoch = findLastCheckpoint(save_dir=G_B_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading epoch %03d' % initial_epoch)
            self.netG_B.load_state_dict(torch.load(os.path.join(G_B_dir, 'model_%03d.pth' % initial_epoch)))

        initial_epoch = findLastCheckpoint(save_dir=D_A_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading epoch %03d' % initial_epoch)
            self.netD_A.load_state_dict(torch.load(os.path.join(D_A_dir, 'model_%03d.pth' % initial_epoch)))

        initial_epoch = findLastCheckpoint(save_dir=D_B_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading epoch %03d' % initial_epoch)
            self.netD_B.load_state_dict(torch.load(os.path.join(D_B_dir, 'model_%03d.pth' % initial_epoch)))

        #load DN network
        self.netDN.load_state_dict(torch.load(opt.DNmodel_dir))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        N, C, H, W = self.real_A.size()
        noise_sigma = torch.FloatTensor([self.noise_level for i in range(N)])
        noise_map = noise_sigma.view(N, 1, 1, 1).repeat(1, C, H, W)
        self.dn_realA = torch.cat((self.real_A, noise_map), 1)

        self.dn_A = self.netDN(self.dn_realA)     # DN(A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
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

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_DN = self.opt.lambda_DN
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss L2(DN(A), G_A(A))
        self.loss_DN = self.criterionDN(self.dn_A, self.fake_B) * lambda_DN
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_DN
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netDN], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights