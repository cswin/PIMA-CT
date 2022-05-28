# LDNS
Low-dose CT noise simulation and denosing



## Requirement

```
python==3.6.9
pytorch==0.4.1
```


<!-- 
## self-learning

`LDNS/CycleGAN`目录下：

```python
# 根据gpu情况更改train_teacher_stu2.py和test_teacher_stu.py的参数
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
```



```shell
# 训练self-learning, 训练pseudo数据时使用L2_loss
python train_teacher_stu2.py --model teacher_stu_unet64_basic_60_unaligned_l2loss --self_training_loss 2
```

```shell
# 测试结果 结果保存在GA_results(simulation)和GB_results(denoised)
python test_teacher_stu.py --modelGA_dir models/teacher_stu_unet64_basic_60_unaligned_l2loss_G_A/model_005_060.pth --modelGB_dir models/teacher_stu_unet64_basic_60_unaligned_l2loss_G_B/model_005_060.pth --GA_results results/GAl2loss --GB_results results/GBl2loss
```



```shell
# 训练self-learning, 训练pseudo数据时使用L2_loss
python train_teacher_stu2.py --model teacher_stu_unet64_basic_60_unaligned_l1loss --self_training_loss 1
```

```shell
# 测试结果 结果保存在GA_results(simulation)和GB_results(denoised)
python test_teacher_stu.py --modelGA_dir models/teacher_stu_unet64_basic_60_unaligned_l1loss_G_A/model_005_060.pth --modelGB_dir models/teacher_stu_unet64_basic_60_unaligned_l1loss_G_B/model_005_060.pth --GA_results results/GAl1loss --GB_results results/GBl1loss
```





## Train DN



### Usage

```shell
# train
python train_acgan.py --model unet64_basic_acgan_4_1_5_0 --lambda0 0.4 --lambda1 0.1 --lambda2 0.5 --lambda3 0.0
```



```shell
# test
python test_infogan.py --model_dir models/unet64_basic_acgan_4_1_5_0_G --noise_level 60 --result_dir results/unet64_basic_acgan_4_1_5_0_60
python test_infogan.py --model_dir models/unet64_basic_acgan_4_1_5_0_G --noise_level 30 --result_dir results/unet64_basic_acgan_4_1_5_0_30
```



```shell
# train(no condition)
python train_gan.py --model unet64_basic_TVLoss_2_7_1 --lambda1 0.2 --lambda2 0.7 --lambda3 0.1
```



```shell
# test(no condition)
python test_gan.py --model_dir models/unet64_basic_TVLoss_2_7_1_G_175mAs-60mAs --result_dir results/unet64_basic_TVLoss_2_7_1_175-60
```



### data_generator_bg.py

用于从数据集生成成对的图像块

```python
def datagenerator(from_dir, to_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False)

# from_dir: 比如我们要训练175mAs到60mAs的网络，这个参数即为175mAs数据集的路径，源数据集路径
# to_dir: 目标数据集路径，60mAs
# batch_size: 训练时的batch_size大小，如果网络用到了batch normalization的话，需要删除一些数据使数据大小是batch_size的倍数
# aug_times: 图像块经过翻转，旋转的操作的次数，用来增加数据量
# patch_size: 图像块的大小
# stride: 选取图像块时的步长间隔
# threshold: 选取图像块时的背景所占最大比例，比如为0.1，则像素点为0的个数不能超过总像素点个数的10%
```



### info_data_generator.py

InfoGAN和ACGAN用于数据生成的程序，生成的数据为一个图像对和一个噪声级别

> [(175mAs, 60mAs), 60]  or  [(175mAs, 30mAs), 30]



### my loss.py

定义自己用到的损失函数

```python
class TVLoss(nn.Module) # TVLoss
class sum_squared_error(_Loss) # sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
class NormalNLLLoss(nn.Module) # InfoGAN中的一个损失函数，后面没有用到这个Loss
```



### networks.py

```python
def init_weights(net, init_type='normal', init_gain=0.02) # 初始化网络参数
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]) # 定义Generator
# netG = 'resnet_9blocks' or netG = 'resnet_6blocks' 使用resnet作为生成网络框架
# netG = 'unet' UNet下采样使用的是MAxPooling实现
# netG = 'unet64_visual' UNet将中间结果输出，用来可视化中间结果
# netG = 'unet_64' or 'unet_128' or 'unet_256' 使用kernel_size=4, stride=2来下采样，因为我们patch大小为64，所以没有用到unet_128和unet_256
def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]) # 定义Discriminator
# netD = 'basic' 输入为64*64的patch，输出为6*6的矩阵
# netD = 'basic_condition' 输入为64*64的patch, 输出为两个6*6的矩阵，一个用来判别真假，一个用来判别噪声级别
# netD = 'QHead' or 'FE_infogan' or 'DHead' 在InfoGAN时使用，但是本质上和basic_condition差不多
```



### test_gan.py

```python
# 参数说明
parser.add_argument('--set_dir', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str, help='directory of test dataset')
parser.add_argument('--model_dir', default=os.path.join('models', 'unet64_basic_only_advloss_G_175mAs-60mAs'), help='directory of the model') # 之前训练保存的网络参数地址
parser.add_argument('--model_name', default='model_090.pth', type=str, help='the model name') # 使用第几个epoch的网络参数
parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset') # 生成的图片保存的地址位置
parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
```





### train_acgan.py

```python
# 参数说明
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--model', default='unet64_basic_acgan', type=str, help='choose a type of model') # 使用的模型名字，根据自己定义的生成器和判别器来命名
parser.add_argument('--lambda0', default=0.1, type=float, help='noise_level loss') # noise_level loss 的权重
parser.add_argument('--lambda1', default=0.1, type=float, help='adver loss') # 判断真假的loss的权重
parser.add_argument('--lambda2', default=0.8, type=float, help='rec loss') # 判断生成的噪声图像和真实的成对有噪声图像差别的L2_loss权重
parser.add_argument('--lambda3', default=0.0, type=float, help='TV loss') # TV loss的权重
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')
parser.add_argument('--from_does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str,
                    help='path of high-does data')
parser.add_argument('--to_60does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str,
                    help='path of low-does data')
parser.add_argument('--to_30does', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_30mAs', type=str,
                    help='path of low-does data')
parser.add_argument('--threshold', default=0.1, type=float, help='background threshold')
parser.add_argument('--epoch', default=90, type=int, help='number of train epoches')
```



```python
errG0 = criterion(output[1], level_map) # noise_level loss
errG1 = criterion(output[0], label) # adver loss
errG2 = criterion(fake, real_cpu) / ((fake.shape[0] / output[0].shape[0]) ** 2) # rec loss 后面除以的数是因为，errG0和errG1计算loss时是在6*6的大小上计算的L2 loss, 而errG2和errG3是在64*64的大小上计算的，所以除以了一下面积大小比
errG3 = criterion1(fake) / (2 * (fake.shape[0] / output[0].shape[0]) ** 2) # TV loss
loss_G = args.lambda0 * errG0 + args.lambda1 * errG1 + args.lambda2 * errG2 + args.lambda3 * errG3
```



### train_gan.py

使用没有使用condition的网络进行训练



### train_infoGAN.py

使用InfoGAN进行训练，和ACGAN的不同是，InfoGAN的判别器输出是一个判别真假，一个判别类别是60mAs还是30mAs，相当于两个分类任务，ACGAN的输出判别是60mAs还是30mAs的时候是一个回归任务



### util.py

```python
def findLastCheckpoint(save_dir) # 找到最近训练得到的网络参数，从上次的训练恢复
def save_result(result, path) # 保存生成的图像的结果
def noise_sample(high_dose, noise_level) # 在InfoGAN和ACGAN中，输入把high_does和noise_level叠加
```



### visual.py

没有使用condition时的可视化测试代码

### visual_unet64.py

没有使用condition时的可视化训练代码



## Train CycleGAN



### Usage

```shell
# train
python train_cyclegan.py
```



### cycle_data_generator.py

用于CycleGAN的数据生成，一个对应于成对的数据，一个对应于不是成对的数据

```python
def aligned_datagenerator(from_dir, to_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False)
class AlignedDenoisingDataset(Dataset)

def unaligned_datagenerator(from_dir, to_dir, batch_size, aug_times, patch_size, stride, threshold, verbose=False)
class UnAlignedDenoisingDataset(Dataset)
```



### cyclegan_model.py

用来定义CycleGAN模型

```python
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

# self.loss_idt_A 和 self.loss_idt_B是论文中提到，比如我要由通过G_A实现A-->B，我们也要确保当输入为B的时候，输出也要为B
```



### train_cyclegan.py

```python
# 参数说明
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--model', default='unet64_basic', type=str, help='choose a type of model') # 使用的模型名字，根据自己定义的生成器和判别器来命名
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--aug_times', default=3, type=int, help='aug times')
parser.add_argument('--patch_size', default=64, type=int, help='patch size')
parser.add_argument('--stride', default=20, type=int, help='stride')
parser.add_argument('--datasetA', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str, help='path of high-dose data') # 数据集A的路径，real high-dose
parser.add_argument('--datasetB', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_60mAs', type=str, help='path of low-dose data') # 数据集B的路径，real low-dose
parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images') # image_pool.py 使用的大小
parser.add_argument('--threshold', default=0.1, type=float, help='background threshold')
parser.add_argument('--epoch', default=90, type=int, help='number of train epoches')
parser.add_argument('--datatype', default='aligned', type=str, help='datatype: aligned or unaligned') # 使用的数据集A和B是否是成对的
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale') # 输入图片通道数
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale') # 输出图片通道数
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', default=True,action='store_true', help='no dropout for the generator')
parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
parser.add_argument('--lambda_DN', type=float, default=10.0, help='weight for DN(A) and fakeB')
parser.add_argument('--noise_level', type=float, default=0.34, help='noise level, default:60/175') # 噪声级别，因为DN网络的输入要有一个噪声级别参数，这个参数应该与训练CycleGAN时所用的low-dose的噪声级别一样
parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. n_layers allows you to specify the layers in the discriminator')
parser.add_argument('--netG', type=str, default='unet_64', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_64 | unet_256 | unet_128]')
parser.add_argument('--isTrain', default=True, help='Train or Test')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.') # 
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]') # 更新学习率的方法，代码没有用到这个参数
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations') # 多少个epoch更新一次学习率，这个参数也没有用到
parser.add_argument('--model_dir', default='models', type=str) # 模型网络参数存储路径
parser.add_argument('--DNmodel_dir', default='../DN/models/unet64_basic_acgan_4_1_5_0_G/model_090.pth', type=str) # DN网络参数的路径
```



### test_cyclegan.py

```python
parser.add_argument('--set_dir', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_30mAs', type=str, help='directory of test dataset') # 测试数据集路径
parser.add_argument('--model_dir', default=os.path.join('models', 'unet64_basic_G_B'), help='directory of the model:G_A==>high2low,simulate, G_B==>low2high,denoising') # 使用的生成器模型路径
parser.add_argument('--model_name', default='model_090.pth', type=str, help='the model name') # 生成器网络参数名字
parser.add_argument('--isTrain', default=False, help='Train or Test')
parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset') # 生成结果存储路径
```



### image_pool.py

是为了实现，当我训练Discriminator的时候，使用的数据除了当前通过Generator生成的，还要加一部分之前生成过的数据

### networks.py & util.py

和DN中的一样 -->

