## cyclegan_model.py中和论文算法图的对应关系

```python
def forward(self):

        self.fake_B = self.netGs(self.real_A)  # Gs(A)
        self.rec_A = self.netGd(self.fake_B)  # Gd(Gs(A))
        self.fake_A = self.netGd(self.align_noise)  # Gd(B)
        self.rec_B = self.netGs(self.fake_A)  # Gs(Gd(B))
```



```python
def backward_G(self):
        """Calculate the loss for generators Gs and Gd"""
        lambda1 = 1
        lambda2 = 1
        lambda3 = 10
        lambda4 = 10
        lambda5 = 1

        self.loss_Gs = self.criterionGAN(self.netDs(self.fake_B), True) * lambda1  # (b)图 Ds Loss

        self.loss_Gd = self.criterionGAN(self.netDd(self.fake_A), True) * lambda2  # (c)图 Dd Loss

        self.loss_Gd_L1 = self.criterionL1(self.fake_A, self.align_free) * lambda3  # (c)图 L1 Loss

        self.loss_cycle_A = self.criterionL1(self.rec_A, self.real_A) * lambda4  # (b)图 L1 Loss

        self.loss_cycle_B = self.criterionGAN(self.netDs(self.rec_B), True) * lambda5  # (c)图 Ds Loss

        # combined loss and calculate gradients
        self.loss_G = self.loss_Gs + self.loss_Gd + self.loss_cycle_A + self.loss_cycle_B + self.loss_Gd_L1
        self.loss_G.backward()
```

![WechatIMG546](/Users/uvo9ono/Desktop/final/WechatIMG546.jpeg) 



## 终端输出的loss解释

![截屏2020-01-1819.56.10](/Users/uvo9ono/Desktop/final/截屏2020-01-1819.56.10.png)

- 前两列$\lambda_1$ 和$\lambda_5$ ，因为b图和c图的$D_s$ 是同一个，两个$loss$ 分别对应的是直接判别由干净图片生成的噪声图片的$loss \lambda_1$ 和 由噪声图片经过去噪和加噪循环生成的噪声图片的$loss \lambda_2$ 
- 第三列的$\lambda_2$ 对应的是$D_d$判别去噪的$loss \lambda3$ 
- 第四和第五列$\lambda_4$ 和$\lambda_3$ ,对应的是由干净图片经过加噪再去噪循环的$\lambda4$ $L_1 loss$ 和直接由噪声图片去噪的$\lambda_3$  $L_1 loss$ 
- 最后一列$total$ 对应的是前五列的加和
- **我们训练table2的实验时，为了使最终结果psnr大，最终目的应该是使**$\lambda_3$ $L_1 loss$  **越小越好**



## train_DnCNNB.py输入参数解释

```python
parser.add_argument('--model', default='DnCNNB-simulation-30mAs', type=str, help='choose a type of model')

parser.add_argument('--dataset_unlabeled', default='../dataset/CT_Data_All_Patients/train', type=str, help='noise free for patient')
parser.add_argument('--dataset_noise_free', default='../dataset/noise_free', type=str,
                    help='noise_free patient and phantom high')
parser.add_argument('--dataset_noises', default='../dataset/noises30mAs', type=str, help='phantom low and patient simulation')
parser.add_argument('--phantom_low', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_30mAs', type=str,
                    help='path of low-dose data')
parser.add_argument('--phantom_high', default='../dataset/phantom/Head_05_VOLUME_4D_CBP_Dynamic_175mAs', type=str,
                    help='path of high-dose data')

parser.add_argument('--epoch', default=30, type=int, help='number of train epoches')
parser.add_argument('--teach_nums', default=3, type=int, help='number of teach times')
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model_dir', default='models', type=str)
parser.add_argument('--pretrainGA', default='models/pretrainGA.pth', type=str)
parser.add_argument('--pretrainGB', default='models/model_044.pth', type=str)
```

- 输入参数主要是改这几个，其余参数不用更改，其中最重要的参数是5个数据的路径
- dataset_unlabeled是病人的清晰数据路径
- dataset_noise_free是病人的清晰数据和phantom175mAs的路径（为了输入处理方便，把这两种图片放到了一个文件夹下，其对应的图片中的数据为real_A或align_free）
- dataset_noises是phantom low的数据和simulation出来的病人数据（同样为了输入处理方便，把这两种图片放到了一个文件夹下，对应图中的align_noise），这里注意，这个文件夹下最初只有phantom low的数据，因为还没有训练，就没有simulation，然后经过teach之后，才会生成伪标签的simulation数据，并且每次teach之后生成的新的伪标签数据会把原来的覆盖
- phantom_low 对应的是图中的realB
- phantom_high对应的是realB所对应的paired的清晰数据



**所以当我们训练table2的时候，应该只使用病人的simulated数据和清晰数据，因为在我们simulated的数据上做的对比实验DnCNN就是那只使用了这两个数据，这样的话，以上5个数据路径对应应该为**

-  dataset_unlabeled是病人的清晰数据路径
- dataset_noise_free是病人的清晰数据路径
- dataset_noises是simulation出来的病人数据，**这里注意之前因为phantom数据和病人数据文件名不同，所以新生成的伪标签数据对phantom数据一直没有影响，但是现在这个文件夹里只有病人的数据，且文件名相同，所以teach后生成的伪标签数据会把最开始的simulated的数据，即我们用做训练集的数据覆盖掉，所以这个代码训练之后，用来测试的模型应该使用model_001_xxx.pth，解决这个问题的话，只能是新建一个文件夹用来单独存储伪标签数据，然后再分别读取再整合，使用train_DnCNNB1.py这个代码即可**
- phantom_low 是simulation出来的病人数据
- phantom_high是病人的清晰数据路径

## train_DnCNNB2.py和cyclegan_model1.py
分别算两次loss，并更改了学习率

