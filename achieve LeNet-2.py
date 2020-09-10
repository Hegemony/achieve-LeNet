import torch.nn as nn
import torch.nn.functional as F


# nn中是定义的类，以class xx来定义的，可以提取变化的学习参数。
# nn.functional中的是函数，由def function( )定义，是一个固定的运算公式。

# 自定义层的步骤
# （1）自定义一个类，继承自Module类，并且一定要实现两个基本的函数，
#  第一是构造函数__init__，第二个是层的逻辑运算函数，即所谓的前向计算函数forward函数。
# （2）在构造函数_init__中实现层的参数定义。比如Linear层的权重和偏置，
#  Conv2d层的in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode='zeros'
#  这一系列参数；
# （3）在前向传播forward函数里面实现前向运算。
#  这一般都是通过torch.nn.functional.***函数来实现，当然很多时候我们也需要自定义自己的运算方式。
#  如果该层含有权重，那么权重必须是nn.Parameter类型，简单说就是Parameter默认需要求导，其他两个类型则不会。
#  另外一般情况下，可能的话，为自己定义的新层提供默认的参数初始化，以防使用过程中忘记初始化操作。
# （4）补充：一般情况下，我们定义的参数是可以求导的，但是自定义操作如不可导，需要实现backward函数。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 调用父类的构造函数
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Torch.nn.Conv2d(in_channels，out_channels，kernel_size，
        # stride=1，padding=0，dilation=1，groups=1，bias=True)
        # in_channels：输入维度
        # out_channels：输出维度（卷积核个数）6
        # kernel_size：卷积核大小 5*5
        # stride：步长大小，默认为1
        # padding：补0的层数，padding=1则每条边补一层0
        # dilation：kernel间距
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 全连接层，y=wx+b
        # 输入图片大小 W*W
        # Filter（卷积核）大小F*F
        # 步长 Step
        # padding（填充）的像素数P，P=1就相当于给图像填充后图像大小为W+1 *W+1
        # 输出图片的大小为N * N
        # N = [(W-F+2P)/Step]+1

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 池化的参数：
        #  input – 输入的张量 (minibatch x in_channels x iH x iW) -
        #  kernel_size – 池化区域的大小，可以是单个数字或者元组 (kh x kw) -
        #  stride – 池化操作的步长，可以是单个数字或者元组 (sh x sw)。默认等于核的大小 -
        #  padding – 在输入上隐式的零填充，可以是单个数字或者一个元组 (padh x padw)，默认: 0 -
        #  ceil_mode – 定义空间输出形状的操作 -
        #  count_include_pad – 除以原始非填充图像内的元素数量或kh * kw
        x = x.view(x.size()[0], -1)
        # reshape ,'-1' 表示自适应
        # x.view(x.size()[0], -1)这句话是说将输出Tensor拉伸成一维
        # x.size()[0]是batch size，比如原来的数据一共12个，
        # batch size为2，就会view成2*6，batch size为4，就会就会view成4*3
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 当模型有可学习的参数时，最好使用nn.Module,否则既可以使用
        # nn.functional也可以使用nn.Module。由于激活函数（RELU,
        # sigmoid,tanh）、池化（MaxPool）等层没有可学习的参数，可以
        # 使用对应的functional函数代替，而卷积、全连接层等具有可学习
        # 参数的网络建议使用nn.Module。但是建议使用nn.Dropout而不是
        # nn.functional.dropout,因为dropput在训练和测试阶段有区别：
        # model.train() ：启用 BatchNormalization 和 Dropout
        # model.eval() ：不启用 BatchNormalization 和 Dropout

        return x


net = Net()
print(net)

# 网络学习的参数可通过net.parameters()返回，
# net.named_parameters()可同时返回学习的参数和名称
# self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
# torch.nn.Parameter()函数，含义是将一个固定不可训练的tensor转换成可以训练
# 的类型parameter，并将这个parameter绑定到这个module里面
# (net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
# 所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
# 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

params = list(net.parameters())
print(len(params))

for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

# LeNet的input为32*32


# # pytorch保存模型的方法
# # 保存模型
# t.save(net.state_dict(),'net.pth')
# # 加载已保存的模型
# net2 = Net()
# net2.load_state_dict(t.load('net.pth'))


# # 将Module放在GPU上运行，只需要两个步骤
# model = model.cuda() # 将模型的所有参数转存到GPU
# input.cuda() # 将输入数据放置到GPU上
# # 在GPU上并行计算，Pytorch提供两个函数：
# nn.parallel.data_parallel(module,inputs,device_ids = None ,output_device = None , dim = 0,module_kwargs = None)
# class torch.nn.DataParallel(module,device_ids = None,output_device = None,dim = 0)
# # device_ids参数可以指定在哪些GPU上进行优化，output_device指定输出到哪个GPU上
# # 唯一的不同在于前者直接利用多GPU并行计算得出结果，后者则返回一个新的module,能够自动在多GPU上进行并行加速
# # DataParallel并行的方式，是将输入一个batch的数据均分成多份，分别送到对应的GPU中进行计算，
# # 然后将各个GPU得到的梯度相加。与Module相关的所有数据也会复制成多份。
