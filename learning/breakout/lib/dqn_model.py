import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class NoisyLinear(nn.Linear):
    """
    噪音全连接层，就是利用深度神经网络，自己去学习合适的噪声强度在探索和自主选择之间进行权衡

    sigma_weight和sigma_bias是网络中的可学习参数，它们表示权重和偏置的噪声强度。通过训练，网络可以学习合适的噪声强度。epsilon_weight和epsilon_bias是从标准正态分布中采样的随机噪声，它们在每次前向传播时都会发生变化

    sigma_weight和sigma_bias是表示权重和偏置中噪声的强度的可学习参数。它们是网络的一部分，并随着网络的训练进行更新。sigma_weight和sigma_bias的初始值设置为一个较小的常数，如0.017。它们的值与权重和偏置中的噪声成正比。

epsilon_weight和epsilon_bias表示从标准正态分布中采样的随机噪声。它们与sigma_weight和sigma_bias相乘，然后将结果添加到原始的权重和偏置中，从而在前向传播过程中引入噪声。epsilon_weight和epsilon_bias不是可学习参数，它们在每次前向传播时被重新采样。

如果省略epsilon_weight和epsilon_bias，网络将失去引入随机噪声的能力。在这种情况下，网络将不再是有噪网络，而是一个普通的DQN。有噪网络可以增强探索能力，因此，省略epsilon_weight和epsilon_bias可能会导致性能下降。为了保持增强探索的优势，我们需要保留epsilon_weight和epsilon_bias。
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        """
        sigma_init: 采用0.017是因为Noisy Network的论文测试得来的
        """
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # 注册模型的权重
        # register_buffer注册的参数不会自动更新，只能认为的进行重置更新，但是在保存、加载模型
        # 时却可以正常的保存和加载https://www.jianshu.com/p/12a8207149b0
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            # 存在偏置则注册模型的偏置
            # 如果省略epsilon_weight和epsilon_bias，只保留sigma_weight和sigma_bias，则网络将无法引入随机噪声。这样，网络将不具备探索能力，可能无法很好地学习。
            # 如果省略掉sigma_weight和sigma_bias，只保留epsilon_weight和epsilon_bias，则网络会在每次前向传播时引入固定强度的随机噪声。这种情况下，网络可能仍然具有一定的探索能力，但由于噪声强度无法通过训练进行调整，网络的表现可能不如完整的有噪网络。
            # 总之，为了在探索和利用之间取得平衡，最好同时使用sigma_weight、sigma_bias、epsilon_weight和epsilon_bias，让网络能够学习合适的噪声强度并在前向传播中引入随机噪声。
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        # 重置全连接层的参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置参数
        """
        # 设置方差的区域范围，根据这个方差的范围重置全连接层的权重和方差
        # 这里采用这种方式进行std的设置也是根据Noisy Network的论文得来的
        # todo 阅读Noisy Network的论文
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        """
        向前推理
        """
        # epsilon_weight在每次前向推理时，都会从一个标准正太分布中
        # 重新采样出新值
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            # epsilon_bias如果存在偏置则偏置也进行正太分布重置初始化
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        # 调用全连接层的函数进行运算，但是偏置和权重都加入了噪声干扰
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)

class DQNMountainCar(nn.Module):
    def __init__(self, input_shape, n_actions):
        '''
        param input_shape: 输入的状态形状，网路接受输入的shape形如(channel, width, height)
        param n_actions: 输出的所要执行的动作得到的激励，对于gym来说即action_space的在当前状态下得到的激励值
        '''

        super(DQNMountainCar, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(input_shape[0], 40),
            nn.ReLU(),
            nn.Linear(40, n_actions)
        )

    def forward(self, x):
        return self.seq(x)

class DQNBreakOut(nn.Module):
    def __init__(self, input_shape, n_actions):
        '''
        param input_shape: 输入的状态形状，网路接受输入的shape形如(channel, width, height)
        param n_actions: 输出的所要执行的动作得到的激励，对于gym来说即action_space的在当前状态下得到的激励值
        '''

        super(DQNBreakOut, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        '''
        获取卷积删除的所有维度大小，便于给之后的全连接层
        '''

        o = self.conv(torch.zeros(1, *shape))
        # np.prod 计算所有维度的乘积
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)