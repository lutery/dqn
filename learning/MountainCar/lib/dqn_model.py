import torch
import torch.nn as nn
import numpy as np

class DQNMountainCar(nn.Module):
    def __init__(self, input_shape, n_actions):
        '''
        param input_shape: 输入的状态形状，网路接受输入的shape形如(channel, width, height)
        param n_actions: 输出的所要执行的动作得到的激励，对于gym来说即action_space的在当前状态下得到的激励值
        '''

        super(DQNMountainCar, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(input_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
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