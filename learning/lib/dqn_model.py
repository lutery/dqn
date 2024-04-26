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
            nn.Linear(input_shape[0], 40),
            nn.ReLU(),
            nn.Linear(40, n_actions)
        )

    def forward(self, x):
        return self.seq(x)
