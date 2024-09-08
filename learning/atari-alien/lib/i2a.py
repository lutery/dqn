import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common

ROLLOUT_HIDDEN = 256

# 环境模型输出的预测下一帧的shape
EM_OUT_SHAPE = (1, ) + common.IMG_SHAPE[1:]


class EnvironmentModel(nn.Module):
    '''
    todo Environment model that predicts the next frame and reward
        环境模型，用来预测下一帧和奖励
    '''
    def __init__(self, input_shape, n_actions):
        '''
        param input_shape: 环境的观测空间维度
        param n_actions: 环境的所能执行动作的维度
        '''
        super(EnvironmentModel, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions

        # input color planes will be equal to frames plus one-hot encoded actions
        # 输入颜色平面将等于帧加上独热编码动作，啥意思
        # 这里是将当前的环境状态和动作平面进行拼接，得到一个新的输入，所以这里计算的是这种方式下的输入平面通道数，根据forward函数可以看出通道数6
        n_planes = input_shape[0] + n_actions
        # todo 通道尺寸的变化
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_planes, 64, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # output is one single frame with delta from the current frame
        # todo 这里的反卷积应该是预测下一帧的图像，也就下一次观测的环境状态
        self.deconv = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4, padding=0)

        # todo 这里的回报卷积应该是预测下一帧的奖励
        self.reward_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        # todo 这个应该是获取奖励卷积输出的尺寸，用于后续的全连接层
        rw_conv_out = self._get_reward_conv_out((n_planes, ) + input_shape[1:])
        # todo 预测回报奖励？是当前的还是下一帧的？
        self.reward_fc = nn.Sequential(
            nn.Linear(rw_conv_out, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_reward_conv_out(self, shape):
        '''
        根据输入的shape获取奖励卷积的输出尺寸
        然后将奖励的尺寸展平为一维得到全连接层的尺寸大小
        '''
        o = self.conv1(torch.zeros(1, *shape))
        o = self.reward_conv(o)
        return int(np.prod(o.size()))

    def forward(self, imgs, actions):
        '''
        param imgs: 当前的环境状态
        param actions: 当前的执行的动作

        return 得到预测得到的下一帧和奖励
        '''
        batch_size = actions.size()[0] # 当前训练的batch_size
        # 创建一个全零的动作平面，shape为(batch_size, n_actions, *input_shape[1:]) 在这次的代理里面里面应该是(batch_size, 4, 84, 84)
        act_planes_v = torch.FloatTensor(batch_size, self.n_actions, *self.input_shape[1:]).zero_().to(actions.device)
        # 由于当前环境执行的动作是一个值（即1或0或2这种），所以以下动作是将所有执行动作的平面
        # 设置为1.0，也就是说当前执行的动作的平面是1.0，其他的动作平面是0.0
        act_planes_v[range(batch_size), actions] = 1.0
        # 将当前的环境状态和动作平面进行拼接，得到一个新的输入
        # 其拼接的维度是在第二维度拼接的，由于imgs的维度是(batch_size, 2, 84, 84)
        # 所以act_planes_v的维度是(batch_size, 4, 84, 84)
        # 所以其拼接后的维度是(batch_size, 6, 84, 84)
        # todo 为什么要这样拼接？
        comb_input_v = torch.cat((imgs, act_planes_v), dim=1)
        # 采集特征
        c1_out = self.conv1(comb_input_v)
        # 进一步采集特征
        c2_out = self.conv2(c1_out)
        # 将两次卷积的结果相加
        c2_out += c1_out
        # 进行一次反卷积，预测和当前状态的差异
        img_out = self.deconv(c2_out)
        # 根据采集到的特征，预测奖励，并将结果铺平
        rew_conv = self.reward_conv(c2_out).view(batch_size, -1)
        # 将铺平的结果进行全连接，得到奖励
        rew_out = self.reward_fc(rew_conv)
        return img_out, rew_out


class RolloutEncoder(nn.Module):
    '''
    todo 作用
    '''
    def __init__(self, input_shape, hidden_size=ROLLOUT_HIDDEN):
        '''
        param input_shape: 环境的观测空间一帧维度（1， 84 ， 84）
        '''

        super(RolloutEncoder, self).__init__()

        # 采集环境观察的特征
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 获取特征特征的维度尺寸
        conv_out_size = self._get_conv_out(input_shape)

        # todo 作用和意义，为什么要使用rnn
        # 这里+1是因为要将回报奖励拼接进输入中
        self.rnn = nn.LSTM(input_size=conv_out_size+1, hidden_size=hidden_size, batch_first=False)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs_v, reward_v):
        """
        Input is in (time, batch, *) order
        输入的是已经采集到的未来观察状态数据以及对应的奖励
        param obs_v: 采集到的未来观察状态数据（每一步都是和上一帧的差异） 维度（time, batch, 1, 84, 84）
        param reward_v: 采集到的未来观察状态数据对应的奖励(time, batch, 1)
        """
        n_time = obs_v.size()[0] # 提取未来n步
        n_batch = obs_v.size()[1] # 提取每一步采集多少数据
        n_items = n_time * n_batch # 未来n步采集的数据总量
        obs_flat_v = obs_v.view(n_items, *obs_v.size()[2:]) # 展平time和batch维度，得到的维度是(n_items, 1, 84, 84)
        # 将展平后的数据传入卷积层采集特征
        conv_out = self.conv(obs_flat_v)
        # 再将特征的维度还原为（n_time, n_batch, ....）
        conv_out = conv_out.view(n_time, n_batch, -1)
        # 将回报奖励拼接到采集到的环境差异特征中
        rnn_in = torch.cat((conv_out, reward_v), dim=2)
        # todo 作用 这里拿到的隐藏层状态
        # todo rnn_hid shape is （1, n_batch, hidden_size）
        _, (rnn_hid, _) = self.rnn(rnn_in)
        # 这里应该是把隐藏层展平
        return rnn_hid.view(-1)


class I2A(nn.Module):
    def __init__(self, input_shape, n_actions, net_em, net_policy, rollout_steps):
        '''
        param input_shape: 环境的观测空间维度
        param n_actions: 环境的所能执行动作的维度
        param net_em: 环境模型
        param net_policy: 动作策略网络
        param rollout_steps: 预测的未来步数
        '''

        super(I2A, self).__init__()

        self.n_actions = n_actions
        self.rollout_steps = rollout_steps

        # 采集环境状态的特征
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        #获取特征的全连接维度
        conv_out_size = self._get_conv_out(input_shape)
        # todo 全连接层的输入维度，采集特征的维度，预测的未来步数*动作的数量
        fc_input = conv_out_size + ROLLOUT_HIDDEN * n_actions

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 512),
            nn.ReLU()
        )
        # 预测的动作
        self.policy = nn.Linear(512, n_actions)
        # 预测的回报
        self.value = nn.Linear(512, 1)

        # used for rollouts todo 未来步数的编码器？
        self.encoder = RolloutEncoder(EM_OUT_SHAPE)
        # 概率动作选择器
        self.action_selector = ptan.actions.ProbabilityActionSelector()
        # save refs without registering
        # 保存环境模型的引用以及动作策略网络的引用
        # A2C策略网络以及环境模型不参与训练，这两个主要是用来进行未来预测
        # 将未来预测的结果，提供给I2A参考判断
        object.__setattr__(self, "net_em", net_em)
        object.__setattr__(self, "net_policy", net_policy)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        '''
        param x: 采集到的环境的观测数据

        return （执行的动作，执行动作的回报）
        '''
        fx = x.float() / 255 # 归一化
        # 将采集到的环境观测数据传入到未来步数的编码器中，拿到未来所有步数以及对应奖励的特征
        enc_rollouts = self.rollouts_batch(fx)
        # 将采集到的环境观测数据传入到卷积层中，拿到环境观测数据的特征
        # 在展平为(batch_size, -1)
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        # 将环境观测数据的特征和预测的未来步数的特征拼接
        fc_in = torch.cat((conv_out, enc_rollouts), dim=1)
        # 通过全连接层采集拼接后的特征
        fc_out = self.fc(fc_in)
        # 预测下一步的动作和回报
        return self.policy(fc_out), self.value(fc_out)

    def rollouts_batch(self, batch):
        '''
        param batch: 环境的观察数据，维度为(batch_size, 2, 84, 84)

        return: 每个环境观察数据通过尝试所有的动作后采集到的未来路径特征， shape is （batch_size， hidden_size*nactions）
        '''
        batch_size = batch.size()[0]
        batch_rest = batch.size()[1:]
        # 以下if else的动作主要是为了实现后续预测未来步数时，是将当前环境所能执行的
        # 每个动作都传入到环境模型中，所以这里必须要将当前的状态状态数据扩展为和actions份
        # 这种主要用于预测的第一步，后续就是用网络预测下一次要执行的动作
        if batch_size == 1:
            # 如果batch_size为1，那么就将其扩展为n_actions个
            # 维度为(batch_size*n_actions, 2, 84, 84)
            obs_batch_v = batch.expand(batch_size * self.n_actions, *batch_rest)
        else:
            # 在第一个维度增加1个维度
            # (batch_size, 1, 2, 84, 84)
            obs_batch_v = batch.unsqueeze(1)
            # 则扩展变成(batch_size, n_actions, 2, 84, 84)
            # 关于expand的用法可以参考笔记
            obs_batch_v = obs_batch_v.expand(batch_size, self.n_actions, *batch_rest)
            # contiguous将存储空间连续化
            # 使用view将其变成(batch_size*n_actions, 2, 84, 84)
            obs_batch_v = obs_batch_v.contiguous().view(-1, *batch_rest)
        # np.arange(0, self.n_actions, dtype=np.int64) = [0, 1, 2, 3]
        # np.tile(np.arange(0, self.n_actions, dtype=np.int64), batch_size) = [0, 1, 2, 3, 0, 1, 2, 3, ...]
        # 这里复制动作的作用是后续进行未来步预测时，需要传入环境所有能执行的动作进行预测测试
        # 所以这里需要复制每一个动作的可能性
        # todo 但是对于连续值的动作或者动作空间很大的情况下，该如何处理？
        actions = np.tile(np.arange(0, self.n_actions, dtype=np.int64), batch_size)
        # todo 作用
        # step_Obs用来存储预测的每一步和上一步环境状态的差异
        step_obs, step_rewards = [], []

        # 开始预测未来rollout_steps步
        for step_idx in range(self.rollout_steps):
            # 创建动作tensor
            actions_t = torch.tensor(actions, dtype=torch.int64).to(batch.device)
            # 将所有能够执行的动作以及对应的当前环境的状态传入到环境模型中
            # 得到下一个状态的差异和回报奖励
            obs_next_v, reward_v = self.net_em(obs_batch_v, actions_t)
            # 将差异存放在step_obs
            # 回报奖励存放在step_rewards
            step_obs.append(obs_next_v.detach())
            step_rewards.append(reward_v.detach())
            # don't need actions for the last step
            # todo 为什么不需要执行最后一步的预测？
            if step_idx == self.rollout_steps-1:
                break
            # combine the delta from EM into new observation
            # 这里是将差异和当前的环境状态进行拼接，得到新的环境状态
            # todo 对于这种shape(batch_size, 2, 84, 84)以下这种获取方式拿到的是
            #  什么样的数据？哪帧的数据？
            cur_plane_v = obs_batch_v[:, 1:2]
            new_plane_v = cur_plane_v + obs_next_v
            # # 由于仅预测的是一帧的差异，所以仅取当前环境状态最新的一帧进行拼接得到
            # 新的环境状态
            obs_batch_v = torch.cat((cur_plane_v, new_plane_v), dim=1)
            # select actions
            # 根据当前的环境状态，预测动作
            logits_v, _ = self.net_policy(obs_batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            # 根据概率随机选择下一次需要执行的动作是什么？
            actions = self.action_selector(probs)
        # 将list按顺序拼接后转换为tensor，shape 是 （rollout_steps - 1， batch_size*n_actions， 1， 84， 84）
        step_obs_v = torch.stack(step_obs)
        # step_rewards 维度是（rollout_steps - 1， batch_size*n_actions， 1）
        step_rewards_v = torch.stack(step_rewards)
        # 将采集到的未来rollout_steps步的环境状态和回报奖励传入到编码器中
        # 拿到编码特征
        flat_enc_v = self.encoder(step_obs_v, step_rewards_v)
        # 将展平后的特征重新reshape为（batch_size, -1）
        # 这里的意思应该是将每一个环境的所有尝试的action路径的特征展平
        # 还原到每一个环境中
        #
        return flat_enc_v.view(batch_size, -1)
