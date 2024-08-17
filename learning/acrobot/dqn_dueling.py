#!/usr/bin/env python3
'''
未适配
'''
import gymnasium as gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import common


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 不同，有两个输出值
        conv_out_size = self._get_conv_out(input_shape)
        # 这个输出的值是bellman公式的选择下一个状态最大Q值的结果
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        # 这个输出的值是bellman公式的本次的激励
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        # 对应书中的138页的公式
        '''
        这里的模型是一个Dueling DQN。在这个网络架构中，原始的DQN被分解为两个独立的估计：一个值函数估计（value function estimator）和一个优势函数估计（advantage function estimator）。这两个部分分别是由fc_val和fc_adv表示的。

        - 值函数（V）预测的是在某个状态下能获得的预期奖励，即无论我们采取什么行动，我们预期会得到多少奖励。
        - 优势函数（A）预测的是采取每个可能的行动相对于平均行动的优势，即每种行动会带来多少额外的奖励。
        
        在Dueling DQN中，模型的输出是由值函数和优势函数共同决定的。最后的输出公式为：
        
        ```
        Q = V(s) + (A(s,a) - mean(A))
        ```
        
        也就是说，每个动作的Q值等于该状态的值函数（V）加上该动作的优势函数（A）减去所有可能动作的优势函数的平均值。这样做的主要目的是帮助网络更好地学习哪些状态是有价值的，而不仅仅是哪些动作是有价值的。
        
        对于 `adv - adv.mean(dim=1, keepdim=True)`，我们是在为每一个动作的优势值减去所有动作优势值的平均数。这个操作可以帮助网络识别出哪些动作对于当前状态的贡献超过了平均水平。简单来说，这是一种让模型更关注行动的相对优势而不是绝对优势的方法。
        
        注意，这并不会改变动作的相对优先级，因为所有的动作都减去了同一个数。但是，这会让优势函数更稳定，因为它会抵消掉任何可能对所有动作的优势值产生统一影响的因素。
        '''
        return val + (adv - adv.mean(dim=1, keepdim=True))


if __name__ == "__main__":
    params = {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-dueling")
    # 不同：采用的是竞争DQN网络
    net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    # 不同，没有eval_states，也就是没有验证步骤，不影响网络的拟合计算
    best_loss = 100

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            # 这里调用的损失计算和基准的DQN是一样
            # 因为双DQN网络实际上是改进的Q值计算，而不是改进计算过程
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
                best_loss = common.save_model("dueling_dqn", loss_v.item(), best_loss, net.state_dict())
                common.save_model("dueling_dqn_target", loss_v.item(), best_loss, tgt_net.target_model.state_dict())
