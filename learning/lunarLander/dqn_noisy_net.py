#!/usr/bin/env python3
"""
未验证
"""
import gymnasium as gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import dqn_model, common


class NoisyDQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(NoisyDQN, self).__init__()

        self.feature_linears = nn.Sequential(
            nn.Linear(obs_size[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        # 不同之处，创建了噪音网络
        self.noisy_layers = [
            dqn_model.NoisyLinear(512, 512),
            dqn_model.NoisyLinear(512, n_actions)
        ]
        # 利用噪音全连接层替换原来的全连接层
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def forward(self, x):
        fx = x.float()
        features = self.feature_linears(fx)
        return self.fc(features)

    def noisy_layers_sigma_snr(self):
        """
        计算有噪层的信噪比来获得内部的噪音水平
        """
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]

import os
save_path = 'saves'

if __name__ == "__main__":
    params = common.HYPERPARAMS['LunarLander']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    env =  gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-noisy-net")
    # 与之前不同 这里创建NoisyDQN网络层
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    # 与之前不同，采用的是ArgmaxActionSelector动作选择的器.选择Q值最大的动作
    agent = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            # 去除了epsilon_tracker.frame(frame_idx)，原因？
            # 因为这里使用了噪音去达到epsilon相类似的随机效果

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                # 虽然没有使用epsilon，是因为这里使用的是ArgmaxActionSelector
                # 不需要记录epsilon
                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            # 使用了还是i之前的计算损失的方式
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            if frame_idx % 500 == 0:
                for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                    writer.add_scalar("sigma_snr_layer_%d" % (layer_idx+1),
                                      sigma_l2, frame_idx)

                torch.save(net.state_dict(), os.path.join(save_path, 'noisy_net.pth'))
                torch.save(tgt_net.target_model.state_dict(), os.path.join(save_path, 'noisy_tgt_net.pth'))
