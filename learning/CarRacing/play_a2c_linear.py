#!/usr/bin/env python3
'''
实现基于trpo的训练方法
验证算法通过，算法有效，且比DDPG要更快达到最好的奖励
'''
import gymnasium as gym
import argparse

from lib import model
from PIL import Image

import numpy as np
import torch
import torch.nn as nn

class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        obs_action = (input_shape[2], input_shape[0], input_shape[1])

        self.conv = nn.Sequential(
            nn.Conv2d(obs_action[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_action)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeObservation, self).__init__(env)

    def observation(self, observation):
        # 将观察从 (H, W, C) 转换为 (C, H, W)
        return observation.transpose(2, 0, 1)


ENV_ID = "CarRacing-v2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=r"M:\Projects\python\my_-nqd\learning\CarRacing\saves\a2c-ppo_carracing_linear\a2c-best-935.19999999999.pth", help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("-s", "--save", type=int, help="If specified, save every N-th step as an image")
    # parser.add_argument("--acktr", default=False, action='store_true', help="Enable Acktr-specific tweaks")
    args = parser.parse_args()

    env = TransposeObservation(gym.make(args.env, render_mode="human", continuous=False))
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = AtariA2C(env.observation_space.shape, env.action_space.n)
    # if args.acktr:
    #     opt = kfac.KFACOptimizer(net)
    net.load_state_dict(torch.load(args.model))
    net.to(torch.device("cuda"))

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs]).to("cuda")
        mu_v = net(obs_v)[0].to("cpu")
        action = mu_v.squeeze(dim=0).data.numpy()
        obs, reward, done, trunc, _ = env.step(np.argmax(action))
        done = done or trunc
        total_reward += reward
        total_steps += 1
        if done:
            break
        if args.save is not None and total_steps % args.save == 0:
            o = env.render('rgb_array')
            img = Image.fromarray(o)
            img.save("img_%05d.png" % total_steps)
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
