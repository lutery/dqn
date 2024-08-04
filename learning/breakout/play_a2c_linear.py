#!/usr/bin/env python3
'''
实现基于trpo的训练方法
验证算法通过，算法有效，且比DDPG要更快达到最好的奖励
'''
import gymnasium as gym
import argparse

from PIL import Image

import numpy as np
import torch
import torch.nn as nn

from typing import Any

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

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        # 以下可知，一些游戏存在FIRE的动作，并且存在FIRE动作的游戏其游戏动作执行有三个以上
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # 这里之所以尝试重置后尝试各种动作，是因为不知道哪个是FIRE，继续游戏，所以一个一个尝试
        # 如果不小心游戏结束了，则继续重置
        # 假设游戏继续游戏的按钮在前3
        self.env.reset(seed=seed, options=options)
        obs, _, done, _, info = self.env.step(1)
        if done:
            self.env.reset(seed=seed, options=options)
        obs, _, done, _, info = self.env.step(2)
        if done:
            self.env.reset(seed=seed, options=options)
        return obs, info

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

    env = FireResetEnv(TransposeObservation(gym.make("ALE/Breakout-v5", obs_type='rgb', frameskip=4, repeat_action_probability=0.0)))
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = AtariA2C(env.observation_space.shape, env.action_space.n)
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
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
