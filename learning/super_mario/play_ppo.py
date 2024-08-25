#!/usr/bin/env python3
'''
'''
import gymnasium as gym
import argparse

from lib import model
from PIL import Image

import numpy as np
import torch
import os
import ptan
# super mario
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import StepAPICompatibility, TimeLimit
#

class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeObservation, self).__init__(env)

    def observation(self, observation):
        # 将观察从 (H, W, C) 转换为 (C, H, W)
        return observation.transpose(2, 0, 1)
    
def gymnasium_reset(self, **kwargs):
    return self.env.reset(), {}

def make_super_mario_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    steps = env._max_episode_steps  # get the original max_episode_steps count
    env = JoypadSpace(env.env, SIMPLE_MOVEMENT)  # set the joypad wrapper
    # overwrite the old reset to accept `seeds` and `options` args
    env.reset = gymnasium_reset.__get__(env, JoypadSpace)

    # set TimeLimit back
    env = TimeLimit(StepAPICompatibility(env, output_truncation_bool=True), max_episode_steps=steps)

    return TransposeObservation(env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=r"M:\Projects\python\my_-nqd\learning\super_mario\saves\ppo-mario\best_+741.000_200000.dat", help="Model file to load")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("-s", "--save", type=int, help="If specified, save every N-th step as an image")
    # parser.add_argument("--acktr", default=False, action='store_true', help="Enable Acktr-specific tweaks")
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = make_super_mario_env()

    # 创建动作预测网络
    net_act = model.ModelActor(env.observation_space.shape, env.action_space.n).to(device)
    # 创建状态、动作评价网络
    print(net_act)
    if (os.path.exists(args.model)):
        net_act.load_state_dict(torch.load(args.model))
        print("加载act模型成功")
    else:
        print("模型不存在")
        exit(1)

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
        mu_v = net_act(obs_v)
        action = mu_v.squeeze(dim=0).data.cpu().argmax().item()
        obs, reward, done, trunc, _ = env.step(action)
        env.render()
        done = done or trunc
        if done:
            print(f"reward is {reward}")
        else:
            print(f"reward is {reward}")
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
