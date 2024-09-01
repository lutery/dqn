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


ENV_ID = "LunarLander-v2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=r"M:\Projects\python\my_-nqd\learning\CarRacing\saves\a2c-ppo_carracing_linear\a2c-best-935.19999999999.pth", help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("-s", "--save", type=int, help="If specified, save every N-th step as an image")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(
        "LunarLander-v2",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5)

    net = model.ModelActorDis(env.observation_space.shape[0], env.action_space.n).to(device)
    net.load_state_dict(torch.load(args.model))
    net.to(device)

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs]).to(device)
        mu_v = net(obs_v)[0].to("cpu")
        action = mu_v.squeeze(dim=0).data.numpy()
        obs, reward, done, trunc, _ = env.step(np.argmax(action))
        done = done or trunc
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
