#!/usr/bin/env python3
'''
'''
import gymnasium as gym
import argparse

from lib import model
from PIL import Image

import numpy as np
import torch


ENV_ID = "Ant-v4"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(ENV_ID)

    net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net.load_state_dict(torch.load(args.model))

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor(obs).to(device=device)
        mu_v = net(obs_v)
        action = mu_v.cpu().squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, trunc, _ = env.step(action)
        done = done or trunc
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
