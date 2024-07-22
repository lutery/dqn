#!/usr/bin/env python3
'''

'''
import gymnasium as gym
import argparse

from lib import model
from PIL import Image

import numpy as np
import torch
import torch.nn as nn


class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeObservation, self).__init__(env)

    def observation(self, observation):
        # 将观察从 (H, W, C) 转换为 (C, H, W)
        return observation.transpose(2, 0, 1)


ENV_ID = "CarRacing-v2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=r"M:\Projects\python\my_-nqd\learning\CarRacing\saves\ddpg-caracing\best_+756.370_105000.dat", help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("-s", "--save", type=int, help="If specified, save every N-th step as an image")
    # parser.add_argument("--acktr", default=False, action='store_true', help="Enable Acktr-specific tweaks")
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = TransposeObservation(gym.make(args.env, render_mode="human", continuous=True))
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = model.DDPGActor(env.observation_space.shape, env.action_space.shape[0]).to(device)
    # if args.acktr:
    #     opt = kfac.KFACOptimizer(net)
    net.load_state_dict(torch.load(args.model))

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs]).to("cuda")
        mu_v = net(obs_v).to("cpu")
        action = mu_v.squeeze(dim=0).data.numpy()
        obs, reward, done, trunc, _ = env.step(action)
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
