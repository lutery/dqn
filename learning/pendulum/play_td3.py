#!/usr/bin/env python3
import argparse
import gymnasium as gym

from lib import model

import numpy as np
import torch

ENV_ID = "Pendulum-v1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    env = gym.make(args.env, g=9.81, render_mode="human")
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = model.TD3Actor(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0]))
    net.load_state_dict(torch.load(args.model)["act_net"])

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        # action = np.clip(action, -1, 1)
        obs, reward, done, trunc, _ = env.step(action)
        done = done or trunc
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
