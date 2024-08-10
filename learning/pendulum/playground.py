#!/usr/bin/env python3
import argparse
import gymnasium as gym
import os
from PIL import Image
ENV_ID = "Pendulum-v1"

if __name__ == "__main__":

    env = gym.make(ENV_ID, g=9.81, render_mode="rgb_array")

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        action = env.action_space.sample()  # Sample random action
        obs, reward, done, trunc, info = env.step(action)
        done = done or trunc
        total_reward += reward
        total_steps += 1
        o = env.render()
        img = Image.fromarray(o)
        img.save("img_%05d.png" % total_steps)
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
