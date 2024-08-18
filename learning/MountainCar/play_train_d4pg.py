import os
import gymnasium as gym
import torch
import numpy as np
from lib import model


def play_mountaincar():
    ENV_ID = "MountainCarContinuous-v0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(ENV_ID, render_mode="human")

    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    save_path = os.path.join("saves", "d4pg-mountaincar")
    model_path = os.path.join(save_path, "act-0.pth")

    if os.path.exists(model_path):
        act_net.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    else:
        print("No saved model found")
        return

    total_reward = 0
    total_steps = 0
    obs, _ = env.reset()

    while True:
        obs_v = torch.FloatTensor([obs]).to(device)
        action_v = act_net(obs_v)
        action = action_v.squeeze(dim=0).data.cpu().numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done or trunc:
            break

    print(f"Total reward: {total_reward} in {total_steps} steps")
    env.close()


if __name__ == "__main__":
    play_mountaincar()