'''
'''
import os
import gymnasium as gym
import torch
import torch.nn as nn
from lib import dqn_model


Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

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

def play_acrobot():
    # Set up the environment
    env = gym.make("LunarLander-v2", render_mode="human", continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the DQN model
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)

    # Load the trained model
    save_path = "saves"
    model_path = os.path.join(save_path, "noisy_net.pth")

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    else:
        print("No saved model found")
        return

    # Play the game
    state, _ = env.reset()
    total_reward = 0
    done = False
    total_steps = 0

    while not done:
        state_v = torch.tensor(state).float().unsqueeze(0).to(device)
        q_vals = net(state_v)
        _, act_v = torch.max(q_vals, dim=1)
        action = int(act_v.item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        state = next_state
        done = terminated or truncated

        env.render()

    print(f"Game finished with total reward: {total_reward} and total steps: {total_steps}")
    env.close()


if __name__ == "__main__":
    play_acrobot()