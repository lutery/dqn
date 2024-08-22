import os
import gymnasium as gym
import torch
from lib import dqn_model


def play_acrobot():
    # Set up the environment
    env = gym.make("Acrobot-v1", render_mode="human")

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the DQN model
    net = dqn_model.DQNAcrobot(env.observation_space.shape[0], env.action_space.n).to(device)

    # Load the trained model
    save_path = "saves/dqn_basic"
    model_path = os.path.join(save_path, "net.pth")

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