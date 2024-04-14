import gym
from gym import spaces, Wrapper
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 测试多少次游戏轮数
MAX_NUM_EPISODES = 50000
# 每轮游戏的最大步数
STEPS_PER_EPISODE = 200
# 游戏动作选择器（即采用随机值还是学习到的动作进行动作执行）
EPSILON_MIN = 0.005
# 总共的游戏步数
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
# epsilon衰减值，这个值根据最大的游戏步数来计算，表示每个步数epsilon的衰减值
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
# 更新Q值的比例
ALPHA = 0.05
# todo
GAMMA = 0.98
# todo
NUM_DISCRETE_BINS = 30

class LinearDecaySchedule:
    def __init__(self, initial_value, final_value, max_steps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.max_steps = max_steps

    def __call__(self, step_num):
        """Calculate the current value of epsilon based on the step number."""
        # Linear decay formula
        current_value = self.initial_value - (self.initial_value - self.final_value) * (step_num / self.max_steps)
        # Clamp the value to ensure it's not less than final_value
        if current_value < self.final_value:
            current_value = self.final_value
        return current_value

class CustomEndConditionWrapper(Wrapper):
    def __init__(self, env):
        super(CustomEndConditionWrapper, self).__init__(env)
        self.step_count = 0

    def step(self, action):
        self.step_count += 1
        observation, reward, done, trun, info = self.env.step(action)
        # 自定义结束条件：例如位置大于0.45或小于-0.45
        if observation[0] >= 0.5 or self.step_count >= STEPS_PER_EPISODE:
            self.step_count = 0
            done = True
        return observation, reward, done, trun, info


class SLP(torch.nn.Module):

    def __init__(self, input_shape, output_shape, device="cpu"):
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = nn.Linear(self.input_shape, self.hidden_shape)
        self.out = nn.Linear(self.hidden_shape, output_shape)


    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = F.relu(self.linear1(x))
        x = self.out(x)
        return x


class Shallow_Q_Learner(object):
    def __init__(self, state_shape, action_shape, learnning_rate=0.005, gamma=0.98):
        # 观察空间
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.learning_rate = learnning_rate
        # Q值的初始化，1、2维是离散的观察值，3维的索引是动作，3维存储的是Q值
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = SLP(self.state_shape, self.action_shape, device=dev).to(dev)
        self.Q_optim = optim.Adam(self.Q.parameters(), lr=learnning_rate)
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max,
                                                final_value=self.epsilon_min,
                                                max_steps=0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        self.step_num = 0


    def get_action(self, obs):
        self.step_num += 1
        return self.policy(obs)


    def epsilon_greedy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device("cpu")).numpy())

        return action


    def learn(self, obs, action, reward, next_obs):
        # 计算TD目标Q值
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        # 计算TD Q值的误差
        td_error = F.mse_loss(self.Q(obs)[action], td_target)
        # 将Q值进行更新，利用alpha进行比例的更新，防止出现不稳定的情况
        self.Q_optim.zero_grad()
        td_error.backward()
        self.Q_optim.step()


if __name__ == "__main__":
    # 创建游戏环境
    env = gym.make("CartPole-v1", render_mode="human")
    env = CustomEndConditionWrapper(env)
    # 创建Q学习器
    agent = Shallow_Q_Learner(env.observation_space.shape, env.action_space.n)
    first_episode = True
    episode_reward = list()
    for episode in range(MAX_NUM_EPISODES):
        obs, _ = env.reset()
        cum_reward = 0
        for step in range(STEPS_PER_EPISODE):
            action = agent.get_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            cum_reward += reward
            if done:
                if first_episode:
                    max_reward = cum_reward
                    first_episode = False
                episode_reward.append(cum_reward)
                if cum_reward > max_reward:
                    max_reward = cum_reward
                print("Episode#:{} ended in {} steps. reward:{} max_reward:{} mean_reward:{}".format(episode, step+1, cum_reward, max_reward, np.mean(episode_reward)))
                break

    env.close()