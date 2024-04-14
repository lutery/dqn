import gym
from gym import spaces, Wrapper
import numpy as np

# 创建游戏环境
env = gym.make("MountainCar-v0", render_mode="human")
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
# todo
ALPHA = 0.05
# todo
GAMMA = 0.98
# todo
NUM_DISCRETE_BINS = 30

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

class Q_Learner(object):
    def __init__(self, env):
        # 观察空间
        self.obs_shape = env.observation_space.shape
        # 观察的最大值（0.6、0.07）
        self.obs_high = env.observation_space.high
        # 观察的最小值
        self.obs_low = env.observation_space.low
        # 动作的离散化，也就是将原先的连续值转换为离散值
        self.obs_bins = NUM_DISCRETE_BINS
        # 环境观察是连续值，这里将其转换为离散值，转换为离散值是因为要根据离散值来从Q中获取Q值
        self.obs_width = (self.obs_high - self.obs_low) / self.obs_bins
        # 动作的个数
        self.action_shape = env.action_space.n
        # Q值的初始化，1、2维是离散的观察值，3维的索引是动作，3维存储的是Q值
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):
        '''
        将观察值从连续值转换为离散值
        :param obs:  观察值
        :return:
        '''

        return tuple(((obs - self.obs_low) / self.obs_width).astype(int))

    def get_action(self, obs):
        '''
        获取动作
        :param obs: 观察值
        :return:
        '''
        # 拿到离散的观察值
        discrete_obs = self.discretize(obs)
        # 更新epsilon值
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        # 选择是随机值还是学习到的动作
        if np.random.random() > self.epsilon:
            # 根据观察值从Q值中获取最大的Q值的动作
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        # 获取离散化的观察值
        discrete_obs = self.discretize(obs)
        # 获取离散化的下一个观察值
        discrete_next_obs = self.discretize(next_obs)
        # 计算TD目标Q值
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
        # 计算TD Q值的误差
        td_error = td_target - self.Q[discrete_obs][action]
        # 将Q值进行更新，利用alpha进行比例的更新，防止出现不稳定的情况
        self.Q[discrete_obs][action] += self.alpha * td_error

def train(agent, env):
    '''
    训练Q学习器
    :param agent: Q_Learner类实例
    :param env:  游戏环境
    :return:
    '''

    # 先设置一个目前能达到的最好奖励值
    best_reward = -float('inf')
    # 开始进行游戏训练
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs, _ = env.reset()
        total_reward = 0
        while not done:
            # 根据观察值获取动作
            action = agent.get_action(obs)
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode, total_reward, best_reward, agent.epsilon))
    return np.argmax(agent.Q, axis=2)

def test(agent, env, policy):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = policy[agent.discretize(obs)]
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    # 创建游戏环境
    env = gym.make("MountainCar-v0", render_mode="human")
    env = CustomEndConditionWrapper(env)
    # 创建Q学习器
    agent = Q_Learner(env)
    policy = train(agent, env)
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, policy)

# for episode in range(MAX_NUM_EPISODES):
#     done = False
#     obs = env.reset()
#     total_reward = 0
#     step = 0
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         next_state, reward, done, truncated, info = env.step(action)
#         total_reward += reward
#         obs = next_state
#         step += 1
#
#         print("Episode #{} ended in {} steps. total_reward={}".format(episode, step+1, total_reward))
#
# env.close()