import sys
import time
import numpy as np

import torch
import torch.nn as nn
import ptan

HID_SIZE = 128

class BipedalWalkerModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        '''
        obs_size: 观测的数据维度，不是图像数据，所以后续用的是全连接层
        act_size: 动作空间的维度，在这个游戏里面，指的是同时执行动作的数量
        '''

        super(BipedalWalkerModelA2C, self).__init__()

        # 只有这个是提取特征，其余的都是输出结果
        self.base = nn.Sequential(
            nn.Linear(obs_size[0], HID_SIZE),
            nn.ReLU(),
        )
        # 输出均值，表示最终要执行的动作内容
        # 这里由于预测的输出动作包含负值，所以采用tanh函数，将输出值限制在-1到1之间
        # 而不是用sigmoid
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size[0]),
            nn.Tanh(),
        )
        # https: // zhuanlan.zhihu.com / p / 461707201
        # var作用 方差平方，怀疑是用来稳定输出的概率范围大小
        # 用来促进网络进行探索以及指导网路朝哪个方向进行训练，使得整体趋近中值，但是在这里并没有直接使用方差，而是使用了信息熵的方式
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size[0]),
            nn.Softplus(), # Relu的替代函数，用于解决梯度消失问题 具体使用场景查看笔记内容
        )
        # 状态值（Q值），用来评价当前Q值，来评估当前执行的动作是否有优势
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        '''
        return 均值，方差平方，Q值
        '''
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch: 收集的游戏数据
    :param net:
    :return: states variable, actions tensor, reference values variable（游戏环境状态、执行的动作、评价的Q值）
    """
    states = [] # 每一步的游戏状态
    actions = [] # 每一步执行的动作
    rewards = [] # 每一步执行动作后获取的奖励
    not_done_idx = [] # 执行动作后没有结束游戏的索引
    last_states = [] # 执行动作后的下一个状态
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(np.array(actions)).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        # 如果存在下一个状态的游戏数据，那么计算Q值
        # 对于已经结束的游戏动作，他们的Q值
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)[2]
        # todo 为什么这里计算的Q值没有选择最大值
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v

def unpack_batch_ddqn(batch, device="cpu"):
    '''
    解压深度确定性策略梯度网络的数据
    '''
    # states: 环境状态
    # actions: 执行的动作
    # rewards： 执行动作后获取的奖励
    # dones: 执行动作后游戏是否结束
    # last_states: 未结束的游戏，执行动作后的达到的状态（针对多步展开，则是展开的最后一个动作）；如果是游戏已经结束的状态，则保存的还是和states中一样的状态，如果不是游戏结束的状态，则保存执行动作后的下一个状态
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.ByteTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v

def unpack_batch_acktr(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch: 收集的游戏数据
    :param net:
    :return: states variable, actions tensor, reference values variable（游戏环境状态、执行的动作、评价的Q值）
    """
    states = [] # 每一步的游戏状态
    actions = [] # 每一步执行的动作
    rewards = [] # 每一步执行动作后获取的奖励
    not_done_idx = [] # 执行动作后没有结束游戏的索引
    last_states = [] # 执行动作后的下一个状态
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(np.array(actions)).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        # 如果存在下一个状态的游戏数据，那么计算Q值
        # 对于已经结束的游戏动作，他们的Q值
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)
        # 为什么这里计算的Q值没有选择最大值
        # 因为评级网络是直接输出最大Q值，所以这里直接取值就可以了
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        # 这里依旧是计算Q值
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
