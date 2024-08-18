import time
import numpy as np
import sys
import torch
import torch.nn as nn
import os
import ptan


def save_model(model_name, loss, best_loss, model):
    if not os.path.exists("saves"):
        os.makedirs("saves")

    if loss < best_loss:
        torch.save(model, f'saves/best_model_{model_name}.dat')
        best_loss = loss

    torch.save(model, f'saves/model_{model_name}.dat')

    return best_loss

def unpack_batch(batch):
    '''
    将经验重放缓冲区的经验进行解包操作，从经验中获取各个经验类别
    '''

    # states 每次采集样本时的环境状态
    # actions 每次状态下执行的动作值
    # rewards 每次执行完动作后获取的激励值
    # dones 执行动作后游戏是否结束
    # last_states 执行动作之后的下一个状态
    #
    # return 将states, actions, rewards, dones, last_states各个list转换为numpy
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)



def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    '''
    计算dqn的损失值
    net: 负责计算当前状态下的动作Q值
    tag_net: 负责计算下一个状态下的动作Q值
    '''
    # 提取样本集中游戏的各个状态和动作
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    # 将数据传输到指定的设备中
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # 将状态传输到神经网路中，获取到神经网路推理出来的执行的动作
    # 网络细节查看第06章 02_dqn_pong.py
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # tag_net负责计算下一个状态的Q值
    # 并将其中会导致游戏结束的动作的Q值设置为0，这样可以将不好的q值降低
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask.bool()] = 0.0

    # 损失值计算：下一个状态的最大Q值+实际获取到的激励值 == 当前net计算出来的Q值
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, epsilon_start, epsilon_final, epsilon_frames):
        '''
        这个跟踪器是根据当前执行的帧数大小，以及epslison的参数来计算动作选择其中
        选择动作的概率

        param epsilon_greedy_selector: 动作选择器
        param params: map类型的参数，也就是本文中的HYPERPARAMS参数
        '''

        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = epsilon_frames
        self.frame(0)

    def frame(self, frame):
        '''
        根据当前的帧数，更新epsilon的值，eplison是用来决定每次执行的策略是
        随机还是神经网络推理获取的
        '''
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
        

"""
该类就是用来跟踪、记录、判断激励的追踪类
"""
class RewardTracker:
    def __init__(self, writer, stop_reward):
        '''
        param writer: tensorboard writer保存
        param stop_reward: 停止训练的激励值\目标值
        '''

        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        # total_rewards 训练期间的每一步的激励值，用来记录
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        '''
        param reward: 样本
        param fream: 当前进行了第frame次的训练
        param epsilon：当前的epsilon值

        return True: 表示已经达到了目标激励值 False： 表示还没有达到目标的激励值
        '''
        # 激励经验存储在总缓存区
        self.total_rewards.append(reward)
        # 计算当前的平均帧率
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        # 将当前帧总数和所花费的时间存储在缓存中
        self.ts_frame = frame
        self.ts = time.time()
        # 计算平均激励值
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
        # 如果当前获取的激励已经达到了目标的激励大小，则返回true
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


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