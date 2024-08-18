#!/usr/bin/env python3
'''
未适配
'''
import gymnasium as gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4 #定义用于训练的完整游戏探索需要几个总流程


class PGN(nn.Module):
    '''
        基于策略梯度的DQN
        '''
    def __init__(self, input_size, n_actions):
        # input_size：依旧是输入的状态的维度
        # n_actions: 依旧还是输出的动作的维度，即执行动作的个数
        super(PGN, self).__init__()

        # 网络输出的部署概率，而是原始动作分布
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        # 这里并未使用softmax，因为后续在计算损失的时候，会使用pytorch的交叉熵函数，该函数会自动进行softmax
        # 这样做的好处就是，在推理阶段可以少计算一步
        # 这里输出的是原始分布
        return self.net(x)


def calc_qvals(rewards):
    '''
    rewards: 一整个游戏从开始到结束时的rewards list
    '''
    res = []
    sum_r = 0.0
    # 计算一个游戏序列中的总奖励
    # 计算方式从最后一步往前开始加权累加， rn-2 + gamma * (rn-1 + gamma * rn) ...
    # 这样做的好处可以控制网络选择的动作可以尽快的获得奖励，完成有游戏
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    # 基于策略的动作选择器，与q值最大的不同之处在于，其选择下一个动作是基于概率的，不是一定
    # 选择概率最大的动作
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # 第一组：主要用于报告，包含episode的总奖励，以及本轮游戏的步数
    total_rewards = []
    step_idx = 0
    # 游戏进行的轮次
    done_episodes = 0

    # 第二组：收集详细的游戏数据，在以下应该是指当前继续游戏一轮的状态数据
    batch_episodes = 0
    # batch_qvals: 通过calc_qvals计算得到的本地奖励的总奖励
    # batch_states和batch_actions 包含训练中看到的状态和指定的动作
    batch_states, batch_actions, batch_qvals = [], [], []
    # 表示当前执行episode的本地奖励
    # 在游戏结束后，这个会清空
    cur_rewards = []

    # 开始训练
    # 从经验源获取当前的步数，（状态，动作，奖励，下一个状态）
    # 如果游戏结束，则下一个状态是None
    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        # 游戏结束，但是这里并没有清楚的将每轮游戏分割开，而是将所有的游戏数据都放在一起
        if exp.last_state is None:
            # 计算每一轮游戏到结束时，每个动作所执行的动作对应的回报奖励
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        # handle new rewards
        # 书上说这部分在episode结束时执行
        # ExperienceSourceFirstLast 只有在一轮游戏结束后，pop_total_rewards才会返回值
        # 这里返回的就是一轮游戏的总奖励，然后将其记录到tensorboard中
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            # 以下主要是计算轮次，总奖励，当前执行所有的轮次的平均值信息记录在tensoerboard中
            # 并判断总奖励是否满足要求，达到要求后推出训练
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        # 需要保证当前收集游戏进行的轮次和配置一样
        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = net(states_v)
        # 这边计算损失的公式就是P176也缩写的公式，但是具体怎么对应还需要看 todo
        # log_softmax公式本质上还是softmax，用来求解每个状态下执行动作的概率分布
        # 加上log可以避免sofmax上溢或者下溢的问题
        log_prob_v = F.log_softmax(logits_v, dim=1)
        # batch_qvals_v是每个状态下执行动作获取的q值奖励，这个奖励和最终的状态有关系
        # 如果奖励是负数，则说明选择执行的动作是错误的，需要惩罚
        # 如果奖励是正数，则说明选择执行的动作是正确的，不需要惩罚甚至鼓励
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        # log_prob_actions_v是每个动作概率和其奖励的乘机，如果其值越大，说明正向的奖励以及对应的
        # 动作概率越高，而负向奖励及其对应的动作概率越低，而这正好可以用来作为损失函数进行训练
        # 所以在这里log_prob_actions_v越大越好，但是我们求解的是损失，损失越小越好，所以需要加上负号，而网络为了使其最小化，梯度必定是将负奖励的概率降低，正奖励的概率提高，这样就可以达到训练的目的
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        # 完成训练，清空历史训练数据，准备下一轮训练
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()
