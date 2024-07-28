#!/usr/bin/env python3
"""
未验证

优先级dqn是利用优先级，保证已经完成拟合的数据少参与训练，没有被拟合的数据多参与训练以达到
加快训练的目的

更新样本优先级的依据是损失的大小
"""
import gymnasium as gym
import ptan
import numpy as np
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000


class PrioReplayBuffer:
    """
    prob_alpha: 样本优先级重视程度参数，如果是0则不重视优先级
    priorities: 样本缓冲区，每一个样本的重置程度队列 初始值为0
    """
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        # 获取最大的优先级，如果buffer存在样本，则从样本优先级缓冲区获取，否则默认为1.0
        # 意味每一个样本同样重视程度
        # 而这个max_prio最大优先级的重视程度是用来给最新的样本
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        采样

        beta参数的作用：
        beta参数在sample方法中的作用是调整重要性采样权重（importance sampling weights）的计算。重要性采样权重用于在训练过程中平衡不同样本对网络更新的贡献。

    在sample方法中，首先根据优先级概率抽取一批样本。由于这种抽样方式可能导致某些样本被过度抽样，因此需要使用重要性采样权重来修正网络更新。beta参数决定了这种修正的强度。

    具体来说，当beta值较小时，修正的强度较弱，导致网络更新主要受到高优先级样本的影响。当beta值较大时，修正的强度较强，使得网络更新更均衡地考虑不同优先级的样本。

    通常，可以在训练开始时设置较小的beta值，并在训练过程中逐渐增大，以实现动态调整。这种策略可以使网络在训练初期更快地收敛到较好的策略，同时在后期避免过度关注某些样本。

    在sample函数中，beta参数会影响所有抽取到的样本。通过计算每个样本的重要性采样权重，beta值会调整这些权重的大小，从而影响网络更新过程中每个样本的贡献。

在计算重要性采样权重时，权重与优先级概率成反比。较大的beta值会使权重相对均衡，降低高优先级样本的权重并提高低优先级样本的权重。这样可以使网络在训练过程中更公平地考虑不同优先级的样本。
        """
        # 根据采样的样本缓冲区长度，获取对应的样本优先级重视程度
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        # 每个样本的优先级首先进行alpha次方计算
        probs = prios ** self.prob_alpha

        # 再除以综合计算平均值得到每个样本的优先级
        # 这一段的计算公示在书p132页
        # 将优先级转换为概率
        probs /= probs.sum()
        # 根据样本的重视程度进行优先级采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        # 计算每个样本的权重 计算公式在书P133
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        """
        更新每个样本的优先级重视度
        样本的权重值来源于在计算误差时，每一个样本的误差损失之，误差损失之越大，那么
        表示这个样本需要重视，重新参与计算的可能性需要越大
        batch_indices: 本次训练采样的样本索引
        batch_priorities:
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    """
    batch: 本次采样的样本
    batch_weights: 每个样本的权重大小

    return: 返回的损失与之前的不一致，返回的损失中没有直接返回MSELoss，经过样本权重修改后的MSELoss值 以及 losses_v + 1e-5的值买这个值的作用就是更新样本缓冲区的优先级值
    """
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    # 权重值大的，这里计算后会放大误差，权重值小的则缩小误差，使最新的样本（误差越大的样本）能够快速参与改变网路而不受就样本的干扰，提高拟合 的速度
    # 而且如果期待值误差较大就会提高优先级，让其参与下一次的训练，多次训练以提高网络的预测参数，避免那些已经拟合的参数再次参与训练
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    # 1e-5的含义书P135
    # 在这段代码中，1e-5是一个较小的常数，它被添加到losses_v中以避免优先级值为零的情况。losses_v表示每个样本的TD误差的平方，而sample_prios_v表示每个样本的优先级值。在更新优先级时，需要使用非零值以确保所有样本都有一定的抽样概率。1e-5的作用是确保即使TD误差非常接近于零，每个样本仍然具有一定的优先级。
    #
    # 通过添加这个较小的常数，可以防止样本由于优先级过低而被完全忽略。这样可以确保所有样本都能在训练过程中得到一定程度的关注。
    return losses_v.mean(), losses_v + 1e-5


if __name__ == "__main__":
    params = {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-prio-replay")
    net = dqn_model.DQNBreakOut(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    # 这里不同，采用的是具备优先级的经验重放缓冲区
    buffer = PrioReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    # 这里不同，没有eval_states， 而是BETA，样本权重参数
    beta = BETA_START
    best_loss = 100

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            # 根据书中描述，BETA的值需要慢慢随着训练增加到1比较有利于收敛
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            # 损失值的计算不一样，使用的是具备优先级样本的计算方式
            loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                               params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()
            # 更新样本的优先级
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
                best_loss = common.save_model("prio_replay_dqn", loss_v.item(), best_loss, net.state_dict())
                common.save_model("prio_replay_dqn_tgt", loss_v.item(), best_loss, tgt_net.target_model.state_dict())
