#!/usr/bin/env python3
'''
未适配
'''
import gymnasium as gym
import ptan
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01  # 熵奖励的规模，todo 用于控制惩罚的力度
BATCH_SIZE = 8

REWARD_STEPS = 10


class PGN(nn.Module):
    '''
    策略梯度，这里直接输出执行的每个动作的可能性
    '''
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 增加了控制是否开启奖励基线计算
    parser.add_argument("--baseline", default=False, action='store_true', help="Enable mean baseline")
    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg" + "-baseline=%s" % args.baseline)

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    # 构建策略梯度网络代理，这里的apply_softmax=True，表示输出的是softmax之后的概率分布
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    # 创建经验池，此经验池只返回第一个和最后一个经验的reward
    # GAMMA用来计算基于当前步数计算的未来REWARD_STEPS的奖励衰减系数
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(exp_source):
        # 这里的reward是REWARD_STEPS展开的奖励和
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        # 不同，增加了是否加入奖励基线的计算
        if args.baseline:
            batch_scales.append(exp.reward - baseline)
        else:
            batch_scales.append(exp.reward)

        # handle new rewards
        # 获取最近一次游戏运行的到结束时的激励状态
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            # 获取最近一次游戏结束时的奖励
            reward = new_rewards[0]
            total_rewards.append(reward)
            # 计算近100轮游戏的平均奖励
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            # 记录到tensorboard
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            # 如果奖励达到目标值，则退出训练
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        # 判断是否收集到指定的训练数据
        if len(batch_states) < BATCH_SIZE:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()
        logits_v = net(states_v)
        # 计算动作概率
        log_prob_v = F.log_softmax(logits_v, dim=1)
        # 获取奖励*动作的加权和
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        # 计算加权和的平均值
        loss_policy_v = -log_prob_actions_v.mean()

        # 不同，这里是设置梯度不自动清除
        # 本质上和之前的计算一致，这里是因为要在tenboard中记录一次传播后
        # 损失地图的大小而改成分开计算
        loss_policy_v.backward(retain_graph=True)
        # 提取更新梯度
        grads = np.concatenate([p.grad.data.numpy().flatten()
                                for p in net.parameters()
                                if p.grad is not None])

        # 计算策略梯度预测的熵
        prob_v = F.softmax(logits_v, dim=1)
        # H(p) = - Σ p(x) log p(x) 信息熵的计算公式，这里就是计算信息熵
        # 这里对应了书中公式
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        # 这里添加负号，是为了在最小化损失函数时，使得信息熵越大越好
        # 因为为了使损失最小，那么entropy_v就要越大，所以添加负号，鼓励智能体进行探索
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        # 不同，这里计算梯度时，是分别计算了策略损失和熵损失的梯度后，然后迭代更新
        # 而之前是计算两个损失的总和后，利用总和计算梯度
        # 这里的分开计算和之前代码的合并计算损失后迭代梯度的效果是等价的
        # 也就是说loss = loss_1 + loss_2 +... + loss_n是可以拆分为
        # 各个loss的，再backward()，唯一要注意的是，如果loss的来源于
        # 相同的net，那么在调用backward()时需要保存计算图，即retain_graph=True
        entropy_loss_v.backward()
        optimizer.step()

        loss_v = loss_policy_v + entropy_loss_v

        # calc KL-div
        # 和之前一样，统计一些训练过程中的信息
        # 比如更新前后的动作分布，基线改变，损失，梯度的方差、均值等信息
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy_v.item(), step_idx)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)

        writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), step_idx)
        writer.add_scalar("grad_var", np.var(grads), step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
