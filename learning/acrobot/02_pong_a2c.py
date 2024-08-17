#!/usr/bin/env python3
'''
未适配
'''
import gymnasium as gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
# todo 游戏环境的数量
NUM_ENVS = 50

# 向前展开的步数，todo 这里为什么从10变成4
REWARD_STEPS = 4
# 增加了梯度裁剪值，防止梯度变得太大
CLIP_GRAD = 0.1


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        # A2C 游戏观察网络，获取游戏特征
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # 根据游戏特征，返回预测动作的概率
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # 根据游戏状态特征，预测得到当前的状态优势
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # 归一化
        # 返回值1：策略
        # 返回值2：值 todo 作用
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = [] # 非结束的游戏数据索引，该索引记录对应batch，states，actions，rewards
    last_states = [] # 记录采样中的执行动作后的状态，仅记录游戏非结束状态下的索引
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            # 处理游戏没有结束的采样数据，或者是一个完整游戏采样的数据
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        # 如果采集的游戏数据有存在非结束的采样
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        # 根据网络传入执行动作后的游戏状态，获取评价分数
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        # 根据197公式计算Q值
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    # 状态batch，动作batch，使用p197公式计算的Q值Batch
    # Q值batch主要用于两个地方
    # 1. 计算均方误差MSE损失提高近似值
    # 2. 计算动作的游戏来评价动作
    return states_v, actions_t, ref_vals_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # 创建游戏环境，并根据训练方式，使用合适的包装器包装游戏
    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    # 创建多个环境
    envs = [make_env() for _ in range(NUM_ENVS)]
    # 创建记录器
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    # 创建网络
    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    # 创建环境代理
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    # 创建环境数据源，能够按照指定的步数进行多步展开
    # 类似ExperienceSourceFirstLast每次遍历仅返回一步的数据
    # 所以这里要使用batch缓冲区存储数据
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    # 创建优化器
    # 通常来说，eps参数的作用是防止发生除0的事件，一般使用的1e-8或更小的值
    # 这里使用的是1e-3，是因为1e-8太小了，导致训练无法收敛，所以这里采用1e-3
    # 书中无法给出实际的证明，猜测可能是1e-8太小，放在分母计算导致梯度太大，而使得
    # 训练不稳定
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    #todo 完善RewardTracker注释 作用统计近100次游戏的奖励，如果奖励达标则退出游戏
    with common.RewardTracker(writer, stop_reward=18) as tracker:
        # todo 完善TBMeanTracker注释，作用将最近的10步游戏记录到tensorboard中
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                # 执行一次
                # 将执行结果存储
                batch.append(exp)

                # handle new rewards
                # # 获取最近一次游戏运行的到结束时的激励状态
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    # 将最后一次游戏完整运行的情况打印出来，如果激励平均值达到了指定目标，
                    # 则判定为训练成功，退出循环
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                # 如果收集的游戏数据不足BATCH_SIZE，则跳过
                if len(batch) < BATCH_SIZE:
                    continue


                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                # 这里有点像bellman展开式的作用
                # 根据书中的197页Q值公式
                # 这里是为了能够训练网络能够根据当前的环境状态计算得到状态Q值，利用状态Q值来
                # 得到动作优势值
                # todo 为什么可以这样计算，因为vals_ref_v是包含状态优势和动作优势
                #  而网络仅包含状态优势，从下面可以看到
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                # 接下来就是策略梯度的损失计算了
                log_prob_v = F.log_softmax(logits_v, dim=1)
                # 根据状态计算的来的奖励基线（动作优势评价基线）
                # 这里是获取动作的优势值 todo
                # todo 为什么可以分为Q = A+V 书P200
                # 之所以可以这么分，是因为Q值是状态Q值和动作Q值

                # 问：为什么这里计算了 loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)后，又  adv_v = vals_ref_v - value_v.squeeze(-1).detach()，这样不会导致adv_v接近0吗？
                # 答：首先，loss_value_v 被计算为估计的状态价值 value_v 和目标状态价值 vals_ref_v 之间的均方误差 (MSE)。目标是减小此损失，从而使网络对状态的价值估计更为准确。
                #
                # 紧接着，代码计算了adv_v，它代表了 Advantage（优势），即真实的价值（vals_ref_v）与估计的价值（value_v）之间的差异。这个 Advantage 用于指导策略的优化。在强化学习中，我们希望采取的行动能够获得超出当前价值估计的回报。
                #
                # 你提到的疑问是：既然我们试图通过 MSE 损失最小化 vals_ref_v 和 value_v 之间的差异，那么adv_v岂不是会接近0吗？
                #
                # 答案是：在训练初期，adv_v可能不会接近0，因为模型尚未充分优化。但随着训练的进行，的确，我们期望adv_v会逐渐减小。但这并不意味着它总是接近于0或没有用。为什么？
                #
                # 实时更新：在强化学习中，模型是在与环境互动的过程中实时更新的。因此，即使在某一时刻，我们的价值估计与目标非常接近，环境的动态性和模型的不断更新意味着adv_v仍然有可能在后续时刻变得相对较大。
                #
                # 策略更新的方向：即使adv_v变小，它仍然为我们提供了一个关于如何更新策略的方向。如果 Advantage 是正的，这意味着我们的行动比预期的要好；如果是负的，则表示行动比预期的要差。这种信息对于调整策略是有用的。
                #
                # 探索与利用：在策略梯度方法中，通常会加入熵正则项（如entropy_loss_v）来鼓励策略探索。这有助于确保模型不会过早地收敛到局部最优，并使adv_v在整个训练过程中保持活跃。
                #
                # 总的来说，尽管在训练的某些时刻，你可能会观察到adv_v接近于0，但它在策略更新中仍然是有用的，尤其是在训练的早期阶段。
                # adv_v主要是用来衡量，我当前指定的动作，是否符合预期，如果adv_v>0，则表示符合甚至超出预期，如果adv_V，则表示不符合预期，执行的动作有问题
                # 实际结果也表明，adv_V也是在0附近徘徊
                adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                # 这里计算熵之所以没有负号，是因为和熵的计算公式中的符号直接对消了
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                # 这里又将损失的梯度组合在一起计算了
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)
