#!/usr/bin/env python3
'''
未验证
'''
import gymnasium as gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


def calc_loss(batch, net, tgt_net, gamma, device="cpu", double=True):
    '''
    添加了计算双dqn的损失方式
    '''
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # 更新网络计算当前状态下，执行动作得到的Q值
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # 判断是否是双步DQN
    if double:
        # 更细网络根据下一个状态得到能够得到最大q值执行的动作
        next_state_actions = net(next_states_v).max(1)[1]
        # 目标网络根据下一个状态得到Q值后，根据更新网络得到的执行动作，选择下一个状态的最大Q值
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        # 通过以上步骤，就实现了通过两个DQN网络，计算出下一个状态的最大Q值
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--double", default=False, action="store_true", help="Enable double DQN")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make("BreakoutNoFrameskip-v4")
    env = ptan.common.wrappers.wrap_dqn(env)

    # 创建训练网络
    writer = SummaryWriter(comment="-" + "dqn-double" + "-double=" + str(args.double))
    net = dqn_model.DQNBreakOut(env.observation_space.shape, env.action_space.n).to(device)

    # 创建目标网络
    # 创建Epsilon训练动作选择器
    # 根据这两个参数得到训练网络代理器
    tgt_net = ptan.agent.TargetNet(net)
    # epsilon 追踪器，用来跟踪epsilon参数的变化，调整参数
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    epsilon_tracker = common.EpsilonTracker(selector, 1.0, 0.1, 10**6)
    # 这个代理的作用，就是可以根据当前的训练进度，选择从net中选择动作还是根据selector选择动作
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # 创建经验重放缓冲区
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=10**6)
    # 创建神经网络优化器
    optimizer = optim.Adam(net.parameters(), lr=2.5e-4)

    frame_idx = 0
    eval_states = None
    best_loss = 100

    with common.RewardTracker(writer, 500) as reward_tracker:
        while True:
            frame_idx += 1
            # 从经验池中获取一次样本存放在缓存中
            # 根据源码，这里执行了populate后会执行一次完整的游戏流程知道游戏结束
            buffer.populate(1)
            # 根据当前的轮数，更新epsilon值
            epsilon_tracker.frame(frame_idx)

            # 这个操作是将exp_source所有采集的激励pop出缓存并清空缓存
            # 这里清空不会导致buffer清空，因为buffer::populate操作会拷贝一份
            # 样本
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                # 判断当前的训练所获取的激励是否满足了指定的标准
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < 50000:
                continue
            if eval_states is None:
                # 从经验缓冲池中随机选择一段，作为验证池
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.asarray(transition.state) for transition in eval_states]
                eval_states = np.asarray(eval_states)

            optimizer.zero_grad()
            batch = buffer.sample(64)
            loss_v = calc_loss(batch, net, tgt_net.target_model, gamma=0.99, device=device,
                               double=args.double)
            loss_v.backward()
            optimizer.step()

            if frame_idx % 10000 == 0:
                tgt_net.sync()
                best_loss = common.save_model("double_dqn", loss_v.item(), best_loss, net.state_dict())
                common.save_model("double_dqn_tgt", loss_v.item(), best_loss, tgt_net.target_model.state_dict())

            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, frame_idx)

