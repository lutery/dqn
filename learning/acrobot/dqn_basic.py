#!/usr/bin/env python3
'''
验证通过，可以再比较少的步数是的自由端达到目标高度，奖励值达到-100以内
'''
import os
import gymnasium as gym
import ptan
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

save_path = "saves/dqn_basic"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建对应的gym环境
    env = gym.make("Acrobot-v1")
    # 为创建的gym环境扩展需要的属性
    # 创建tensor board的存储目录
    writer = SummaryWriter(comment="Acrobot-dqn-basic")
    # 创建dqn网络模型
    net = dqn_model.DQNAcrobot(env.observation_space.shape[0], env.action_space.n).to(device)

    # 考虑dqn网络，创建一个目标网络,负责计算在线网络net的目标q值
    tgt_net = ptan.agent.TargetNet(net)
    if (os.path.exists(os.path.join(save_path, "net.pth"))):
        net.load_state_dict(torch.load(os.path.join(save_path, "net.pth")))
        print("加载模型成功")

    if (os.path.exists(os.path.join(save_path, "tgt_net.pth"))):
        tgt_net.model.load_state_dict(torch.load(os.path.join(save_path, "tgt_net.pth")))
        print("加载目标模型成功")

    # 创建一个基于epsilon进行动作动作选择的选择器
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    # epsion的值含义:epsion标识一个会随着训练的增加而逐渐减少的随机值
    # 表示训练一开始以随机选择执行的动作为主，因为此时网络并没有训练，进行随机探索
    # 随着训练的增加，网络已经具备一些情况下的决策能力，可以进行动作的选择
    epsilon_tracker = common.EpsilonTracker(selector, 1.0, 0.1, 10**6)
    # 创建一个dqn推理网络的代理
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # 创建经验重放缓冲区（也就是训练过程中采集的样本）
    # ExperienceSourceFirstLast应该是存储样本的缓存去
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1)
    # 第二个应该是从缓存区提取一小段样本进行训练的
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=3*10 ** 5)

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # 统计所执行经理的帧数（轮数）
    frame_idx = 0

    # 创建一个奖励跟踪器
    with common.RewardTracker(writer, stop_reward=-100) as reward_tracker:
        while True:
            frame_idx += 1
            # 从经验池中获取一次样本存放在缓存中
            buffer.populate(1)
            # 更新epsilon值
            epsilon_tracker.frame(frame_idx)

            # 这个操作是将所有采集的激励pop出缓存并清空缓存
            # 这里清空仅仅只清空exp_source中的记录，在buffer中的记录并没有清空
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                # 这里主要是判断最近的一次激励是否达到了目标值，达到则停止训练
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    # 将当前采集的首个激励，帧总数传入到reward_tracker
                    # 如果平均激励值已经达到了所要的目标值，则break跳出循环
                    break

            # 检查经验池中的样本长度是否达到了目标的长度大小，
            if len(buffer) < 20000:
                continue

            # 清空优化器的梯度
            optimizer.zero_grad()
            # 从经验缓冲区中采集batch_size大小的样本
            batch = buffer.sample(64)
            # 计算损失值并更新神经网路
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=0.99, device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % 1000 == 0:
                tgt_net.sync()
            if frame_idx % 10000 == 0:
                torch.save(net.state_dict(), os.path.join(save_path, "net.pth"))
                torch.save(tgt_net.model.state_dict(), os.path.join(save_path, "tgt_net.pth"))
