#!/usr/bin/env python3
'''
未适配
'''
import gymnasium as gym
import ptan
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common
import os

REWARD_STEPS_DEFAULT = 2

save_path = "saves"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", default=REWARD_STEPS_DEFAULT, type=int, help="Count of steps to unroll Bellman")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # 创建游戏环境
    env = gym.make("BreakoutNoFrameskip-v4")
    env = ptan.common.wrappers.wrap_dqn(env)

    # 创建训练网络
    writer = SummaryWriter(comment="-" + "dqn-n-steps" + "-%d-step" % args.n)
    net = dqn_model.DQNBreakOut(env.observation_space.shape, env.action_space.n).to(device)

    if (os.path.exists(os.path.join(save_path, "model_n_steps_dqn.dat"))):
        net.load_state_dict(torch.load(os.path.join(save_path, "model_n_steps_dqn.dat")))
        print("加载模型成功")
    # 创建目标网络
    # 创建Epsilon训练动作选择器
    # 根据这两个参数得到训练网络代理器
    tgt_net = ptan.agent.TargetNet(net)

    if (os.path.exists(os.path.join(save_path, "model_n_steps_dqn_tgt.dat"))):
        tgt_net.target_model.load_state_dict(torch.load(os.path.join(save_path, "model_n_steps_dqn_tgt.dat")))
        print("加载目标网络成功")

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    epsilon_tracker = common.EpsilonTracker(selector, 1.0, 0.1, 10**6)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # 创建经验重放缓冲区
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=args.n)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=10**6)
    # 创建神经网络优化器
    optimizer = optim.Adam(net.parameters(), lr=2.5e-4)

    frame_idx = 0
    best_loss = 100

    with common.RewardTracker(writer, 500) as reward_tracker:
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
                # 判断是否已经达到了预期激励值，如果达到了，将退出训练
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    # 将当前采集的首个激励，帧总数传入到reward_tracker
                    # 如果平均激励值已经达到了所要的目标值，则break跳出循环
                    break

            # 如果训练缓存中的训练数据未达到目标的缓存大小，则需要继续进行缓存采集直到达到指定的数量后
            # 再开始训练，
            if len(buffer) < 50000:
                continue

            # 清空优化器的梯度
            optimizer.zero_grad()
            # 从经验缓冲区中采集batch_size大小的样本
            batch = buffer.sample(64)
            # 计算损失值并更新神经网路
            # 根据网络上的资料显示，这里之所以仅做了n次方，是因为容许误差，所以就省略了中间
            # 步骤的max操作，直接做了n次方，这样做的好处是可以减少计算量，但同时也限制了n步的大小
            # 这里计算的损失对应书中第120页
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model,
                                          gamma=0.99**args.n, device=device)
            loss_v.backward()
            optimizer.step()

            # 如果当前的轮次已经达到了同步到目标网络的轮次，则进行目标网络的同步更新
            if frame_idx % 5000 == 0:
                tgt_net.sync()
                best_loss = common.save_model("n_steps_dqn", loss_v.item(), best_loss, net.state_dict())
                common.save_model("n_steps_dqn_tgt", loss_v.item(), best_loss, tgt_net.target_model.state_dict())
