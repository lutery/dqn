#!/usr/bin/env python3
'''
验证失败，不知道是算法本身不行还是代码有问题
'''
import os
import time
import math
import ptan
import gymnasium as gym
import argparse
from tensorboardX import SummaryWriter

from lib import model, common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "BipedalWalker-v3"
GAMMA = 0.99 # Q值计算奖励衰减GAMMA系数
REWARD_STEPS = 2 # 计算Q值观测展开的步数
BATCH_SIZE = 32 # 训练batch 数据大小
LEARNING_RATE = 5e-5 # 学习率
ENTROPY_BETA = 1e-4 # 熵

TEST_ITERS = 1000 # 用于判断多少轮测试一次，更新保存一次网络


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        # 重置环境
        obs, _ = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            # 计算均值，在高斯分布中，均值是所有的分布中，最有可能的值，所以使用均值作为执行的动作值
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, trunc, _ = env.step(action)
            done = done or trunc
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    net = model.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)

    writer = SummaryWriter(comment="-a2c_" + args.name)
    agent = model.AgentA2C(net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                # 获取最近N轮游戏记录的奖励值以及这些轮次的游戏总共执行的步数
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    # 获取奖励、奖励对应的步数
                    rewards, steps = zip(*rewards_steps)
                    # 记录当前训练进度下游戏能够执行的步数，以及能够获取的奖励大小
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print("Test done is %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_a2c(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                # 清空收集的游戏数据
                batch.clear()

                # 清空梯度
                optimizer.zero_grad()
                # 将游戏状态传入神经网络，得到预测的均值、方差、Q值
                mu_v, var_v, value_v = net(states_v)

                # 第一个损失，预测的Q值需要和计算的Q值接近
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                # 这里就对应之前策略梯度中计算每个动作的奖励以及对应动作的概率的损失
                # 为什么采用高斯分布，而不是直接网络输出预测目标值val，然后目标值与执行的动作值做平方差等方式计算
                # 1. 计算连续值的概率的公式之一就是高斯分布
                # 2. 高斯概率计算公式输出均值和方差，均值可以代表目标动作，方差代表预测动作是否稳定，通过这两个值既可以实现探索，也可以促进网络往稳定的方向训练（但是这里并没有使用方差去进行探索），而是采用了信息熵的方式去促使网络进行探索
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                # 去负值，计算最小值
                loss_policy_v = -log_prob_v.mean()
                # 高斯概率密度分度熵的计算公式 P292页，同样这里也取了负值
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

                # 损失之和，计算最小值，更细梯度梯度
                loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)

