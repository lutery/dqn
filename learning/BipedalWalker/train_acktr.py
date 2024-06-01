#!/usr/bin/env python3
'''
验证通过，但是最高分比不上ddpg等其他的算法
'''
import os
import math
import ptan
import time
import gymnasium as gym
import argparse
from tensorboardX import SummaryWriter

from lib import model, common, kfac

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "BipedalWalker-v3"
GAMMA = 0.99
REWARD_STEPS = 5 # 环境展开步数的长度
BATCH_SIZE = 32
# 动作网络和评价网络的学习率
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
# todo 作用
ENTROPY_BETA = 1e-3
# todo 环境数量
ENVS_COUNT = 16

TEST_ITERS = 100000 # 训练到第几次采样时，进行一次测试 todo 这里是步数还是游戏轮数


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
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


def calc_logprob(mu_v, logstd_v, actions_v):
    '''
        使用高斯密度函数，计算连续值的概率
        '''
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default=" + ENV_ID)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join("saves", "acktr-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    # 创建训练多个环境
    envs = [gym.make(args.env) for _ in range(ENVS_COUNT)]
    # 创建单个测试环境
    test_env = gym.make(args.env)

    net_act = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
    net_crt = model.ModelCritic(envs[0].observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment="-acktr_" + args.name)
    agent = model.AgentAcktr(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)

    # 动作预测网路使用了自己实现的优化器
    opt_act = kfac.KFACOptimizer(net_act, lr=LEARNING_RATE_ACTOR)
    # 评价网络使用了Adam优化器
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    # 依旧是绘制最近100次的平均奖励和步数
                    # todo 这100次是步数还是轮数
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % TEST_ITERS == 0:
                    # 测试网络，并根据回报奖励更新最优网络
                    ts = time.time()
                    rewards, steps = test_net(net_act, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net_act.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_acktr(batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                # 这里依旧是评级网路预测的评价需要与计算的Q值接近
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                # 预测动作
                mu_v = net_act(states_v)
                # 计算概率
                log_prob_v = calc_logprob(mu_v, net_act.logstd, actions_v)
                if opt_act.steps % opt_act.Ts == 0:
                    # todo 这段代码的作用
                    # 这行检查是否达到了更新策略网络的步骤。opt_act.Ts 是一个预设的阈值，表示每隔多少步更新一次策略网络
                    opt_act.zero_grad()
                    # todo 这行代码将动作概率平均值取反，如果计算最小值，则会变成将预测的动作与实际的动作相接近
                    # 这行计算策略梯度损失的反向（因为优化器通常是最小化损失）。log_prob_v 是一个包含选取特定动作的对数概率的变量。它的平均值被用作损失函数，这是为了计算 Fisher 信息。
                    pg_fisher_loss = -log_prob_v.mean()
                    # 在计算梯度之前，这行启用了K-FAC优化器的状态累积。这意味着在后续的反向传播中，优化器会收集和累积计算所需的统计数据（如Fisher信息矩阵的近似）。
                    opt_act.acc_stats = True
                    # 这行执行反向传播，计算关于策略网络参数的梯度。retain_graph=True 参数意味着在反向传播后保留计算图，这通常是因为后续还会再次使用到这个图
                    pg_fisher_loss.backward(retain_graph=True)
                    # 完成了必要的梯度和统计信息计算后，关闭状态累积。
                    opt_act.acc_stats = False

                # 以下这段是和之前一样的普通连续值最大话的损失计算
                # 不同点在于优化方式采用了二阶导数的优化器
                opt_act.zero_grad()
                # 计算优势
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                # 计算优势和动作概率的损失
                loss_policy_v = -(adv_v * log_prob_v).mean()
                # 应该是预测动作的计算熵损失，这里仅让方差参与了计算
                # ENTROPY_BETA * (-(torch.log 这里的负号表示我们希望最大化熵（因为通常我们最小化损失）。计算所有动作熵的平均值。
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                # 计算总损失最小化
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
