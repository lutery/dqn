#!/usr/bin/env python3
'''
实现基于trpo的训练方法
验证算法通过，算法有效，且比DDPG要更快达到最好的奖励
'''
import os
import math
import ptan
import time
import gymnasium as gym
import argparse
from tensorboardX import SummaryWriter

from lib import model, trpo

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "BipedalWalker-v3"
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_CRITIC = 1e-3

TRPO_MAX_KL = 0.01
TRPO_DAMPING = 0.1

TEST_ITERS = 100000


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
    """
    计算连续纸高斯概率分布的对应动作的概率
    """
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2



def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    # 对状态评估Q值
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    # val 当前状态的Q值，到达的下一个状态的Q值，当前的的经验
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            # 如果执行的动作导致游戏结束
            # delta = 获取的奖励（实际的Q值） - 预测的Q值 计算动作优势
            delta = exp.reward - val
            last_gae = delta
        else:
            # delta = 实际的Q值 - 预测的Q值 是在计算动作优势
            delta = exp.reward + GAMMA * next_val - val
            # 这里是在计算广义优势值GAE，需要考虑到后续多步的GAE优势值，并将其融入到当前的gae优势，很容易理解，如果所要计算的未来是较长的未来（也就是动作可以使得游戏保持长久的运行），那么gae就会比较大；而所要计算的未来是游戏结束的状态，那么即使一步优势大，但是总体来说会相对执行步数长的gae小，这样就可以保证训练的稳定性，从而让模型走向稳定，正确的收敛方向。
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default=" + ENV_ID)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "trpo-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(args.env)
    test_env = gym.make(args.env)

    # 创建动作预测网络
    net_act = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # 创建状态、动作评价网络
    net_crt = model.ModelCritic(env.observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment="-trpo_" + args.name)
    agent = model.AgentTRPO(net_act, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    trajectory = [] # 经验轨迹，不同之处在于这里序列必须是连续的，而不是随机采样的
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

            if step_idx % TEST_ITERS == 0:
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

            trajectory.append(exp)
            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
            # 预测执行的动作
            mu_v = net_act(traj_states_v)
            # 得到执行动作的概率
            old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

            # normalize advantages
            # 正则话优势
            traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()
            traj_states_v = traj_states_v[:-1]
            traj_actions_v = traj_actions_v[:-1]
            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            # critic step
            # 这边应该是对预测的Q值进行优化计算
            opt_crt.zero_grad()
            value_v = net_crt(traj_states_v)
            loss_value_v = F.mse_loss(value_v.squeeze(-1), traj_ref_v)
            loss_value_v.backward()
            opt_crt.step()

            # actor step
            def get_loss():
                mu_v = net_act(traj_states_v)
                logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)
                action_loss_v = -traj_adv_v.unsqueeze(dim=-1) * torch.exp(logprob_v - old_logprob_v)
                return action_loss_v.mean()

            def get_kl():
                mu_v = net_act(traj_states_v)
                logstd_v = net_act.logstd
                mu0_v = mu_v.detach()
                logstd0_v = logstd_v.detach()
                std_v = torch.exp(logstd_v)
                std0_v = std_v.detach()
                kl = logstd_v - logstd0_v + (std0_v ** 2 + (mu0_v - mu_v) ** 2) / (2.0 * std_v ** 2) - 0.5
                return kl.sum(1, keepdim=True)

            trpo.trpo_step(net_act, get_loss, get_kl, TRPO_MAX_KL, TRPO_DAMPING, device=device)

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_value", loss_value_v.item(), step_idx)

