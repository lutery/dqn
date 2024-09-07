#!/usr/bin/env python3
'''
未整改
'''
import os
import gymnasium as gym
import ptan
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim

from lib import common, i2a


LEARNING_RATE = 5e-4
NUM_ENVS = 16
BATCH_SIZE = 64
SAVE_EVERY_BATCH = 1000
# todo 为什么权重要这么设置
OBS_WEIGHT = 10.0
REWARD_WEIGHT = 1.0

def get_obs_diff(prev_obs, cur_obs):
    '''
    对比两个观察状态，得到他们之间的差异
    差异是直接相减得到的，所以差异的维度也和他们本身的维度一致
    '''
    prev = np.array(prev_obs)[-1]
    cur = np.array(cur_obs)[-1]
    prev = prev.astype(np.float32) / 255.0
    cur = cur.astype(np.float32) / 255.0
    return cur - prev


def iterate_batches(envs, net, device="cpu"):
    '''
    param envs: 游戏环境
    param net: 网络

    return mb_obs（上一次的环境状态观察列表）, mb_obs_next（存储当前状态和上一次状态的差异，和mb_obs一一对应）, mb_actions（存储造成差异对应的执行的动作）, mb_rewards（存储执行动作后得到的奖励，和mb_obs一一对应）, done_rewards（存储存在游戏结束状态时累计的总奖励）, done_steps（存储存在游戏结束时状态累计的总步数）
    '''
    # 创建随机动作选择器
    act_selector = ptan.actions.ProbabilityActionSelector()
    # mb_obs shape is (BATCH_SIZE, FRAMES_COUNT, 84, 84)
    # 创建观察数据的，值全为0
    mb_obs = np.zeros((BATCH_SIZE, ) + common.IMG_SHAPE, dtype=np.uint8)
    # 存储由上一次观察的环境状态和当前观察的环境状态得到差异和mb_obs_next一一对应
    mb_obs_next = np.zeros((BATCH_SIZE, ) + i2a.EM_OUT_SHAPE, dtype=np.float32)
    # 保存执行的动作，和mb_obs一一对应
    mb_actions = np.zeros((BATCH_SIZE, ), dtype=np.int32)
    # 保存执行动作后得到的奖励回报，mb_actions一一对应
    mb_rewards = np.zeros((BATCH_SIZE, ), dtype=np.float32)
    # 重置环境，将重置后的环境状态存放到obs中，这里存的应该是上一次的环境状态
    obs = [e.reset() for e in envs]
    # 保存每个环境获取的总奖励回报
    total_reward = [0.0] * NUM_ENVS
    # 保存每个环境执行的总步数
    total_steps = [0] * NUM_ENVS
    batch_idx = 0
    done_rewards = []
    done_steps = []

    while True:
        # 使用默认的环境状态代理器，将环境转换为矩阵
        obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
        # 网络根据输入的状态预测出动作的概率以及预测状态回报奖励
        logits_v, values_v = net(obs_v)
        probs_v = F.softmax(logits_v, dim=1)
        # 将预测的动作大小转换为概率
        probs = probs_v.data.cpu().numpy()
        # 根据概率选择执行的动作
        actions = act_selector(probs)

        # 遍历每一个环境e_idx表示当前遍历的是哪个环境的索引
        for e_idx, e in enumerate(envs):
            # 根据环境的index，按顺序提取动作后传给环境，执行动作
            # 得到下一帧的观察图像、回报奖励、是否结束、额外信息
            o, r, done, _ = e.step(actions[e_idx])
            # 将上一次环境返回的状态保存到mb_obs中
            mb_obs[batch_idx] = obs[e_idx]
            # 对比上一次的观察状态和当前的观察状态，得到差异，保存到mb_obs_next中
            mb_obs_next[batch_idx] = get_obs_diff(obs[e_idx], o)
            # 保存当前执行的动作到mb_actions
            mb_actions[batch_idx] = actions[e_idx]
            # 保存当前执行动作后的回报奖励到mb_rewards
            mb_rewards[batch_idx] = r
            # 统计包奖励回报以及总步数
            total_reward[e_idx] += r
            total_steps[e_idx] += 1

            # batch_idx+1，如果超过BATCH_SIZE，则归零
            batch_idx = (batch_idx + 1) % BATCH_SIZE
            if batch_idx == 0:
                # 如果batch_idx等于0，说明已经收集了BATCH_SIZE个数据
                # 可以进行一次训练
                # 则返回当前收集的结果
                yield mb_obs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps
                # 将记录的存在结束的游戏奖励缓存清零
                done_rewards.clear()
                # 将记录的存在结束的游戏步数缓存清零
                done_steps.clear()
            if done:
                # 如果执行动作后得到的是游戏结束，则重置当前遍历的环境
                o = e.reset()
                # 将游戏结束的奖励回报和总步数保存到缓存中
                # 这个缓存不会记录是哪个环境的总奖励和总步数
                done_rewards.append(total_reward[e_idx])
                done_steps.append(total_steps[e_idx])
                # 将游戏结束的奖励回报和总步数清零
                total_reward[e_idx] = 0.0
                total_steps[e_idx] = 0
            # 更新上一次的环境状态
            obs[e_idx] = o


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-m", "--model", required=True, help="File with model to load")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", "02_env_" + args.name)
    os.makedirs(saves_path, exist_ok=True)

    envs = [common.make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-02_env_" + args.name)

    # 创建A2C网络
    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n)
    net_em = i2a.EnvironmentModel(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    # 加载之前训练好的A2C网络，该A2C是01_a2c.py训练好的
    # net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    net = net.to(device)
    print(net_em)
    optimizer = optim.Adam(net_em.parameters(), lr=LEARNING_RATE)

    step_idx = 0
    best_loss = np.inf
    with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
        # 采集环境数据
        for mb_obs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps in iterate_batches(envs, net, device):
            # 如果采集到了游戏结束的数据，则计算平均奖励和平均步数
            # 并记录起来
            if len(done_rewards) > 0:
                m_reward = np.mean(done_rewards)
                m_steps = np.mean(done_steps)
                print("%d: done %d episodes, mean reward=%.2f, steps=%.2f" % (
                    step_idx, len(done_rewards), m_reward, m_steps))
                tb_tracker.track("total_reward", m_reward, step_idx)
                tb_tracker.track("total_steps", m_steps, step_idx)

            # 张量化
            obs_v = torch.FloatTensor(mb_obs).to(device)
            obs_next_v = torch.FloatTensor(mb_obs_next).to(device)
            actions_t = torch.LongTensor(mb_actions.tolist()).to(device)
            rewards_v = torch.FloatTensor(mb_rewards).to(device)

            optimizer.zero_grad()
            # 环境模型根据输入的状态和动作，预测出下一帧的状态和当前状态的差异和得到奖励回报
            out_obs_next_v, out_reward_v = net_em(obs_v.float()/255, actions_t)
            # 预测的下一帧状态和当前状态的差异和真实的差异计算损失
            loss_obs_v = F.mse_loss(out_obs_next_v.squeeze(-1), obs_next_v)
            # 预测的回报奖励和真实的回报奖励计算损失
            loss_rew_v = F.mse_loss(out_reward_v.squeeze(-1), rewards_v)
            # 计算总损失
            loss_total_v = OBS_WEIGHT * loss_obs_v + REWARD_WEIGHT * loss_rew_v
            # 计算梯度
            loss_total_v.backward()
            optimizer.step()
            # 记录损失
            tb_tracker.track("loss_em_obs", loss_obs_v, step_idx)
            tb_tracker.track("loss_em_reward", loss_rew_v, step_idx)
            tb_tracker.track("loss_em_total", loss_total_v, step_idx)

            # 记录损失总损失
            loss = loss_total_v.data.cpu().numpy()
            if loss < best_loss:
                # 如果总损失小于之前的最小损失，则更新最小损失，并保存模型
                print("Best loss updated: %.4e -> %.4e" % (best_loss, loss))
                best_loss = loss
                fname = os.path.join(saves_path, "best_%.4e_%05d.dat" % (loss, step_idx))
                torch.save(net_em.state_dict(), fname)

            step_idx += 1
            if step_idx % SAVE_EVERY_BATCH == 0:
                # 如果遍历到了SAVE_EVERY_BATCH次，也保存模型
                fname = os.path.join(saves_path, "em_%05d_%.4e.dat" % (step_idx, loss))
                torch.save(net_em.state_dict(), fname)
