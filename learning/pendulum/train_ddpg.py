#!/usr/bin/env python3
'''
验证貌似不通过，最好的奖励只有-109，得找下代码出错在哪里
通过实际的验证结果可知，实际上时已经训练成功了，只是这个奖励最高为0，所以当训练到最好的
结果时，其最好的结果奖励也是负数
另外实际上执行的动作也要根据实际的游戏进行调整，而不是直接使用clip

最后虽然拿训练好的模型去测试发现是可以完成游戏的，但是貌似这份代码里面的动作预测的范围是-1~1,而游戏的动作范围是-2~2，是否存在问题？
'''

import os
import ptan
import time
import gymnasium as gym
import argparse
from tensorboardX import SummaryWriter
import numpy as np

from lib import model, common

import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "Pendulum-v1"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000 # 重放缓冲区长度，这么长是为了提高稳定性
REPLAY_INITIAL = 10000 # 重放缓冲区初始化大小

TEST_ITERS = 1000


def test_net(net, env, count=10, device="cpu"):
    '''
    count: 执行游戏的次数（每次都是执行到游戏结束）

    return: （平均奖励，平均步数）
    '''
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            # 根据环境预测动作
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            # 执行动作
            obs, reward, done, truncated, _ = env.step(action)
            rewards += reward
            steps += 1
            if done or truncated:
                break
    return rewards / count, steps / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "td3-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID, g=9.81)
    test_env = gym.make(ENV_ID, g=9.81)

    # 构建动作网络和评价网络
    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    # 对于直接输出Q值网络，需要构建一个稳定的目标，因为Q值网络是会根据历史数据进行更新
    # 所以不能马上更新目标网络，为了稳定，否则会因为部分不稳定的数据（偶发的高分或者低分影响）
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
    # 构建DDPG代理
    agent = model.AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                # 从经验缓冲区执行一轮游戏或者执行游戏过程中采集到指定长度的游戏数据
                buffer.populate(1)
                # 提取目前位置所有的记录
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    # 记录当前的训练进度并判断是否达到了奖励目标
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                # 从缓冲区里面采样数据
                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                # 根据状态和动作，得到评价，这里是根据实际游戏的状态和动作获取评价
                q_v = crt_net(states_v, actions_v)
                # 使用目标动作预测网路，根据下一个状态预测执行的动作
                last_act_v = tgt_act_net.target_model(last_states_v)
                # 使用目标评测网络，根据下一个状态和下一个状态将要执行的动作得到下一个状态的评价Q值
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                # 如果是结束状态则将奖励置为0
                q_last_v[dones_mask.bool()] = 0.0
                # 计算Q值 bellman公式
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                # 计算预测的当前Q值和Bellman计算的到的Q值之间的差异
                # 并更新梯度 这里的方式就和之前的Q值单元的一致
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                # 预测动作
                cur_actions_v = act_net(states_v)
                # 根据状态和预测的动作计算Q值的负值
                # 这里是根据网络预测的动作和实际的状态获取评价
                # 在评价前取负号，就是简单粗暴的取最小值，从而达到最大值Q值的目的
                # 由于这里评价网络是固定是，所以最大化Q值，只有更新预测的动作，使得
                # 评价Q值达到最大值的目的
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                # 将训练网路同步到目标网络上，但是这里是每次都同步，与之前每隔n步同步一次不同
                # 这里之所以这样做，是根据测试可知，每次都同步，并使用较小的权重进行同步
                # 缓存的同步效果更好，并且能够保持平滑的更新
                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    # 测试并保存最好测试结果的庶数据
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

    pass
