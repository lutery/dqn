#!/usr/bin/env python3
'''
python .\train_ddpg_simple.py -n caracing_simple

todo 根据以下训练记录可知，貌似是可以提高的，还有待实验
1001: done 1 episodes, mean reward -49.254, speed 49.03 f/s
2001: done 2 episodes, mean reward -53.721, speed 51.34 f/s
3001: done 3 episodes, mean reward -48.791, speed 52.23 f/s
4001: done 4 episodes, mean reward -52.118, speed 47.59 f/s
5001: done 5 episodes, mean reward -48.610, speed 49.69 f/s
6001: done 6 episodes, mean reward -53.552, speed 51.00 f/s
7001: done 7 episodes, mean reward -52.814, speed 53.48 f/s
8001: done 8 episodes, mean reward -50.762, speed 53.51 f/s
9001: done 9 episodes, mean reward -52.649, speed 54.86 f/s
Test done in 186.86 sec, reward -93.294, steps 1000
10001: done 10 episodes, mean reward -53.655, speed 4.83 f/s
Test done in 189.01 sec, reward -91.472, steps 1000
Best reward updated: -93.294 -> -91.472
11001: done 11 episodes, mean reward -50.051, speed 2.07 f/s
Test done in 188.15 sec, reward -91.163, steps 1000
Best reward updated: -91.472 -> -91.163
12001: done 12 episodes, mean reward -51.275, speed 2.05 f/s
Test done in 191.79 sec, reward -88.097, steps 1000
Best reward updated: -91.163 -> -88.097
13001: done 13 episodes, mean reward -53.270, speed 2.02 f/s
Test done in 196.94 sec, reward -10.841, steps 1000
Best reward updated: -88.097 -> -10.841
14001: done 14 episodes, mean reward -52.526, speed 1.99 f/s
Test done in 195.29 sec, reward 83.870, steps 1000
Best reward updated: -10.841 -> 83.870
15001: done 15 episodes, mean reward -54.166, speed 2.02 f/s
Test done in 196.78 sec, reward 30.166, steps 1000
16001: done 16 episodes, mean reward -50.824, speed 2.01 f/s
Test done in 198.75 sec, reward 100.002, steps 1000
Best reward updated: 83.870 -> 100.002
17001: done 17 episodes, mean reward -49.674, speed 2.01 f/s
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


ENV_ID = "CarRacing-v2"
GAMMA = 0.99
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000 # 重放缓冲区长度，这么长是为了提高稳定性
REPLAY_INITIAL = 10000 # 重放缓冲区初始化大小

TEST_ITERS = 1000

class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeObservation, self).__init__(env)

    def observation(self, observation):
        # 将观察从 (H, W, C) 转换为 (C, H, W)
        return observation.transpose(2, 0, 1)


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
            mu_v, _ = net(obs_v)
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
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ddpg-simple-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = TransposeObservation(gym.make(ENV_ID, domain_randomize=True, continuous=True))
    test_env = TransposeObservation(gym.make(ENV_ID, continuous=True))

    # 构建动作网络和评价网络
    act_crt_net = model.DDPGActorSimple(env.observation_space.shape, env.action_space.shape[0]).to(device)
    print(act_crt_net)
    if (os.path.exists(os.path.join(save_path, "act-crt-0.pth"))):
        act_crt_net.load_state_dict(torch.load(os.path.join(save_path, "act-crt-0.pth")))
        print("加载act-crt-模型成功")

    # 对于直接输出Q值网络，需要构建一个稳定的目标，因为Q值网络是会根据历史数据进行更新
    # 所以不能马上更新目标网络，为了稳定，否则会因为部分不稳定的数据（偶发的高分或者低分影响）
    tgt_act_crt_net = ptan.agent.TargetNet(act_crt_net)

    writer = SummaryWriter(comment="-simple-ddpg_" + args.name)
    # 构建DDPG代理
    agent = model.AgentDDPGSimple(act_crt_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_crt_net.get_action_parameter(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(act_crt_net.get_qvalue_parameter(), lr=LEARNING_RATE)

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
                act_crt_net.set_train_action(True)
                # 根据状态和动作，得到评价，这里是根据实际游戏的状态和动作获取评价
                _, q_v = act_crt_net(states_v, actions_v)
                # 使用目标动作预测网路，根据下一个状态预测执行的动作
                last_act_v, q_last_v = tgt_act_crt_net.target_model(last_states_v)
                # 如果是结束状态则将奖励置为0
                q_last_v[dones_mask.bool()] = 0.0
                # 计算Q值 bellman公式
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                # 计算预测的当前Q值和Bellman计算的到的Q值之间的差异
                # 并更新梯度 这里的方式就和之前的Q值单元的一致
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                act_crt_net.set_train_action(False)
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                act_crt_net.set_train_qvalue(True)
                # 预测动作
                cur_actions_v, _ = act_crt_net(states_v)
                actor_loss_v = -(act_crt_net(states_v, cur_actions_v)[1])
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                act_crt_net.set_train_qvalue(False)
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                # 将训练网路同步到目标网络上，但是这里是每次都同步，与之前每隔n步同步一次不同
                # 这里之所以这样做，是根据测试可知，每次都同步，并使用较小的权重进行同步
                # 缓存的同步效果更好，并且能够保持平滑的更新
                tgt_act_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    # 测试并保存最好测试结果的庶数据
                    ts = time.time()
                    rewards, steps = test_net(act_crt_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_crt_net.state_dict(), fname)
                        best_reward = rewards
                                    #保存act模型和crt模型
                    torch.save(act_crt_net.state_dict(), os.path.join(save_path, f"act-crt-{frame_idx % 10}.pth"))


    pass
