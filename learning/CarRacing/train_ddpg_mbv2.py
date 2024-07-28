#!/usr/bin/env python3
'''
未验证
python .\train_ddpg_mbv2.py -n caracing_mbv2
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
MIN_BATCH_SIZE = 16
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


import copy
import multiprocessing

def test_process_func(net_state_dict, env, frame_idx, device, result_queue):
    net = model.DDPGActorMBv2(env.observation_space.shape, env.action_space.shape[0]).to(device)
    net.load_state_dict(net_state_dict)
    
    ts = time.time()
    rewards, steps = test_net(net, env, device=device)
    print("Test done in %.2f sec, reward %.3f, steps %d" % (
        time.time() - ts, rewards, steps))
    
    result_queue.put((frame_idx, rewards, steps))


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ddpg-mbv2-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = TransposeObservation(gym.make(ENV_ID, domain_randomize=True, continuous=True))
    test_env = TransposeObservation(gym.make(ENV_ID, continuous=True))

    # 构建动作网络和评价网络
    act_net = model.DDPGActorMBv2(env.observation_space.shape, env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCriticMBv2(env.observation_space.shape, env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    print(f"act_net Number of parameters: {common.count_parameters(act_net)}")
    print(f"crt_net Number of parameters: {common.count_parameters(crt_net)}")

    if (os.path.exists(os.path.join(save_path, "act-mbv2-0.pth"))):
        act_net.load_state_dict(torch.load(os.path.join(save_path, "act-mbv2-0.pth")))
        print("加载act模型成功")

    if (os.path.exists(os.path.join(save_path, "crt-mbv2-0.pth"))):
        crt_net.load_state_dict(torch.load(os.path.join(save_path, "crt-mbv2-0.pth")))
        print("加载crt模型成功")
    # 对于直接输出Q值网络，需要构建一个稳定的目标，因为Q值网络是会根据历史数据进行更新
    # 所以不能马上更新目标网络，为了稳定，否则会因为部分不稳定的数据（偶发的高分或者低分影响）
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg-mbv2_" + args.name)
    # 构建DDPG代理
    agent = model.AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    result_queue = multiprocessing.Queue()
    test_process = None

    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            # start_buffer_time = time.time()
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

                # print(f"Buffer time: {time.time() - start_buffer_time}")

                critic_loss_list = []
                q_mean_list = []
                actor_loss_list = []
                # start_train_time = time.time()
                # 从缓冲区里面采样数据
                batch_total = buffer.sample(BATCH_SIZE)
                for i in range(0, BATCH_SIZE, MIN_BATCH_SIZE):
                    batch = batch_total[i:i+MIN_BATCH_SIZE]
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
                    critic_loss_list.append(critic_loss_v.item())
                    q_mean_list.append(q_ref_v.mean().item())

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

                    actor_loss_list.append(actor_loss_v.item())


                # 将训练网路同步到目标网络上，但是这里是每次都同步，与之前每隔n步同步一次不同
                # 这里之所以这样做，是根据测试可知，每次都同步，并使用较小的权重进行同步
                # 缓存的同步效果更好，并且能够保持平滑的更新
                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                tb_tracker.track("loss_critic", np.mean(critic_loss_list), frame_idx)
                tb_tracker.track("critic_ref", np.mean(q_mean_list), frame_idx)
                tb_tracker.track("loss_actor", np.mean(actor_loss_list), frame_idx)

                # print(f"Train time: {time.time() - start_train_time}")

                if frame_idx % TEST_ITERS == 0:
                    # # 测试并保存最好测试结果的庶数据
                    # ts = time.time()
                    # rewards, steps = test_net(act_net, test_env, device=device)
                    # print("Test done in %.2f sec, reward %.3f, steps %d" % (
                    #     time.time() - ts, rewards, steps))
                    # writer.add_scalar("test_reward", rewards, frame_idx)
                    # writer.add_scalar("test_steps", steps, frame_idx)
                    # if best_reward is None or best_reward < rewards:
                    #     if best_reward is not None:
                    #         print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    #         name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                    #         crt_name = "best_crt_%+.3f_%d.dat" % (rewards, frame_idx)
                    #         fname = os.path.join(save_path, name)
                    #         crt_fname = os.path.join(save_path, crt_name)
                    #         torch.save(act_net.state_dict(), fname)
                    #         torch.save(crt_net.state_dict(), crt_fname)
                    #     best_reward = rewards
                                    #保存act模型和crt模型
                    torch.save(act_net.state_dict(), os.path.join(save_path, f"act-mbv2-{frame_idx % 10}.pth"))
                    torch.save(crt_net.state_dict(), os.path.join(save_path, f"crt-mbv2-{frame_idx % 10}.pth"))

                if frame_idx % TEST_ITERS == 0:
                    # 启动测试进程
                    if test_process is None or not test_process.is_alive():
                        net_state_dict = copy.deepcopy(act_net.state_dict())
                        test_process = multiprocessing.Process(
                            target=test_process_func,
                            args=(net_state_dict, test_env, frame_idx, device, result_queue)
                        )
                        test_process.start()

                # 检查是否有测试结果
                if not result_queue.empty():
                    test_frame_idx, rewards, steps = result_queue.get()
                    writer.add_scalar("test_reward", rewards, test_frame_idx)
                    writer.add_scalar("test_steps", steps, test_frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best-mbv2_%+.3f_%d.dat" % (rewards, test_frame_idx)
                            crt_name = "best-mbv2_crt_%+.3f_%d.dat" % (rewards, test_frame_idx)
                            fname = os.path.join(save_path, name)
                            crt_fname = os.path.join(save_path, crt_name)
                            torch.save(act_net.state_dict(), fname)
                            torch.save(crt_net.state_dict(), crt_fname)
                        best_reward = rewards

                # start_buffer_time = time.time()


    pass
