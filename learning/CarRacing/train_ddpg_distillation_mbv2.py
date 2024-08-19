#!/usr/bin/env python3
'''
未验证
python .\train_ddpg_distillation_mbv2.py -n caracing
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

SAVE_ITERS = 10
TEST_ITERS = 100

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
    print("Start Test")
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

    save_path = os.path.join("saves", "ddpg-distillation-" + args.name)

    env = TransposeObservation(gym.make(ENV_ID, domain_randomize=True, continuous=True))
    test_env = TransposeObservation(gym.make(ENV_ID, continuous=True))

    # 构建动作网络和评价网络
    act_net = model.DDPGActor(env.observation_space.shape, env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCritic(env.observation_space.shape, env.action_space.shape[0]).to(device)
    # todo 确认是否需要将模型设置为eval模式
    act_mbv2_net = model.DDPGActorMBv2(env.observation_space.shape, env.action_space.shape[0]).to(device)
    crt_mbv2_net = model.DDPGCriticMBv2(env.observation_space.shape, env.action_space.shape[0]).to(device)
    tgt_act_mbv2_net = ptan.agent.TargetNet(act_mbv2_net)
    tgt_crt_mbv2_net = ptan.agent.TargetNet(crt_mbv2_net)
    if (os.path.exists(os.path.join(save_path, "best_756.dat"))):
        act_net.load_state_dict(torch.load(os.path.join(save_path, "best_756.dat")))
        print("加载act模型成功")

    if (os.path.exists(os.path.join(save_path, "best_crt_756.dat"))):
        crt_net.load_state_dict(torch.load(os.path.join(save_path, "best_crt_756.dat")))
        print("加载crt模型成功")

    if (os.path.exists(os.path.join(save_path, "act-distillation-mbv2.dat"))):
        act_mbv2_net.load_state_dict(torch.load(os.path.join(save_path, "act-distillation-mbv2.dat")))
        print("加载act-mbv2模型成功")

    if (os.path.exists(os.path.join(save_path, "crt-distillation_mbv2.dat"))):
        crt_mbv2_net.load_state_dict(torch.load(os.path.join(save_path, "crt-distillation_mbv2.dat")))
        print("加载crt-mbv2模型成功")

    if (os.path.exists(os.path.join(save_path, "act-distillation-mbv2-tgt.dat"))):
        tgt_act_mbv2_net.target_model.load_state_dict(torch.load(os.path.join(save_path, "act-distillation-mbv2-tgt.dat")))
        print("加载tgt_act-mbv2模型成功")

    if (os.path.exists(os.path.join(save_path, "crt-distillation-mbv2-tgt.dat"))):
        tgt_crt_mbv2_net.target_model.load_state_dict(torch.load(os.path.join(save_path, "crt-distillation-mbv2-tgt.dat")))
        print("加载tgt_crt-mbv2模型成功")

    writer = SummaryWriter(comment="-ddpg-distillation-" + args.name)
    # 构建DDPG代理
    agent = model.AgentDDPG(act_mbv2_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_mbv2_opt = optim.Adam(act_mbv2_net.parameters(), lr=LEARNING_RATE)
    crt_mbv2_opt = optim.Adam(crt_mbv2_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    result_queue = multiprocessing.Queue()
    test_process = None

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

                critic_loss_list = []
                q_mean_list = []
                actor_loss_list = []

                # 从缓冲区里面采样数据

                batch_total = buffer.sample(BATCH_SIZE)
                for i in range(0, BATCH_SIZE, MIN_BATCH_SIZE):
                    batch = batch_total[i:i+MIN_BATCH_SIZE]
                    states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)

                    # train critic
                    crt_mbv2_opt.zero_grad()
                    # 根据状态和动作，得到评价，这里是根据实际游戏的状态和动作获取评价
                    with torch.no_grad():
                        q_v_best = crt_net(states_v, actions_v)
                    q_mbv2_v = crt_mbv2_net(states_v, actions_v)
                    # 计算预测的当前Q值和Bellman计算的到的Q值之间的差异
                    # 并更新梯度 这里的方式就和之前的Q值单元的一致
                    critic_best_loss_v = F.mse_loss(q_mbv2_v, q_v_best)

                    last_act_v = tgt_act_mbv2_net.target_model(last_states_v)
                    # 使用目标评测网络，根据下一个状态和下一个状态将要执行的动作得到下一个状态的评价Q值
                    q_last_v = tgt_crt_mbv2_net.target_model(last_states_v, last_act_v)
                    # 如果是结束状态则将奖励置为0
                    q_last_v[dones_mask.bool()] = 0.0
                    # 计算Q值 bellman公式
                    q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                    # 计算预测的当前Q值和Bellman计算的到的Q值之间的差异
                    # 并更新梯度 这里的方式就和之前的Q值单元的一致
                    critic_loss_v = F.mse_loss(q_mbv2_v, q_ref_v.detach())

                    total_crt_loss_v = 0.3 * critic_best_loss_v + 0.7 * critic_loss_v

                    total_crt_loss_v.backward()
                    crt_mbv2_opt.step()
                    critic_loss_list.append(total_crt_loss_v.item())
                    q_mean_list.append(q_ref_v.mean().item())

                    # train actor
                    act_mbv2_opt.zero_grad()
                    # 预测动作
                    with torch.no_grad():
                        cur_actions_v = act_net(states_v)
                    cur_actions_mbv2_v = act_mbv2_net(states_v)
                    # 根据状态和预测的动作计算Q值的负值
                    # 这里是根据网络预测的动作和实际的状态获取评价
                    # 在评价前取负号，就是简单粗暴的取最小值，从而达到最大值Q值的目的
                    # 由于这里评价网络是固定是，所以最大化Q值，只有更新预测的动作，使得
                    # 评价Q值达到最大值的目的
                    actor_best_loss_v = F.mse_loss(cur_actions_mbv2_v, cur_actions_v)

                    actor_loss_v = -crt_mbv2_net(states_v, cur_actions_mbv2_v)
                    actor_loss_v = actor_loss_v.mean()

                    total_actor_loss_v = 0.3 * actor_best_loss_v + 0.7 * actor_loss_v
                    total_actor_loss_v.backward()
                    act_mbv2_opt.step()
                    actor_loss_list.append(total_actor_loss_v.item())

                # 将训练网路同步到目标网络上，但是这里是每次都同步，与之前每隔n步同步一次不同
                # 这里之所以这样做，是根据测试可知，每次都同步，并使用较小的权重进行同步
                # 缓存的同步效果更好，并且能够保持平滑的更新
                tgt_act_mbv2_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_mbv2_net.alpha_sync(alpha=1 - 1e-3)

                tb_tracker.track("loss_critic", np.mean(critic_loss_list), frame_idx)
                tb_tracker.track("critic_ref", np.mean(q_mean_list), frame_idx)
                tb_tracker.track("loss_actor", np.mean(actor_loss_list), frame_idx)

                if frame_idx % SAVE_ITERS == 0:
                    print(f"Train done and actor_loss_v is {np.mean(actor_loss_list)} and critic_loss_v is {np.mean(critic_loss_list)}")
                    torch.save(act_mbv2_net.state_dict(), os.path.join(save_path, f"act-distillation-mbv2.dat"))
                    torch.save(crt_mbv2_net.state_dict(), os.path.join(save_path, f"crt-distillation_mbv2.dat"))
                    torch.save(tgt_act_mbv2_net.target_model.state_dict(), os.path.join(save_path, f"act-distillation-mbv2-tgt.dat"))
                    torch.save(tgt_crt_mbv2_net.target_model.state_dict(), os.path.join(save_path, f"crt-distillation-mbv2-tgt.dat"))

                if frame_idx % TEST_ITERS == 0:
                    # 测试并保存最好测试结果的庶数据
                    # 启动测试进程
                    if test_process is None or not test_process.is_alive():
                        net_state_dict = copy.deepcopy(act_mbv2_net.state_dict())
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
                            name = "best_migrate_mbv2_%+.3f_%d.dat" % (rewards, frame_idx)
                            crt_name = "best_migrate_mbv2_crt_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            crt_fname = os.path.join(save_path, crt_name)
                            torch.save(act_mbv2_net.state_dict(), fname)
                            torch.save(crt_mbv2_net.state_dict(), crt_fname)
                        best_reward = rewards

    pass
