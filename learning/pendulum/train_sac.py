#!/usr/bin/env python3
'''
未验证
1. 本游戏
2. 其他游戏
3. 补充注释，对比ddpg和SAC，提取关键点
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
import torch.distributions as dist
import torch.nn.functional as F


ENV_ID = "Pendulum-v1"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000 # 重放缓冲区长度，这么长是为了提高稳定性
REPLAY_INITIAL = 10000 # 重放缓冲区初始化大小

TEST_ITERS = 1000

eval_noise_scale = 0.5
explore_noise_scale = 1.0
reward_scale = 1.
policy_target_update_interval = 3

class SAC():
    def __init__(self, obs_space, action_space, hidden_dim, action_range, device="cpu"):
        self.device = device
        self.act_net = model.SACActor(obs_space.shape[0], action_space.shape[0], hidden_dim, action_range).to(device)
        self.crt1_net = model.DDPGCritic(obs_space.shape[0], action_space.shape[0]).to(device)
        self.crt2_net = model.DDPGCritic(obs_space.shape[0], action_space.shape[0]).to(device)
        self.target_crt1_net = ptan.agent.TargetNet(self.crt1_net)
        self.target_crt2_net = ptan.agent.TargetNet(self.crt2_net)

        self.act_opt = optim.Adam(self.act_net.parameters(), lr=LEARNING_RATE)
        self.crt1_opt = optim.Adam(self.crt1_net.parameters(), lr=LEARNING_RATE)
        self.crt2_opt = optim.Adam(self.crt2_net.parameters(), lr=LEARNING_RATE)

        self.action_range = action_range
        self.device = device
        self.update_cnt = 0


    def update(self):
        self.update_cnt += 1


class SACAgent(ptan.agent.BaseAgent):
    def __init__(self, net, explore_noise_scale, action_range=1., step=0, device="cpu"):
        '''

        :param net:
        :param explore_noise_scale:
        :param action_range: todo 这里的动作范围是否需要考虑到不同的游戏
        '''
        assert isinstance(net, model.SACActor)
        self.net = net
        self.explore_noise_scale = explore_noise_scale
        self.action_range = action_range
        self.step = step
        self.device = device
        self.rng = np.random.default_rng(int(time.time()))

    def __call__(self, states, agent_states):
        # todo 这边用原来的OU方法是否也可以？
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        self.step += 1
        # todo 500是否可以不用
        if self.step < 500:
            actions = self.rng.uniform(-1, 1, actions.shape)
            actions = self.action_range * actions
        else:
            noise = self.rng.normal(0, 1, actions.shape) * self.explore_noise_scale
            actions = actions + noise

        return actions.clip(-1 * self.action_range, 1 * self.action_range), agent_states



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

    save_path = os.path.join("saves", "sac-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID, g=9.81)
    test_env = gym.make(ENV_ID, g=9.81)

    action_range = float(env.action_space.high[0])

    # 构建动作网络和评价网络
    sac = SAC(env.observation_space, env.action_space, action_range=action_range, device=device)
    print(sac.act_net)
    print(sac.crt1_net)
    print(sac.crt2_net)

    agent = SACAgent(sac.act_net, action_range=action_range, explore_noise_scale=explore_noise_scale, device=device)
    if (os.path.exists(os.path.join(save_path, "checkpoint_0.dat"))):
        checkpoint = torch.load(os.path.join(save_path, "checkpoint_0.dat"))
        sac.act_net.load_state_dict(checkpoint["act_net"])
        sac.crt1_net.load_state_dict(checkpoint["crt1_net"])
        sac.crt2_net.load_state_dict(checkpoint["crt2_net"])
        sac.target_act_net.target_model.load_state_dict(checkpoint["target_act_net"])
        sac.target_crt1_net.target_model.load_state_dict(checkpoint["target_crt1_net"])
        sac.target_crt2_net.target_model.load_state_dict(checkpoint["target_crt2_net"])
        sac.update_cnt = checkpoint["update_cnt"]
        agent.step = checkpoint["step"]
        print("加载模型成功")

    writer = SummaryWriter(comment="-sac_" + args.name)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)

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

                sac.update()
                # 从缓冲区里面采样数据
                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)

                # 归一化奖励
                rewards_v = reward_scale * (rewards_v - rewards_v.mean()) / (rewards_v.std() + 1e-6)
                # 使用目标动作预测网路，根据下一个状态预测执行的动作
                last_act_v = sac.target_act_net.target_model.evaluate(last_states_v, eval_noise_scale)
                # 使用目标评测网络，根据下一个状态和下一个状态将要执行的动作得到下一个状态的评价Q值
                q_last_v = torch.min(sac.target_crt1_net.target_model(last_states_v, last_act_v), sac.target_crt2_net.target_model(last_states_v, last_act_v))
                # 如果是结束状态则将奖励置为0
                q_last_v[dones_mask.bool()] = 0.0
                # 计算Q值 bellman公式
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                # 计算预测的当前Q值和Bellman计算的到的Q值之间的差异
                # 并更新梯度 这里的方式就和之前的Q值单元的一致
                # train critic
                sac.crt1_opt.zero_grad()
                # 根据状态和动作，得到评价，这里是根据实际游戏的状态和动作获取评价
                q_v = sac.crt1_net(states_v, actions_v)
                critic_1_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_1_loss_v.backward()
                sac.crt1_opt.step()
                tb_tracker.track("loss_critic_1", critic_1_loss_v, frame_idx)

                sac.crt2_opt.zero_grad()
                q_v = sac.crt2_net(states_v, actions_v)
                critic_2_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_2_loss_v.backward()
                sac.crt2_opt.step()
                tb_tracker.track("loss_critic_2", critic_2_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # 预测动作
                if sac.update_cnt % policy_target_update_interval == 0:
                    # train actor
                    sac.act_opt.zero_grad()
                    new_action, new_log_prob, _, _ = sac.act_net.evaluate(states_v, device)
                    target_q_min = torch.min(sac.target_crt1_net.target_model(states_v, new_action),
                                         sac.target_crt2_net.target_model(states_v, new_action)) - alpha * new_log_prob
                    # 根据状态和预测的动作计算Q值的负值
                    # 这里是根据网络预测的动作和实际的状态获取评价
                    # 在评价前取负号，就是简单粗暴的取最小值，从而达到最大值Q值的目的
                    # 由于这里评价网络是固定是，所以最大化Q值，只有更新预测的动作，使得
                    # 评价Q值达到最大值的目的
                    actor_loss_v = -sac.crt1_net(states_v, cur_actions_v)
                    actor_loss_v = actor_loss_v.mean()
                    actor_loss_v.backward()
                    td3.act_opt.step()
                    tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                    # 将训练网路同步到目标网络上，但是这里是每次都同步，与之前每隔n步同步一次不同
                    # 这里之所以这样做，是根据测试可知，每次都同步，并使用较小的权重进行同步
                    # 缓存的同步效果更好，并且能够保持平滑的更新
                    td3.target_act_net.alpha_sync(alpha=1 - 1e-3)
                    td3.target_crt1_net.alpha_sync(alpha=1 - 1e-3)
                    td3.target_crt2_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    # 测试并保存最好测试结果的庶数据
                    ts = time.time()
                    rewards, steps = test_net(td3.act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    checkpoint = {
                        "act_net": td3.act_net.state_dict(),
                        "crt1_net": td3.crt1_net.state_dict(),
                        "crt2_net": td3.crt2_net.state_dict(),
                        "update_cnt": td3.update_cnt,
                        "frame_idx": frame_idx,
                        "best_reward": best_reward,
                        "target_crt1_net": td3.target_crt1_net.target_model.state_dict(),
                        "target_crt2_net": td3.target_crt2_net.target_model.state_dict(),
                        "target_act_net": td3.target_act_net.target_model.state_dict(),
                        "step": agent.step
                    }
                    torch.save(checkpoint, os.path.join(save_path, "checkpoint_%d.dat" % (frame_idx % 10)))
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)

                            torch.save(checkpoint, fname)
                        best_reward = rewards
    pass
