#!/usr/bin/env python3
'''
todo
2. 如果未完成进一步考虑如何提高精度以提高平均奖励值
'''
import gymnasium as gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import os

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from typing import Any
from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1

SAVE_ITERS = 100

class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeObservation, self).__init__(env)

    def observation(self, observation):
        # 将观察从 (H, W, C) 转换为 (C, H, W)
        return observation.transpose(2, 0, 1)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        # 以下可知，一些游戏存在FIRE的动作，并且存在FIRE动作的游戏其游戏动作执行有三个以上
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # 这里之所以尝试重置后尝试各种动作，是因为不知道哪个是FIRE，继续游戏，所以一个一个尝试
        # 如果不小心游戏结束了，则继续重置
        # 假设游戏继续游戏的按钮在前3
        self.env.reset(seed=seed, options=options)
        obs, _, done, _, info = self.env.step(1)
        if done:
            self.env.reset(seed=seed, options=options)
        obs, _, done, _, info = self.env.step(2)
        if done:
            self.env.reset(seed=seed, options=options)
        return obs, info

class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        obs_action = (input_shape[2], input_shape[0], input_shape[1])

        self.conv = nn.Sequential(
            nn.Conv2d(obs_action[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_action)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = [] # 非结束的游戏数据索引，该索引记录对应batch，states，actions，rewards
    last_states = [] # 记录采样中的执行动作后的状态，仅记录游戏非结束状态下的索引
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", default="breakout", required=False, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "a2c-" + args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    envs = [FireResetEnv(TransposeObservation(gym.make("ALE/Breakout-v5", obs_type='rgb', frameskip=4, repeat_action_probability=0.0))) for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-a2c_" + args.name)

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)
    if (os.path.exists(os.path.join(save_path, "a2c.pth"))):
        net.load_state_dict(torch.load(os.path.join(save_path, "a2c.pth")))
        print("加载模型成功")

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []
    best_reward = 0
    train_count = 0

    with common.RewardTracker(writer, stop_reward=700) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                    if new_rewards[0] > best_reward:
                        best_reward = new_rewards[0]
                        torch.save(net.state_dict(), os.path.join(save_path, "a2c-best-{}.pth".format(best_reward)))

                if len(batch) < BATCH_SIZE:
                    continue


                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                loss_v += loss_policy_v
                train_count += 1

                if train_count % SAVE_ITERS == 0:
                    torch.save(net.state_dict(), os.path.join(save_path, "a2c.pth"))

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)