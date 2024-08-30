#!/usr/bin/env python3
'''
未验证
1. 了解离散动作下的PPO

出现错误：
NaN or Inf found in input tensor.
Traceback (most recent call last):
  File "/home/lutery/projects/python/my_-nqd/learning/super_mario/train_ppo.py", line 183, in <module>
    for step_idx, exp in enumerate(exp_source):
  File "/home/lutery/anaconda3/envs/pytorch-gym/lib/python3.10/site-packages/ptan-0.8-py3.10.egg/ptan/experience.py", line 127, in __iter__
  File "/home/lutery/anaconda3/envs/pytorch-gym/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/lutery/anaconda3/envs/pytorch-gym/lib/python3.10/site-packages/ptan-0.8-py3.10.egg/ptan/agent.py", line 163, in __call__
  File "/home/lutery/anaconda3/envs/pytorch-gym/lib/python3.10/site-packages/ptan-0.8-py3.10.egg/ptan/actions.py", line 64, in __call__
  File "numpy/random/mtrand.pyx", line 971, in numpy.random.mtrand.RandomState.choice
ValueError: probabilities contain NaN
'''
import os
import math
import ptan
import time
import gymnasium as gym
import argparse
from tensorboardX import SummaryWriter

from lib import model

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from collections import deque

# super mario
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import StepAPICompatibility, TimeLimit
#


GAMMA = 0.99
GAE_LAMBDA = 0.95 # 优势估计器的lambda因子，0.95是一个比较好的值

TRAJECTORY_SIZE = 2049 # todo 作用 看代码好像是采样的轨迹长度（轨迹，也就是连续采样缓存长度，游戏是连续的）
LEARNING_RATE_ACTOR = 5e-5
LEARNING_RATE_CRITIC = 5e-5

PPO_EPS = 0.2
PPO_EPOCHES = 10 # todo 执行ppo的迭代次数 作用
PPO_BATCH_SIZE = 64 # 每次进行轨迹样本计算的batch长度

TEST_ITERS = 100000 # 采样迭代多少次，进行一次游戏测试

CLIP_GRAD = 0.5

class StackFrameWrapper(gym.Wrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.env = env
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)

        low = np.repeat(self.observation_space.low, n_frames, axis=2)
        high = np.repeat(self.observation_space.high, n_frames, axis=2)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

        self.obs = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0), reward, terminated, truncated, info



class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeObservation, self).__init__(env)

    def observation(self, observation):
        # 将观察从 (H, W, C) 转换为 (C, H, W)
        return observation.transpose(2, 0, 1)

def gymnasium_reset(self, **kwargs):
    return self.env.reset(), {}

def make_super_mario_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    steps = env._max_episode_steps  # get the original max_episode_steps count
    env = JoypadSpace(env.env, SIMPLE_MOVEMENT)  # set the joypad wrapper
    # overwrite the old reset to accept `seeds` and `options` args
    env.reset = gymnasium_reset.__get__(env, JoypadSpace)

    # set TimeLimit back
    env = TimeLimit(StepAPICompatibility(env, output_truncation_bool=True), max_episode_steps=steps)

    return StackFrameWrapper(TransposeObservation(env))


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().argmax().item()
            obs, reward, done, trunc, _ = env.step(action)
            # env.render()
            done = done or trunc
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    通过轨迹计算优势和1步参考值
    :param trajectory: trajectory list 收集的连续采样记录
    :param net_crt: critic network 评价网络
    :param states_v: states tensor 状态张量
    :return: tuple with advantage numpy array and reference values
    """
    with torch.no_grad():
        values_v = net_crt(states_v) # 得到预测的Q值
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    # 广义优势估计量:优势的平滑版
    last_gae = 0.0 # 作用 存储动作优势值，这里的优势值与之前不同之处在于
    # 这里会将未来的优势获取的情况考虑在内
    result_adv = [] # 存储动作的优势值
    result_ref = [] # 存储实际的Q值
    # zip(reversed(values[:-1]), reversed(values[1:] 是将中的数据按照
    # ((-2, -1), (-3, -2), (-4, -3)......)的顺序进行组合，对应了val和next_val
    # 并且每一个组合都和trajectory中的经验进行了逆序进行组合也就是(（（-2,-1）,-2）,((-3,-2)-3)......)
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            # 如果游戏的状态是结束的
            delta = exp.reward - val # 计算实际的Q值和预测的Q值的差值
            last_gae = delta # 由于没有后续的动作，那么不考虑之前的优势了
        else:
            # 如果游戏的状态不是结束的
            # 根据bellman公式计算实际Q值后，计算实际Q值和预测Q值的差值
            delta = exp.reward + GAMMA * next_val - val
            # 这个公式是计算优势的公式，这个公式的作用是将优势进行平滑
            # 因为使用的是平滑版本的优势估计，所以这里的每一步的优势值是会包含
            # 后续步骤（因为是逆序遍历）的优势迭代之与折扣因子GAMMA * GAE_LAMBDA
            # 的和
            # 这步体现了累计的优势，即当前获得的优势和之后都有关系
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    # 这里的逆序的作用
    # adv_v保存的好像是动作优势，也就是实际执行的和预测的Q值的差值
    # ref_v保存的好像是实际Q值
    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v

def ppo_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    这个预处理器的方法就是将list转换为矩阵的形式
    如果state是一维的，那么就将其转换为[1, D]的形式
    如果state是多维的，那么就将其转换为[N, E, D]的形式
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states.copy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ppo-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = make_super_mario_env()
    test_env = make_super_mario_env()

    # 创建动作预测网络
    net_act = model.ModelActor(env.observation_space.shape, env.action_space.n).to(device)
    # 创建状态、动作评价网络
    net_crt = model.ModelCritic(env.observation_space.shape).to(device)
    print(net_act)
    print(net_crt)
    if (os.path.exists(os.path.join(save_path, "act-net.data"))):
        net_act.load_state_dict(torch.load(os.path.join(save_path, "act-net.data")))
        print("加载act模型成功")

    if (os.path.exists(os.path.join(save_path, "crt-net.data"))):
        net_crt.load_state_dict(torch.load(os.path.join(save_path, "crt-net.data")))
        print("加载crt模型成功")

    writer = SummaryWriter(comment="-ppo_" + args.name)
    agent = ptan.agent.PolicyAgent(net_act, apply_softmax=True, device=device, preprocessor=ppo_states_preprocessor)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    trajectory = [] # 注意，缓冲区更名为轨迹
    best_reward = None
    grad_index = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

            if step_idx > 0 and step_idx % TEST_ITERS == 0:
                ts = time.time()
                rewards, steps = test_net(net_act, test_env, count=1, device=device)
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

            # 这里之所以会需要使用
            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(np.array(traj_states)).to(device)
            traj_actions_v = torch.FloatTensor(np.array(traj_actions)).to(device)
            # 计算优势值和实际Q值
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
            # 根据状态预测动作
            with torch.no_grad():
                mu_v = net_act(traj_states_v)
            # 计算上一轮训练的评价网络、动作网络动作的概率
            old_logprob_v = torch.log(mu_v.gather(1, torch.tensor(traj_actions, dtype=torch.int64).to(device).unsqueeze(-1))).detach()

            # normalize advantages 归一化计算得到的Q值 作用是提高训练的稳定性
            traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            # 这里的作用，为什么截取去除最后一个 P316
            # 可能是因为要和traj_adv_v和traj_ref_v一一对应，因为里面在计算时，是
            # 去掉最后一个的
            trajectory = trajectory[:-1]
            # 这里获取的是从[0到-1)范围的数据
            old_logprob_v = old_logprob_v[:-1].detach()

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            # 开始进行PPO的迭代（近端策略优化）
            for epoch in range(PPO_EPOCHES):
                for batch_ofs in range(0, len(trajectory), PPO_BATCH_SIZE):
                    states_v = traj_states_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                    actions_v = traj_actions_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                    batch_adv_v = traj_adv_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE].unsqueeze(-1)
                    batch_ref_v = traj_ref_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                    batch_old_logprob_v = old_logprob_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]

                    # critic training
                    # 这边就是在计算预测Q值和实际Q值之间的差异损失
                    opt_crt.zero_grad()
                    value_v = net_crt(states_v)
                    if torch.isnan(value_v).any() or torch.isinf(value_v).any():
                        print(f"Warning: NaN or inf detected in value_v at step {step_idx}")
                        torch.save(net_crt.state_dict(), os.path.join(save_path, f"nan_inf_detected_crt_net_{step_idx}.pth"))
                        raise ValueError("NaN or inf detected in value_v") 
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                    loss_value_v.backward()
                    nn_utils.clip_grad_norm_(net_crt.parameters(), CLIP_GRAD)
                    opt_crt.step()

                    # actor training
                    opt_act.zero_grad()
                    mu_v = net_act(states_v)
                    if torch.isnan(mu_v).any() or torch.isinf(mu_v).any():
                        print(f"Warning: NaN or inf detected in mu_v at step {step_idx}")
                        torch.save(net_act.state_dict(), os.path.join(save_path, f"nan_inf_detected_act_net_{step_idx}.pth"))
                        raise ValueError("NaN or inf detected in mu_v") 
        

                    # 计算预测执行动作的高斯概率
                    indices = actions_v.long().to(device).unsqueeze(-1)
                    gathered_values = mu_v.gather(1, indices)
                    min_value = gathered_values.min().item()
                    max_value = gathered_values.max().item()
                    zero_count = (gathered_values == 0).sum().item()
                    near_zero_count = (gathered_values.abs() < 1e-7).sum().item()

                    writer.add_scalar("gathered_min", min_value, grad_index)
                    writer.add_scalar("gathered_max", max_value, grad_index)
                    writer.add_scalar("gathered_zero_count", zero_count, grad_index)
                    writer.add_scalar("gathered_near_zero_count", near_zero_count, grad_index)
                    
                    logprob_pi_v = torch.log(mu_v.gather(1, indices) + 1e-7)
                    writer.add_scalar("logprob_pi_v mean", logprob_pi_v.mean().item(), grad_index)
                    writer.add_scalar("logprob_pi_v max", logprob_pi_v.max().item(), grad_index)
                    # 计算实时更新的动作预测网络和之前的动作预测网络之间的预测差异比例
                    # 公式P317
                    # 这里使用了exp的除法变换公式（log），所以书中的P317中的在这里是减号
                    if torch.isnan(logprob_pi_v).any() or torch.isinf(logprob_pi_v).any():
                        print(f"Warning: NaN or inf detected in logprob_pi_v at step {step_idx}")
                        raise ValueError("NaN or inf detected in logprob_pi_v") 
                    
                    if torch.isnan(batch_old_logprob_v).any() or torch.isinf(batch_old_logprob_v).any():
                        print(f"Warning: NaN or inf detected in batch_old_logprob_v at step {step_idx}")
                        raise ValueError("NaN or inf detected in batch_old_logprob_v") 
                    ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                    writer.add_scalar("ratio_v mean", ratio_v.mean().item(), grad_index)
                    # ratio_v的作用
                    # 用于计算新旧策略之间的比例，这个比例用于计算新旧策略之间的差异
                    # 根据这个差异调整网络的参数，使其能够往更好的方向调整
                    # batch_adv_v对应书中P317中的At
                    # ratio_v对应书中的rt(theta)
                    # torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)对应书中的clip
                    writer.add_scalar("batch_adv_v mean", batch_adv_v.mean().item(), grad_index)
                    writer.add_scalar("batch_adv_v min", batch_adv_v.min().item(), grad_index)
                    writer.add_scalar("batch_adv_v mean", batch_adv_v.max().item(), grad_index)
                    surr_obj_v = batch_adv_v * ratio_v
                    writer.add_scalar("surr_obj_v mean", surr_obj_v.mean().item(), grad_index)
                    writer.add_scalar("surr_obj_v min", surr_obj_v.min().item(), grad_index)
                    writer.add_scalar("surr_obj_v mean", surr_obj_v.max().item(), grad_index)

                    clipped_surr_v = batch_adv_v * torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                    writer.add_scalar("clipped_surr_v mean", clipped_surr_v.mean().item(), grad_index)
                    writer.add_scalar("clipped_surr_v min", clipped_surr_v.min().item(), grad_index)
                    writer.add_scalar("clipped_surr_v mean", clipped_surr_v.max().item(), grad_index)

                    writer.add_scalar("torch.clamp mean", torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS).mean().item(), grad_index)
                    writer.add_scalar("torch.clamp min", torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS).min().item(), grad_index)
                    writer.add_scalar("torch.clamp mean", torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS).max().item(), grad_index)

                    loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                    loss_policy_v.backward()
                    nn_utils.clip_grad_norm_(net_act.parameters(), CLIP_GRAD)

                    grad_max = 0.0
                    grad_means = 0.0
                    grad_count = 0
                    for p in net_act.parameters():
                        grad_max = max(grad_max, p.grad.abs().max().item())
                        grad_means += (p.grad ** 2).mean().sqrt().item()
                        grad_count += 1
                    writer.add_scalar("grad_l2", grad_means / grad_count, grad_index)
                    writer.add_scalar("grad_max", grad_max, grad_index)

                    opt_act.step()

                    weights_max = 0.0
                    weights_means = 0.0
                    weights_count = 0
                    for p in net_act.parameters():
                        weights_max = max(weights_max, p.data.abs().max().item())
                        weights_means += (p.data ** 2).mean().sqrt().item()
                        weights_count += 1
                    writer.add_scalar("weights_l2", weights_means / weights_count, grad_index)
                    writer.add_scalar("weights_max", weights_max, grad_index)

                    # 记录总损失，用于计算平均损失变化
                    sum_loss_value += loss_value_v.item()
                    sum_loss_policy += loss_policy_v.item()
                    count_steps += 1
                    grad_index += 1

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)

            torch.save(net_act.state_dict(), os.path.join(save_path, "act-net.data"))
            torch.save(net_crt.state_dict(), os.path.join(save_path, "crt-net.data"))



