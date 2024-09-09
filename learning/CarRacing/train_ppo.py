#!/usr/bin/env python3
'''
未验证
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


ENV_ID = "CarRacing-v2"
GAMMA = 0.99
GAE_LAMBDA = 0.95 # 优势估计器的lambda因子，0.95是一个比较好的值

TRAJECTORY_SIZE = 2049 # todo 作用 看代码好像是采样的轨迹长度（轨迹，也就是连续采样缓存长度，游戏是连续的）
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3

PPO_EPS = 0.2
PPO_EPOCHES = 10 # todo 执行ppo的迭代次数 作用
PPO_BATCH_SIZE = 64 # 每次进行轨迹样本计算的batch长度

TEST_ITERS = 100000 # 采样迭代多少次，进行一次游戏测试

class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeObservation, self).__init__(env)

    def observation(self, observation):
        # 将观察从 (H, W, C) 转换为 (C, H, W)
        return observation.transpose(2, 0, 1)

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
    这里依旧采用的是高斯概率分布计算预测的动作概率
    """
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    通过轨迹计算优势和1步参考值
    :param trajectory: trajectory list 收集的连续采样记录
    :param net_crt: critic network 评价网络
    :param states_v: states tensor 状态张量
    :return: tuple with advantage numpy array and reference values
    """
    # values_v = net_crt(states_v) # 得到预测的Q值
    values_v = process_in_batches(states_v, net_crt, PPO_BATCH_SIZE)
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


def process_in_batches(data, model, batch_size):
    """
    分批处理数据并通过模型计算结果。

    参数:
    - data: 要处理的数据，应为 torch.Tensor 类型。
    - model: 用于处理数据的神经网络模型。
    - batch_size: 每个批次的数据大小。

    返回:
    - 处理后的数据，所有批次的结果合并后的 torch.Tensor。
    """
    total_size = data.size(0)
    results = []  # 存储每个批次的输出结果

    # 遍历并处理每个批次
    with torch.no_grad():  # 关闭梯度计算
        for i in range(0, total_size, batch_size):
            # print("Processing batch {}/{}".format(i // batch_size + 1, total_size // batch_size))
            # 获取当前批次的数据
            batch = data[i:min(i + batch_size, total_size)]
            
            # 传入模型进行预测
            batch_output = model(batch)
            
            # 收集输出结果
            results.append(batch_output)

    # 将所有批次结果合并成一个 Tensor
    combined_results = torch.cat(results, dim=0)
    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", default="ppo_carracing", required=False, help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default=" + ENV_ID)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ppo-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = TransposeObservation(gym.make(args.env, domain_randomize=True, continuous=True))
    test_env = TransposeObservation(gym.make(args.env, continuous=True))

    # 创建动作预测网络
    net_act = model.ModelActor(env.observation_space.shape, env.action_space.shape[0]).to(device)
    # 创建状态、动作评价网络
    net_crt = model.ModelCritic(env.observation_space.shape).to(device)
    if (os.path.exists(os.path.join(save_path, "act-net.dat"))):
        net_act.load_state_dict(torch.load(os.path.join(save_path, "act-net.dat")))
        net_crt.load_state_dict(torch.load(os.path.join(save_path, "crt-net.dat")))
        print("加载模型成功")

    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment="-ppo_" + args.name)
    agent = model.AgentPPO(net_act, device=device)
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

            # 这里之所以会需要使用
            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(np.array(traj_states)).to(device)
            traj_actions_v = torch.FloatTensor(np.array(traj_actions)).to(device)
            # 计算优势值和实际Q值
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
            # 根据状态预测动作
            # mu_v = net_act(traj_states_v)
            mu_v = process_in_batches(traj_states_v, net_act, PPO_BATCH_SIZE)
            # 计算上一轮训练的评价网络、动作网络动作的概率
            old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

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
            old_ratio_v_mean = 0
            is_interrupt = False

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
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                    loss_value_v.backward()
                    opt_crt.step()

                    # actor training
                    opt_act.zero_grad()
                    mu_v = net_act(states_v)
                    if torch.isnan(mu_v).any() or torch.isinf(mu_v).any():
                        print(f"Warning: NaN or inf detected in mu_v at step {step_idx}")
                        torch.save(net_act.state_dict(), os.path.join(save_path, f"nan_inf_detected_act_net_{step_idx}.pth"))
                        raise ValueError("NaN or inf detected in mu_v")
                    # 计算预测执行动作的高斯概率
                    logprob_pi_v = calc_logprob(mu_v, net_act.logstd, actions_v)
                    min_value = logprob_pi_v.min().item()
                    max_value = logprob_pi_v.max().item()
                    mean_value = logprob_pi_v.mean().item()

                    writer.add_scalar("logprob_pi_v min", min_value, grad_index)
                    writer.add_scalar("logprob_pi_v max", max_value, grad_index)
                    writer.add_scalar("logprob_pi_v mean", mean_value, grad_index)

                    min_value = batch_old_logprob_v.min().item()
                    max_value = batch_old_logprob_v.max().item()
                    mean_value = batch_old_logprob_v.mean().item()

                    writer.add_scalar("batch_old_logprob_v min", min_value, grad_index)
                    writer.add_scalar("batch_old_logprob_v max", max_value, grad_index)
                    writer.add_scalar("batch_old_logprob_v mean", mean_value, grad_index)

                    pre_ratio_v = logprob_pi_v - batch_old_logprob_v
                    min_value = pre_ratio_v.min().item()
                    max_value = pre_ratio_v.max().item()
                    mean_value = pre_ratio_v.mean().item()

                    writer.add_scalar("pre_ratio_v min", min_value, grad_index)
                    writer.add_scalar("pre_ratio_v max", max_value, grad_index)
                    writer.add_scalar("pre_ratio_v mean", mean_value, grad_index)

                    ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                    if abs(ratio_v.mean().item() - old_ratio_v_mean) > 10:
                        opt_act.zero_grad()
                        is_interrupt = True
                        break
                    old_ratio_v_mean = ratio_v.mean().item()
                    writer.add_scalar("ratio_v mean", ratio_v.mean().item(), grad_index)
                    writer.add_scalar("ratio_v max", ratio_v.max().item(), grad_index)
                    writer.add_scalar("ratio_v min", ratio_v.min().item(), grad_index)
                    writer.add_scalar("batch_adv_v mean", batch_adv_v.mean().item(), grad_index)
                    writer.add_scalar("batch_adv_v min", batch_adv_v.min().item(), grad_index)
                    writer.add_scalar("batch_adv_v max", batch_adv_v.max().item(), grad_index)

                    surr_obj_v = batch_adv_v * ratio_v
                    writer.add_scalar("surr_obj_v mean", surr_obj_v.mean().item(), grad_index)
                    writer.add_scalar("surr_obj_v min", surr_obj_v.min().item(), grad_index)
                    writer.add_scalar("surr_obj_v max", surr_obj_v.max().item(), grad_index)

                    clipped_surr_v = batch_adv_v * torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                    writer.add_scalar("clipped_surr_v mean", clipped_surr_v.mean().item(), grad_index)
                    writer.add_scalar("clipped_surr_v min", clipped_surr_v.min().item(), grad_index)
                    writer.add_scalar("clipped_surr_v max", clipped_surr_v.max().item(), grad_index)

                    loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                    loss_policy_v.backward()
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
                if is_interrupt:
                    is_interrupt = False
                    break

            torch.save(net_act.state_dict(), os.path.join(save_path, "act-net.dat"))
            torch.save(net_crt.state_dict(), os.path.join(save_path, "crt-net.dat"))
            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)

