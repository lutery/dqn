#!/usr/bin/env python3
'''
验证通过，而且下降速度也很快，看来PPO算法对于离散动作空间的游戏也是非常有效的
训练到中途，发现最终会计算出nan值，根据跟踪是因为ratio_v差值太大，导致最终计算出来的值过大，最后导致了在梯度计算时造成了梯度更新过大，导致nan值
虽然有torch.min，但是如果是一个非常大的负值（-xe-30这种级别的值），也是会造成梯度爆炸的，因为torch.min会选择一个大负值，大负值也会造成梯度爆炸
根据以上现象以及结合gpt得出可能时在梯度更新时，由于新旧策略之间差异过大，导致ratio_v过大，这个时候可以中止训练，重新使用新的网络进行探索训练
gpt描述：
长时间未更新旧策略
如果每次 PPO 更新的迭代次数太多，旧策略和新策略之间的差异会累积，从而导致非常大的 ratio_v。
解决方法：
你可以减少每次更新的轨迹长度（TRAJECTORY_SIZE），以更频繁地更新旧策略。

所以可以采用以下方式解决nan值：
1. 如果ratio_v值非常大，可以中止训练，重新使用新的网络进行探索训练（已验证）
2. torch.min使用绝对值选择小值，防止选择了爆炸的负值造成梯度爆炸（未验证）
3. 减少更新的轨迹长度（TRAJECTORY_SIZE），以更频繁地更新旧策略（未验证）

另外关于学习率的参数，发现学习率太小比如1e-5，无法收敛，可能是更新非常慢或者进入了局部最优解，发现将学习率设置如下可以收敛，可能时使用
较大的学习率可以使得网路跳出假的谷底，找到真正的谷底
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
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


GAMMA = 0.99
GAE_LAMBDA = 0.95 # 优势估计器的lambda因子，0.95是一个比较好的值

TRAJECTORY_SIZE = 2049 # todo 作用 看代码好像是采样的轨迹长度（轨迹，也就是连续采样缓存长度，游戏是连续的）
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3

PPO_EPS = 0.2
PPO_EPOCHES = 10 # todo 执行ppo的迭代次数 作用
PPO_BATCH_SIZE = 64 # 每次进行轨迹样本计算的batch长度

TEST_ITERS = 100000 # 采样迭代多少次，进行一次游戏测试

CLIP_GRAD = 0.5


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

    save_path = os.path.join("saves", "ppo-discrete-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5)
    test_env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5)

    # 创建动作预测网络
    net_act = model.ModelActorDis(env.observation_space.shape[0], env.action_space.n).to(device)
    # 创建状态、动作评价网络
    net_crt = model.ModelCriticDis(env.observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)
    if (os.path.exists(os.path.join(save_path, "act-net.data"))):
        net_act.load_state_dict(torch.load(os.path.join(save_path, "act-net.data")))
        print("加载act模型成功")

    if (os.path.exists(os.path.join(save_path, "crt-net.data"))):
        net_crt.load_state_dict(torch.load(os.path.join(save_path, "crt-net.data")))
        print("加载crt模型成功")

    writer = SummaryWriter(comment="-ppo-discrete_" + args.name)
    agent = ptan.agent.PolicyAgent(net_act, device=device, preprocessor=ppo_states_preprocessor)
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
                    writer.add_scalar("logprob_pi_v min", logprob_pi_v.min().item(), grad_index)
                    writer.add_scalar("batch_old_logprob_v mean", batch_old_logprob_v.mean().item(), grad_index)
                    writer.add_scalar("batch_old_logprob_v max", batch_old_logprob_v.max().item(), grad_index)
                    writer.add_scalar("batch_old_logprob_v min", batch_old_logprob_v.min().item(), grad_index)
                    writer.add_scalar("ratio_v_pre mean", (logprob_pi_v - batch_old_logprob_v).mean().item(), grad_index)
                    writer.add_scalar("ratio_v_pre max", (logprob_pi_v - batch_old_logprob_v).max().item(), grad_index)
                    writer.add_scalar("ratio_v_pre min", (logprob_pi_v - batch_old_logprob_v).min().item(), grad_index)
                    
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

                    writer.add_scalar("torch.clamp mean", torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS).mean().item(), grad_index)
                    writer.add_scalar("torch.clamp min", torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS).min().item(), grad_index)
                    writer.add_scalar("torch.clamp max", torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS).max().item(), grad_index)

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

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)

            torch.save(net_act.state_dict(), os.path.join(save_path, "act-net.data"))
            torch.save(net_crt.state_dict(), os.path.join(save_path, "crt-net.data"))



