#!/usr/bin/env python3
'''
未验证
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


ENV_ID = "Ant-v4"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 5 # N步展开的步数大小

TEST_ITERS = 1000

# Q值概率分布的参数
Vmax = 10 # 概率分布最大值
Vmin = -10 # 概率分布最小值
N_ATOMS = 51 # 概率分布的区间范围
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, trunc, _ = env.step(action)
            rewards += reward
            steps += 1
            if done or trunc:
                break
    return rewards / count, steps / count


def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device="cpu"):
    '''
    这里的分布运算，实际上就是将实际的得到回报计算是具体分布到分布的哪个点上
    '''
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(bool)
    batch_size = len(rewards)
    # 初始化投影分布，坐标中-10~-1,表示不好的分布，0~10表示好的分布，对应奖励中的负向奖励和正向奖励
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        # l n这里计算的是分布中第几个坐标点，整个横向坐标点都是奖励回报的分布
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        # 这里计算的是上界和下界一样的batch
        eq_mask = u == l
        # l[eq_mask]表示从下界点中，选取上下界相等的那些坐标点，以下相加就是累计哪个坐标点的回报概率是多少，因为next_distr[eq_mask, atom]就是是经过sotmax计算得到的结果，所以整个预测的Q值回报分布的结果概率相加为1
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        # 将游戏结束的分布重置为0
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64) # l和n包含所有结束状态的计算结果
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask # eq_dones[dones_mask]仅修改结束标识里面为True的数据，将游戏结束的游戏对应位置标志上下值是否相等
        if eq_dones.any():
            # 上一步中，eq_mask如果有存在一个为True，那么eq_dnones中就会进入到这里处理
            proj_distr[eq_dones, l[eq_mask]] = 1.0 # l[eq_mask]从l中找到l==n的索引位置，然后设置proj_distr在计算得到的概率分布点设置为1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any(): # 下面就是将l!=n的位置，将概率按照距离分配至最近的两个原点
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "d4pg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    # 构建动作网络和评价网络
    # 动作网络和ddpg一样
    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # 评价网络是不同的
    crt_net = model.D4PGCritic(env.observation_space.shape[0], env.action_space.shape[0], N_ATOMS, Vmin, Vmax).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-d4pg_" + args.name)
    # 和p305不一样
    # 因为在分布式的版本中，可以使用基于OU过程的DDPG，也可以使用简单基于随机噪音的D4PG代理器
    agent = model.AgentDDPG(act_net, device=device, ou_sigma=0.3, ou_teta=0.05)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    # 经验重放缓冲区
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    frame_idx = 0

    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        # 增加加载模型的代码
        checkpoints = sorted(os.listdir(save_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        checkpoint = torch.load(os.path.join(save_path, checkpoints[-1]), map_location=device)
        frame_idx = checkpoint['frame_idx']
        act_net.load_state_dict(checkpoint['act_net'])
        crt_net.load_state_dict(checkpoint['crt_net'])
        tgt_act_net.target_model.load_state_dict(checkpoint['tgt_act_net'])
        tgt_crt_net.target_model.load_state_dict(checkpoint['tgt_crt_net'])
        act_opt.load_state_dict(checkpoint['act_opt'])
        crt_opt.load_state_dict(checkpoint['crt_opt'])
        print("加载模型成功")

    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                # 解压采样
                states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                # 根据实际的动作和状态获取评价分布
                crt_distr_v = crt_net(states_v, actions_v)
                # tgt_act_net 预测下一个状态的执行的动作
                last_act_v = tgt_act_net.target_model(last_states_v)
                # 根据目标网路预测的下一个状态执行的动作和状态的目标评测网络评价分布
                # 使用softmax后，Q值分布就变成了概率分布，所有分布之和为1
                # 也就是说，所有基于分布的Q值网络，实际上就是将Q值分布到一个分布上，表达的是在某个状态下，执行某个动作，获得的每种回报的概率分布，每个方格就是一种回报大小的可能性，正向的方格是正向的回报概率，负向的方格是负向的回报概率，回报的所有可能性大小都被归纳到Vmin到Vmax之间
                last_distr_v = F.softmax(tgt_crt_net.target_model(last_states_v, last_act_v), dim=1)
                # 其实就是和普通的单值Qz值一样，将目标网络预测出来的奖励和实际的回报结合起来，如果奖励是正向的，那么预测概率分布都是集中上正向坐标上
                # 如果奖励是负向的，那么预测的概率分布是负向坐标上
                # 在数学上，网络就会根据不同状态下+动作，预测出不同分布的回报分布，比如状态1下，正向回报概率多一些，状态2下，负向回报多一些；虽然在计算
                # 回报概率分布上，好像只有：奖励是正向的，那么预测概率分布都是集中上正向坐标上；奖励是负向的，那么预测的概率分布是负向坐标上。但是最终预测出来的
                # 会在正负上都有分布，因为网络更新是梯度更新，不会一次性更新到和计算的Q值分布一致，所以就会导致网路慢慢的左右调整，最终得到一个预测回报Q值有正负分布的回报
                proj_distr_v = distr_projection(last_distr_v, rewards_v, dones_mask,
                                                gamma=GAMMA**REWARD_STEPS, device=device)
                # 计算预测Q值网络预测的Q值概率分布和目标分布预测网路之间的KL分布差值
                prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v
                # 然后计算两者之间的差值大小，取最小化损失，使得预测的分布接近计算的分布
                critic_loss_v = prob_dist_v.sum(dim=1).mean()
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)

                # train actor
                # 以下优化动作的理解是：crt_net计算执行动作的Q值，如果想要让actor_loss_v最大，则会尽量调整执行的动作，使得预测出来的动作Q值的概率分布能够尽量分布在区间的正向区域（选择执行动作后，获得的正向区域概率较多/大的动作执行）
                # act_net预测执行的动作，crt_net计算预测到的Q值，由于crt_net网路不可改变，那么想要Q值最大，那么执行的动作会接近期望Q值是正的，远离期望q值是负的区域
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                crt_distr_v = crt_net(states_v, cur_actions_v)
                # 将预测到的Q值评价分布转换为期望Q值（单个值），加负号表示求最小值（也就是求最大Q值）
                actor_loss_v = -crt_net.distr_to_q(crt_distr_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                # 根据权重同步更新梯度后的权重到目标网络，之所以是权重大小更新，是否了防止部分错误的部分波动较大的回报干扰
                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
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

                    checkpoint = {
                        "act_net": act_net.state_dict(),
                        "crt_net": crt_net.state_dict(),
                        "tgt_act_net": tgt_act_net.target_model.state_dict(),
                        "tgt_crt_net": tgt_crt_net.target_model.state_dict(),
                        "act_opt": act_opt.state_dict(),
                        "crt_opt": crt_opt.state_dict(),
                        "frame_idx": frame_idx,
                    }
                    common.save_checkpoints(frame_idx, checkpoint, save_path, "d4pg", keep_last=10)

    pass
