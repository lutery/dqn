#!/usr/bin/env python3
'''
未适配
'''
import gymnasium as gym
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
# 训练数据样本缓冲区的长度,这里之所以训练数据缓冲区的长度比之前的长
# 是因为多个环境并行获取数据，缓冲区内的训练数据基本都是最新的，如果是单个
# 环境下，这么长的缓冲区，其必定会有过旧的训练数据
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

# 创建子进程的总数
PROCESSES_COUNT = 4
NUM_ENVS = 15

# 简单配置选择还是的游戏
if True:
    ENV_NAME = "PongNoFrameskip-v4"
    NAME = 'pong'
    REWARD_BOUND = 18
else:
    ENV_NAME = "BreakoutNoFrameskip-v4"
    NAME = "breakout"
    REWARD_BOUND = 400


def make_env():
    # 创建环境，将不同环境统一成统一操作
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

# 创建一个可以给成员命名的元组，可以类似于类一样使用
TotalReward = collections.namedtuple('TotalReward', field_names='reward')


def data_func(net, device, train_queue):
    '''
    子进程主要执行的动作是
    不断的从环境中获取数据，并将其存储到共享游戏数据队列中
    '''

    # 每个子进程创建NUM_ENVS个环境
    envs = [make_env() for _ in range(NUM_ENVS)]
    # 创建策略代理
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    # 创建经验源，经验源本身支持多环境
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    for exp in exp_source:
        # 获取经验源中所有的回报
        # pop_total_rewards存在返回值，那么表示一次游戏
        # 已经结束
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            # 将游戏结束奖励存储到train_queue中
            # np.mean计算游戏统计游戏多次结束的奖励平均值
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        # 如果游戏没有结束，则将目前进行的游戏记录的数据记录在train_queue中
        train_queue.put(exp)


if __name__ == "__main__":
    # 该方法主要有两个作用:
    #
    # 设置多进程的启动方式(spawn或fork)。默认是spawn。
    # 避免多进程模块的多次导入导致的错误。
    # 关于启动方式:
    #
    # spawn: 子进程将只继承父进程的信号处理机制,而不会继承父进程的句柄、锁等资源。所以spawn方式更安全,也是推荐的默认方式。
    # fork: 子进程通过fork启动,会通过复制父进程获得父进程的全部资源,包括锁、信号处理等,所以可能会导致在多进程中出现问题。
    # 由于pytorch的限制，spawn是最佳的方式
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", default='a3c-data', help="Name of the run")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    # 为记录统计设置名字
    writer = SummaryWriter(comment="-a3c-data_" + NAME + "_" + args.name)

    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    # 生命网络net为共享内存，share_memory必须在调用fork前调用
    # 这样net就可以很方便的在每个进程间使用了
    # 如果net设置的dev是cuda，那么网络张量默认的共享的，不用调用share_memory
    # 如果net设置的dev是cpu，那么网络张量是不共享的，需要调用share_memory
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    # 创建子进程共享队列，最大存储的数量为子进程的数量
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        # 创建子进程，将net、目标设备、训练子进程队列传入子进程的参数中
        data_proc = mp.Process(target=data_func, args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0

    try:
        # todo 完善RewardTracker注释 作用统计近100次游戏的奖励，如果奖励达标则退出游戏
        with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
            # todo 完善TBMeanTracker注释，作用将最近的10步游戏记录到tensorboard中
            with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
                while True:
                    # 从子进程获取环境数据
                    train_entry = train_queue.get()
                    # todo 从代码可知，train_entry有两种类型，其中一种类型是TotalReward
                    #  TotalReward内部只有奖励回报，为什么可以混在一起不会出错
                    if isinstance(train_entry, TotalReward):
                        # 如果感知到有一次游戏结束，则判断获取的奖励是否符合训练结束的条件
                        if tracker.reward(train_entry.reward, step_idx):
                            break
                        continue

                    step_idx += 1
                    batch.append(train_entry)
                    if len(batch) < BATCH_SIZE:
                        continue

                    states_v, actions_t, vals_ref_v = \
                        common.unpack_batch(batch, net, last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                    batch.clear()

                    # 清空梯度
                    optimizer.zero_grad()
                    # 得到动作预测，当前状态下的Q值
                    logits_v, value_v = net(states_v)

                    # 训练网络预测Q值，预测出来的值和bellman计算出来的Q值应该接近
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                    # 计算每个动作的概率
                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    # 得到执行动作的评价，两边都是Q值，一个Q值是网路根据以往经验预测出来的，一个是根据实际数据计算出来的
                    # 而Q值分为状态Q值和动作Q值，状态Q值是固定的，所以会造成差异的是动作Q值
                    # 而这点就可以拿来作为对执行动作的评价
                    # 如果adv_v是正数，则表示当前执行的动作比原先已经记录的动作更好
                    # 如果adv_v是零，则表示当前执行的动作和原先一样
                    # 如果adv_v是负数，则表示执行的动作比原先已经记录的动作更差
                    adv_v = vals_ref_v - value_v.detach()
                    # 这里会将动作评价和实际执行动作对应的概率相乘
                    # 为了使得log_prob_action_v最大（loss_policy_v最小）
                    # 在梯度更新的时候，就会朝着将adv_v负数的哪个动作概率调整小，adv_v正数
                    # 的动作概率调整大
                    log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                    loss_policy_v = -log_prob_actions_v.mean()

                    prob_v = F.softmax(logits_v, dim=1)
                    # 计算信息熵
                    # 这里计算熵之所以没有负号，是因为和熵的计算公式中的符号直接对消了
                    entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                    # 将所有损失汇总，然后计算梯度
                    loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                    loss_v.backward()
                    # 梯度裁剪，防止梯度过大或者过小
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    # 更新参数
                    optimizer.step()

                    # 将相关的值记录起来
                    tb_tracker.track("advantage", adv_v, step_idx)
                    tb_tracker.track("values", value_v, step_idx)
                    tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                    tb_tracker.track("loss_value", loss_value_v, step_idx)
                    tb_tracker.track("loss_total", loss_v, step_idx)
    finally:
        # 防止出现异常时，没有管子子进程
        for p in data_proc_list:
            p.terminate()
            p.join()
