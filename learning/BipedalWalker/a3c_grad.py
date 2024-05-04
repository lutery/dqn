#!/usr/bin/env python3
# 验证可以运行，但是貌似没办法达到训练目标
import gymnasium as gym
import ptan
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import math
from lib import common
from lib import model

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 15

# todo 作用
GRAD_BATCH = 64 # 控制用来训练的梯度数据缓存中长度，这里之所以将GRAD_BATCH设置的比较大，是因为损失计算和反向传播是非常繁重的任务，这样设置可以提高效率；如果设置的比较小，梯度传播很频繁，会导致因为同步梯度导致网络权重被锁，效率低下
TRAIN_BATCH = 2 # 用来控制几轮一次计算同步梯度到网路权重，由于使用了较大的GRAD_BATCH,那么TRAIN_BATCH就尽量小点，以便尽快让得到的梯度更新到网络权重，让子线程使用最新的权重进行推理，这样有利于效率

# 简单配置选择还是的游戏
# 不同的游戏有不同的结束训练的目标回报值
ENV_NAME = "BipedalWalker-v3"
NAME = 'BipedalWalker'
REWARD_BOUND = 18


def make_env():
    # # 创建环境，将不同环境统一成统一操作
    # 比如有些游戏只有一条命，有些游戏有多条命
    return gym.make(ENV_NAME)

def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def grads_func(proc_name, net, device, train_queue):
    '''
    梯度传播子进程执行函数
    '''
    # 每个子进程创建NUM_ENVS个环境
    envs = [make_env() for _ in range(NUM_ENVS)]
    # 创建策略代理
    agent = model.AgentA2C(net, device=device)
    # 创建经验源，经验源本身支持多环境
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    batch = []
    frame_idx = 0
    # 为记录统计设置名字，设置为每个子进程的名字
    writer = SummaryWriter(comment=proc_name)

    # todo 完善RewardTracker注释 作用统计近100次游戏的奖励，如果奖励达标则退出游戏
    with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
        # todo 完善TBMeanTracker注释，作用将最近的10步游戏记录到tensorboard中
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            # 从经验源中获取经验样本
            for exp in exp_source:
                frame_idx += 1
                # 获取经验源中所有的回报
                # pop_total_rewards存在返回值，那么表示一次游戏
                # 已经结束
                new_rewards = exp_source.pop_total_rewards()
                # 判断游戏目前运行获取的奖励是否已经达到标准，达到了就退出训练
                if new_rewards and tracker.reward(new_rewards[0], frame_idx):
                    break

                # 判断目前已经收集到的经验训练数据数是否已经达到目标
                batch.append(exp)
                if len(batch) < GRAD_BATCH:
                    continue

                # 解压出训练数据
                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_a2c(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                # 清空收集的游戏数据
                batch.clear()
                net.zero_grad()
                # 将游戏状态传入神经网络，得到预测的均值、方差、Q值
                mu_v, var_v, value_v = net(states_v)
                # 第一个损失，预测的Q值需要和计算的Q值接近
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                # 去负值，计算最小值
                loss_policy_v = -log_prob_v.mean()
                # 高斯概率密度分度熵的计算公式 P292页，同样这里也取了负值
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2 * math.pi * var_v) + 1) / 2).mean()

                # 损失之和，计算最小值，更细梯度梯度
                loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                loss_v.backward()

                # 将相关的鹅值记录起来
                tb_tracker.track("advantage", adv_v, frame_idx)
                tb_tracker.track("values", value_v, frame_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, frame_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, frame_idx)
                tb_tracker.track("loss_policy", loss_policy_v, frame_idx)
                tb_tracker.track("loss_value", loss_value_v, frame_idx)
                tb_tracker.track("loss_total", loss_v, frame_idx)

                # gather gradients
                # 梯度裁剪，防止梯度过大或者过小
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                # 将网络参数中每个参数的梯度提取出来
                grads = [param.grad.data.cpu().numpy() if param.grad is not None else None
                         for param in net.parameters()]
                # 放到train_queue共享队列中
                train_queue.put(grads)

    # 这里之所以将None置入队列，是因为主进程判断到队列中有None，说明已经检测到满足了
    # 训练的目标，可以达到目标的奖励值了
    train_queue.put(None)


if __name__ == "__main__":
    # 该方法主要有两个作用:
    #
    # 设置多进程的启动方式(spawn或fork)。默认是spawn。
    # 避免多进程模块的多次导入导致的错误。
    # 关于启动方式:
    #
    # spawn: 子进程将只继承父进程的信号处理机制,而不会继承父进程的句柄、锁等资源。所以spawn方式更安全,也是推荐的默认方式。
    # fork: 子进程通过fork启动,会通过复制父进程获得父进程的全部资源,包括锁、信号处理等,所以可能会导致在多进程中出现问题。
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", default="a3c-grad", required=False, help="Name of the run")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    env = make_env()
    net = common.BipedalWalkerModelA2C(env.observation_space.shape, env.action_space.shape).to(device)
    # 生命网络net为共享内存，share_memory必须在调用fork前调用
    # 这样net就可以很方便的在每个进程间使用了
    # 使用share_memory会共享网络权重，而不会共享计算图、梯度
    # 在PyTorch中，`share_memory`方法是针对多进程场景设计的，特别是当使用`torch.multiprocessing`时。当你在主进程中创建一个`nn.Module`模型并想在多个子进程中共享这个模型时，使用`share_memory`方法可以将模型的权重存放在共享内存中，从而允许多个子进程访问。
    #
    # 然而，这并不意味着梯度是共享的。当每个子进程对模型进行前向和反向传播时，它们各自会有自己的计算图和梯度存储。这意味着，如果多个进程同时对同一个模型进行反向传播，它们不会在共享内存中累积梯度。
    #
    # 如果你想在多个进程中累积梯度或同步模型参数，你需要使用更高级的同步策略。这种策略在分布式训练场景中更为常见，例如使用`torch.nn.parallel.DistributedDataParallel`。
    #
    # 简而言之，虽然`share_memory`方法允许在多进程中共享模型权重，但梯度并不共享。每个进程都会有自己的计算图和梯度存储。
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    # 创建子进程共享队列，最大存储的数量为子进程的数量
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for proc_idx in range(PROCESSES_COUNT):
        # 创建子进程
        # 为啥梯度同步需要创建进程名字
        # 因为每个梯度同步就是每个子进程内部都包含一个自己训练网路网络的流程
        # 由于要记录summary，所以需要区分每个进程的记录名字
        proc_name = "-a3c-grad_" + NAME + "_" + args.name + "#%d" % proc_idx
        data_proc = mp.Process(target=grads_func, args=(proc_name, net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0
    # 梯度缓冲区，接受子进程中返回的梯度信息
    grad_buffer = None

    try:
        while True:
            # 从梯度进程共享缓冲区提取梯度信息
            # get时会等待，知道有数据，每次get只会获取一个梯度信息
            # 而根据返回值，每次返回的梯度信息均包含每个网路节点的梯度信息
            # 顺序则按照parameters返回值而定
            train_entry = train_queue.get()
            if train_entry is None:
                break

            step_idx += 1

            # 接收子进程中的梯度信息
            if grad_buffer is None:
                # 如果grad_buffer是空，则直接赋值
                grad_buffer = train_entry
            else:
                # 如果grad_buffer不为空，则需要将gard_buffer和train_entry中的梯度相加汇总（符合书上）
                # 而多个子进程的梯度信息也只是简单的加起来
                for tgt_grad, grad in zip(grad_buffer, train_entry):
                    tgt_grad += grad

            # 判断是否需要同步梯度到网路权重
            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buffer):
                    # 这里是将子进程汇总的梯度设置到对应的权重的梯度上
                    param.grad = torch.FloatTensor(grad).to(device)

                # 梯度裁剪
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                # 梯度更新，更新网路权重
                optimizer.step()
                grad_buffer = None
    finally:
        # 之所以这里要额外进行这个动作，是为了防止僵尸进程占据GPU资源
        # 因为这里是采用多进程的方式，主进程退出子进程很可能不会退出
        for p in data_proc_list:
            p.terminate()
            p.join()
