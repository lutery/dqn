#!/usr/bin/env python3
'''
未适配
'''
import gymnasium as gym
import ptan
import argparse
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

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 15

# todo 作用
GRAD_BATCH = 64 # 控制用来训练的梯度数据缓存中长度，这里之所以将GRAD_BATCH设置的比较大，是因为损失计算和反向传播是非常繁重的任务，这样设置可以提高效率；如果设置的比较小，梯度传播很频繁，会导致因为同步梯度导致网络权重被锁，效率低下
TRAIN_BATCH = 2 # 用来控制几轮一次计算同步梯度到网路权重，由于使用了较大的GRAD_BATCH,那么TRAIN_BATCH就尽量小点，以便尽快让得到的梯度更新到网络权重，让子线程使用最新的权重进行推理，这样有利于效率

# 简单配置选择还是的游戏
# 不同的游戏有不同的结束训练的目标回报值
if True:
    ENV_NAME = "PongNoFrameskip-v4"
    NAME = 'pong'
    REWARD_BOUND = 18
else:
    ENV_NAME = "BreakoutNoFrameskip-v4"
    NAME = "breakout"
    REWARD_BOUND = 400
    TRAIN_BATCH = 4


def make_env():
    # # 创建环境，将不同环境统一成统一操作
    # 比如有些游戏只有一条命，有些游戏有多条命
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))


def grads_func(proc_name, net, device, train_queue):
    '''
    梯度传播子进程执行函数
    '''
    # 每个子进程创建NUM_ENVS个环境
    envs = [make_env() for _ in range(NUM_ENVS)]
    # 创建策略代理
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
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
                states_v, actions_t, vals_ref_v = \
                    common.unpack_batch(batch, net, last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                batch.clear()

                # 在PyTorch中，`zero_grad` 方法是用于将模型参数的梯度设置为零的。当我们在迭代和训练模型时，梯度是累积的，因此在每次开始新的前向和反向传播之前，我们需要清除之前的梯度值。`zero_grad` 就是用于此目的的。
                #
                # 在PyTorch中，你可以看到两种常见的 `zero_grad` 用法，它们分别与模型的 `nn.Module` 和优化器 `torch.optim.Optimizer` 相关。
                #
                # 1. **Module的 `zero_grad`**:
                #     ```python
                #     model.zero_grad()
                #     ```
                #     当你调用模型（一个 `nn.Module` 子类）的 `zero_grad` 方法时，它会为模型中的所有参数设置梯度为零。
                #
                # 2. **Optimizer的 `zero_grad`**:
                #     ```python
                #     optimizer.zero_grad()
                #     ```
                #     当你调用一个优化器的 `zero_grad` 方法时，它将为优化器正在跟踪的所有参数设置梯度为零。在大多数常见的使用场景中，优化器跟踪的参数是模型中的所有参数，因此这两者效果上是一样的。但在某些高级场景中，你可能有多个优化器，每个优化器负责不同的参数子集。在这种情况下，优化器的 `zero_grad` 只会为它跟踪的参数设置梯度为零。
                #
                # 总的来说，如果你只有一个优化器，它负责模型中的所有参数，那么两者是等效的。但在代码中，大多数人更倾向于使用优化器的 `zero_grad`，因为这样可以确保只为优化器管理的参数清除梯度，而不是模型中的所有参数。
                # 这里为什么要将模型的梯度清零？没有清空优化器的梯度
                # 因为优化是在主进程中做的，而不是在子进程，所以这里仅对net中梯度清零
                # 保证每次计算的梯度不会互相影响
                net.zero_grad()
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
                log_prob_actions_v = adv_v * log_prob_v[range(GRAD_BATCH), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                # 计算信息熵
                # 这里计算熵之所以没有负号，是因为和熵的计算公式中的符号直接对消了
                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # 将所有损失汇总，然后计算梯度
                # 子进程中没有使用优化器更新参数
                loss_v = entropy_loss_v + loss_value_v + loss_policy_v
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
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
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
