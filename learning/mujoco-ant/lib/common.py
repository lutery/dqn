import numpy as np
import torch
from torch.autograd import Variable

import ptan


def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch: 收集的游戏数据
    :param net:
    :return: states variable, actions tensor, reference values variable（游戏环境状态、执行的动作、评价的Q值）
    """
    states = [] # 每一步的游戏状态
    actions = [] # 每一步执行的动作
    rewards = [] # 每一步执行动作后获取的奖励
    not_done_idx = [] # 执行动作后没有结束游戏的索引
    last_states = [] # 执行动作后的下一个状态
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(np.array(actions)).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        # 如果存在下一个状态的游戏数据，那么计算Q值
        # 对于已经结束的游戏动作，他们的Q值
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)[2]
        # todo 为什么这里计算的Q值没有选择最大值
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


def unpack_batch_ddqn(batch, device="cpu"):
    '''
    解压深度确定性策略梯度网络的数据
    '''
    # states: 环境状态
    # actions: 执行的动作
    # rewards： 执行动作后获取的奖励
    # dones: 执行动作后游戏是否结束
    # last_states: 未结束的游戏，执行动作后的达到的状态（针对多步展开，则是展开的最后一个动作）；如果是游戏已经结束的状态，则保存的还是和states中一样的状态，如果不是游戏结束的状态，则保存执行动作后的下一个状态
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.ByteTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v
