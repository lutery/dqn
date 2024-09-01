import sys
import time
import numpy as np
import torch
import torch.nn as nn

# 游戏的统一参数
HYPERPARAMS = {
    'LunarLander': {
        'env_name':         "LunarLander-v2",
        'stop_reward':      200.0,
        'run_name':         'LunarLander',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    }
}


def unpack_batch(batch):
    '''
    将经验重放缓冲区的经验进行解包操作，从经验中获取各个经验类别
    '''

    # states 每次采集样本时的环境状态
    # actions 每次状态下执行的动作值
    # rewards 每次执行完动作后获取的激励值
    # dones 执行动作后游戏是否结束
    # last_states 执行动作之后的下一个状态
    #
    # return 将states, actions, rewards, dones, last_states各个list转换为numpy
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    '''
    计算dqn的损失值
    net: 负责计算当前状态下的动作Q值
    tag_net: 负责计算下一个状态下的动作Q值
    '''
    # 提取样本集中游戏的各个状态和动作
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    # 将数据传输到指定的设备中
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # 将状态传输到神经网路中，获取到神经网路推理出来的执行的动作
    # 网络细节查看第06章 02_dqn_pong.py
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # tag_net负责计算下一个状态的Q值
    # 并将其中会导致游戏结束的动作的Q值设置为0，这样可以将不好的q值降低
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    # 损失值计算：下一个状态的最大Q值+实际获取到的激励值 == 当前net计算出来的Q值
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


"""
该类就是用来跟踪、记录、判断激励的追踪类
"""
class RewardTracker:
    def __init__(self, writer, stop_reward):
        '''
        param writer: tensorboard writer保存
        param stop_reward: 停止训练的激励值\目标值
        '''

        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        # total_rewards 训练期间的每一步的激励值，用来记录
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        '''
        param reward: 样本
        param fream: 当前进行了第frame次的训练
        param epsilon：当前的epsilon值

        return True: 表示已经达到了目标激励值 False： 表示还没有达到目标的激励值
        '''
        # 激励经验存储在总缓存区
        self.total_rewards.append(reward)
        # 计算当前的平均帧率
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        # 将当前帧总数和所花费的时间存储在缓存中
        self.ts_frame = frame
        self.ts = time.time()
        # 计算平均激励值
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        # 如果当前获取的激励已经达到了目标的激励大小，则返回true
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

# Epsilon跟踪器
class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        '''
        这个跟踪器是根据当前执行的帧数大小，以及epslison的参数来计算动作选择其中
        选择动作的概率

        param epsilon_greedy_selector: 动作选择器
        param params: map类型的参数，也就是本文中的HYPERPARAMS参数
        '''

        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        '''
        根据当前的帧数，更新epsilon的值，eplison是用来决定每次执行的策略是
        随机还是神经网络推理获取的
        '''
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    这里就是通过人工的方式得到一个更加合理的概率分布坐标，然后通过kl散步，得到原始坐标分布和
    投影坐标分布之间的差值作为损失函数，进行优化拟合神经网络

    param next_distr: 下一个状态最大Q值执行动作的概率分布
    param rewards: 每次执行动作获取的激励
    param dones: 游戏是否结束
    param Vmin:
    param Vmax:
    param n_atoms: 分布的范围
    param gama:

    """
    batch_size = len(rewards)
    # 创建保存投影结果的数组
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1) # 计算投影坐标之间的间隔
    for atom in range(n_atoms):
        # 计算激励在（todo 不确定 分布坐标系里面的位置），并保证计算的结果不会越界
        # 计算的公式 pos= 当前的激励 +（坐标最小值 + 当前的坐标位置 * 每个坐标之间的间隔）* 遗忘参数
        # tz_j: 计算当前激励在当前的坐标下偏移的位置
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        # 但是实际上在表示时，实际索引是从0开始的，所以这里计算的是以0为起点时的索引坐标
        # 如果激励是0，那么投影的坐标是在原点上，如果激励是其他值，那么投影的坐标将偏移出
        # 原点
        # 这里是要找出投影的坐标在原点上的位置动作激励
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        # 如果偏移的坐标正好在原点上，则直接将当前的Q值的对用的分布坐标值累计到proj_distr中
        # l[eq_mask]表示reward落在坐标点上的坐标索引，将这些Q值分布值累加在proj_distr
        # 而l[ne_mask]同理，表示落在原点之间的值该如何累加分布到两个原点上
        # 这里实际上就是在计算源坐标到目标的投影坐标并将值累加进去
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        # 如果偏移坐标落在两个原点之间，那么就需要按照线性插值算法一样，根据距离两个
        # 原点的距离值分配Q值到两个原点上
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        # 如果存在已经结束的点值，那么将已经结束的索引位置的概率分布值全部设置为0
        proj_distr[dones] = 0.0
        # 然后在根据激励值计算在概率分布上的坐标位置
        # 通常情况下，正激励表示整理，负激励表示失败
        # 所以这里正激励正好落在坐标上，负激励则不在坐标上
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        # 这边也依旧是找到结束激励正好坐落在概率分布坐标位置上的索引
        eq_mask = u == l
        # 通过深拷贝，得到一个哪些结束激励正好在概率分布坐标位置上的索引bool列表
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        # 如果存在正好在坐标位置上的索引，则说明该游戏是正结束（也就是胜利）
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        # 得到结束概率分布不在坐标位置上的索引位置
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            # 这边应该就是计算，游戏结束时，不在概率分布坐标索引位置上的激励的投影坐标
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr
