import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 256


'''
在强化学习的Actor-Critic方法中，特别是在处理连续动作空间时，网络通常预测动作的均值（`mu`）和方差（`var`）。在你给出的代码中，模型使用不同的激活函数来处理这两个输出，即`tanh`用于均值，而`Softplus`用于方差。让我们分别来看看这两个激活函数的选择原因：

### 1. 均值 (`mu`) 的 `tanh` 激活函数

- **动作范围限制**：`tanh` 函数的输出范围在 [-1, 1] 之间。在许多连续动作空间的问题中，动作的有效范围被限制在这个区间内。例如，如果你在一个物理模拟环境中控制机械臂，动作可能是旋转角度，这些角度通常有最大和最小值。

- **输出规范化**：使用 `tanh` 确保网络输出的均值永远不会超出这个范围，这样可以保证生成的动作始终在有效的动作空间内。

### 2. 方差 (`var`) 的 `Softplus` 激活函数

- **保证正值**：方差是衡量数据离散程度的度量，它必须是正值。`Softplus` 函数是一个平滑的函数，其输出始终为正值。`Softplus` 函数的公式是 `log(1 + exp(x))`，对于所有的输入 `x`，输出都是正的。

- **灵活性**：与 `ReLU` 函数相比，`Softplus` 在 `x` 接近0时更加平滑，这允

许更精细的控制方差的值。`Softplus` 提供了一种有效的方式来表示方差，即使在其原始预测值接近零或负值时也能保持正数。这在学习过程中非常有用，特别是当网络还未充分训练，对方差的估计可能接近零或变化较大时。

### 结合使用 `tanh` 和 `Softplus`

在这个A2C模型中，`tanh` 用于输出规范化到特定范围内的动作均值，而 `Softplus` 用于确保方差始终为正值且平滑。这种设计使模型能够有效地操作连续动作空间，同时保持必要的灵活性和稳定性。使用这两个激活函数的组合，模型可以生成符合实际物理限制的动作（如限制在一定范围内的力或速度），并且以合理的方式探索这些动作的不确定性。
'''
class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        '''
        obs_size: 观测的数据维度，不是图像数据，所以后续用的是全连接层
        act_size: 动作空间的维度，在这个游戏里面，指的是同时执行动作的数量
        '''

        super(ModelA2C, self).__init__()

        # 只有这个是提取特征，其余的都是输出结果
        self.base = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, HID_SIZE),
            nn.ReLU(),
        )
        # 输出均值，表示最终要执行的动作内容
        # 这里由于预测的输出动作包含负值，所以采用tanh函数，将输出值限制在-1到1之间
        # 而不是用sigmoid
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        # https: // zhuanlan.zhihu.com / p / 461707201
        # var作用 方差平方，怀疑是用来稳定输出的概率范围大小
        # 用来促进网络进行探索以及指导网路朝哪个方向进行训练，使得整体趋近中值，但是在这里并没有直接使用方差，而是使用了信息熵的方式
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(), # Relu的替代函数，用于解决梯度消失问题 具体使用场景查看笔记内容
        )
        # 状态值（Q值），用来评价当前Q值，来评估当前执行的动作是否有优势
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        '''
        return 均值，方差平方，Q值
        '''
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class DDPGActor(nn.Module):
    '''
    深度确定性策略梯度动作预测网络
    '''
    def __init__(self, obs_size, act_size):
        '''
        obs_size: 环境的维度
        act_size: 能够同时执行动作的个数（比如有多个手，不是每个手可以执行哪些动作）
        '''
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    '''
    深度确定性策略梯度网络Q值评价网络
    '''
    def __init__(self, obs_size, act_size):
        '''
        obs_size: 环境的维度
        act_size: 能够同时执行动作的个数（比如有多个手，不是每个手可以执行哪些动作）
        '''
        super(DDPGCritic, self).__init__()

        # 构建环境观察网络
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
        )

        # 构建观察+动作合并网路，输出评价Q值
        # 以前的Q值网络是输出每个动作的Q值，在连续值里面是直接输出评价Q值
        # 这里的400是self.obs_net的输出维度，act_size是动作网络的输出维度
        # 为了后续合并预测评价做准备
        self.out_net = nn.Sequential(
            nn.Linear(512 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


class D4PGCritic(nn.Module):
    '''
    D4PG自己的Q值评价网路
    '''
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()

        # 环境特征采样观测网络
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        # 环境特征采样+动作特征采样组合特征采集网络
        # 不同之处是返回的是一个Q值的概率分布
        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, n_atoms)
        )

        # 概率分布每个区间的大小
        delta = (v_max - v_min) / (n_atoms - 1)
        # todo supports作用
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x, a):
        '''
        推理并返回预测Q值的概率分布
        '''

        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        '''
        将Q值分布转换为Q值
        '''
        # 计算概率分布，将概率分布乘以回报分布（-10到10），得到回报分布值
        weights = F.softmax(distr, dim=1) * self.supports
        # 计算加权回报分布得到期望Q值
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)


class AgentA2C(ptan.agent.BaseAgent):
    '''
    A2C连续策略值网络代理，之所以自己写，是因为需要应对连续值的动作输出
    '''
    def __init__(self, net, device="cpu"):
        '''
        net: 神经网路
        device: 指定网路的设备
        '''
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        '''
        states: 观察的状态
        agent_state: 代理器的状态，在这里没有直接使用，在某些地方可能需要用到代理器的内部状态

        return: 选择执行的动作值，代理器的状态
        '''
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)

        # 将观测传入网络得到均值和方差
        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        # 以下实在计算高斯分布吗？
        # 不是：以mu为均值，sigma为标准差的正态分布中抽取随机数，作为每个状态下采取的动作。在连续动作空间的问题中，这种方法能够在探索和利用之间取得平衡。随机性（通过sigma体现）允许探索不同的动作，而均值（mu）通常代表了当前网络认为最佳的动作。
        # 综上所述，这个是在从均值和方差中随机选择需要执行的动作值
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    实现一个具备探索能力的智能体
    """
    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        # 统一接口，但是这里不需要使用，返回None而不是使用pass
        return None

    def __call__(self, states, agent_states):
        '''
        states：当前的环境状态
        agent_states: 内部智能体的状态,之前的代理器里面这个基本无用,一开始的时候，agent_states是空的，但在这里因为要使用OU过程对执行的动作进行噪音干扰，所以需要使用了智能体的内部状态
        '''
        # 将环境转换为目标的数据类型
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        # 得到执行的动作输出
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            # 启动随机探索
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    # 为什么智能体状态为空时可以构建一个维度和动作相同，且全为零的状态
                    # 这是针对初始状态，初始情况下，智能体的状态为空，则创建一个为0的状态
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                # 书p298也 todo 了解OU过程
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        # 这个步骤 的意思是修正动作值到-1和1之间，否则将导致PyBullet抛出异常
        actions = np.clip(actions, -1, 1)
        return actions, new_a_states


class AgentD4PG(ptan.agent.BaseAgent):
    """
    Agent implementing noisy agent
    实现一个具备噪音的代理器
    """
    def __init__(self, net, device="cpu", epsilon=0.3):
        self.net = net
        self.device = device
        self.epsilon = epsilon

    def __call__(self, states, agent_states):
        '''
        states: 环境的状态
        agent_states: 智能体代理器的状态，在这里不需要关注这个状态

        return 执行的动作值，智能体的状态
        '''
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        # 预测执行的动作值
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        # 给动作值添加噪音 todo 这里可以直接这么简单的+添加噪音吗？
        actions += self.epsilon * np.random.normal(size=actions.shape)
        # 防止动作值超过边界
        actions = np.clip(actions, -1, 1)
        return actions, agent_states
