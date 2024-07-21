import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

HID_SIZE = 128


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
            nn.Linear(obs_size, HID_SIZE),
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


class SACActor(nn.Module):
    '''
    深度确定性策略梯度动作预测网络
    '''
    def __init__(self, obs_size, act_size, hidden_dim, action_range = 1., init_w = 3e-3, log_std_min = -20, log_std_max = 2):
        '''
        obs_size: 环境的维度
        act_size: 能够同时执行动作的个数（比如有多个手，不是每个手可以执行哪些动作）
        '''
        super(SACActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range
        self.action_range_tensor = torch.tensor(action_range, dtype=torch.float32)
        self.act_size = act_size

        # sac代码中对网络权重进行了初始化，这里是对网络权重进行初始化，是否是必须的？
        # 经过试验不是必须要的
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 和TD3的网络结构不同点，输出的是均值和对数标准差
        # 为什么采用对数标准差：
        # 在强化学习的上下文中，特别是在使用类似Soft Actor-Critic (SAC) 这样的算法时，`log_std`是指对数标准差，即动作分布的标准差的对数。这种表示方法在深度学习和统计模型中很常见，特别是在参数化某些概率分布（如正态分布）时。
        #
        # ### **为什么使用对数标准差（log_std）?**
        #
        # 使用对数标准差而不是直接使用标准差（std）有几个好处：
        #
        # 1. **数值稳定性**：通过将标准差的对数作为优化参数，可以确保在梯度下降过程中标准差保持正数。标准差必须是正数，而直接优化标准差可能在更新过程中导致其变为零或负数，这在数学上没有意义并可能导致运行时错误。
        #
        # 2. **无限制的值范围**：对数标准差可以取任意实数值（正无穷到负无穷），这使得优化过程更灵活和容易。对数转换后，参数的范围变得无约束，这对使用诸如梯度下降之类的优化算法更为方便。，而标准差只能是正数，所以采用对数标准差
        #
        # 3. **改善学习动态**：对数变换能够扩展参数的有效学习范围，使得小的变化也能在梯度下降中有效反映，增强了模型在参数空间的探索能力。
        #
        # ### **在策略网络中的应用**
        #
        # 在SAC等算法中，策略网络通常输出一个动作的均值（mean）和对数标准差（log_std）。这些输出被用来参数化动作的概率分布，通常是一个多元正态分布。策略网络的输出如下：
        #
        # - **均值（mean）**：描述了给定状态下预期动作的中心位置。
        # - **对数标准差（log_std）**：描述了动作取值的离散程度，高标准差意味着高不确定性和探索性，低标准差则表示动作更加确定。
        #
        # 在从策略中采样动作时，通常的步骤是首先从网络获取均值和对数标准差，计算标准差（通过指数化log_std），然后从以均值为中心、计算得到的标准差为离散度的正态分布中采样。使用`tanh`等激活函数可以进一步将动作限制在特定范围内（如[-1, 1]）。
        #
        # 总的来说，`log_std`在算法中的使用增强了模型的灵活性和稳定性，同时支持了有效的探索机制，这对于解决复杂的连续动作空间问题至关重要。
        self.mean_linear = nn.Linear(hidden_dim, act_size)
        self.log_std_linear = nn.Linear(hidden_dim, act_size)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean_linear(x)
        # 限制标准差的范围，作用如下：
        # 避免极端行为、数值稳定性、保持有效的探索、便于调参
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


    def evaluate(self, state, device, epsilon=1e-6):
        state_v = state
        mean, log_std = self(state_v)
        std = torch.exp(log_std)
        normal = dist.Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = torch.tanh(mean + std * z)
        # action = action * self.action_range
        log_prob = dist.Normal(mean, std).log_prob(mean + std * z) - torch.log(1. - action ** 2 + epsilon) - torch.log(self.action_range_tensor)
        log_prob = log_prob.sum(dim=1).unsqueeze(1)

        return action, log_prob, z, mean, log_std


    def get_action(self, state, device, greedy=False):
        # 与TD3的不同点，对于获取的动作探索
        # 采用的均值和方差的计算方式
        state_v = state
        # 获取均值和对数标准差
        mean, log_std = self.forward(state_v)
        mean, log_std = mean.data.cpu(), log_std.data.cpu()
        # 将对数标准差转换为标准差
        std = np.exp(log_std)
        # 构建标准正太分布，并从中采样和预测的动作均值一样的维度的动作噪声
        normal = dist.Normal(0, 1)
        z = normal.sample(mean.shape)
        # 将噪声添加仅预测的动作均值中，生成动作
        # 并将值压缩到[-1, 1]，在转换为实际游戏要执行的动作范围
        # 从数学上来说，mean + std * z是将上面均值为0标准差为1的噪声分布转换为
        # 预测的动作均值mean和标准差std的分布中，也就是对动作进行了扰动，也就是探索
        # todo 这样做之后，虽然进行了随机探索，但是由于公式的是明确的，所以梯度是可以传播的
        # 整个表达式是可微的，可以用来训练网络
        action = torch.tanh(mean + std * z)
        action = action * self.action_range

        # 如果在测试、验证过程中，需要将greedy设置为True
        # 这个时候就不添加噪声，而是直接输出mean
        action = self.action_range * torch.tanh(mean) if greedy else action
        return action.cpu().numpy()


    def sample_action(self):
        action = torch.rand((1, self.act_size)) * 2 - 1
        return (self.action_range * action).cpu().numpy()


class TD3Actor(nn.Module):
    '''
    深度确定性策略梯度动作预测网络
    '''
    def __init__(self, obs_size, act_size, action_range = 1):
        '''
        obs_size: 环境的维度
        act_size: 能够同时执行动作的个数（比如有多个手，不是每个手可以执行哪些动作）
        '''
        super(TD3Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

        self.action_range = action_range

    def forward(self, x):
        return self.net(x) * self.action_range


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
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
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
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        # 构建观察+动作合并网路，输出评价Q值
        # 以前的Q值网络是输出每个动作的Q值，在连续值里面是直接输出评价Q值
        # 这里的400是self.obs_net的输出维度，act_size是动作网络的输出维度
        # 为了后续合并预测评价做准备
        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
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


class AgentAcktr(ptan.agent.BaseAgent):
    '''
    创建代理器
    '''
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        '''
        states: 观测的环境状态
        agent_states：智能体自己的状态，在这里是没有使用的
        '''
        # 创建环境预处理器，将环境状态转换为float32类型
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)

        # 通过环境状态预测执行的动作
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        # 该动作的作用，是对预测的动作添加随机噪音，实现动作的探索
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        # 将执行的动作压缩到-1到1中，可能是因为输入给网络的值不能超过-1和1
        actions = np.clip(actions, -1, 1)
        return actions, agent_states



class AgentPPO(ptan.agent.BaseAgent):
    '''
    创建代理器
    '''
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        '''
        states: 观测的环境状态
        agent_states：智能体自己的状态，在这里是没有使用的
        '''
        # 创建环境预处理器，将环境状态转换为float32类型
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)

        # 通过环境状态预测执行的动作
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        # 该动作的作用，是对预测的动作添加随机噪音，实现动作的探索
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        # 将执行的动作压缩到-1到1中，可能是因为输入给网络的值不能超过-1和1
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    实现一个具备探索能力的智能体
    """
    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        '''
        net: 动作执行网络
        device: 执行的设备
        ou_enabled: 是否开启OU过程探索
        ou_mu: 查看OU过程说明
        ou_teta: 查看OU过程说明
        ou_sigma: 查看OU过程说明
        ou_epsilon: 用于控制OU过程对最终动作的执行影响大小

        Ornstein-Uhlenbeck（OU）过程是一个用来生成噪声的数学工具，特别适合用于模拟物理过程中存在的摩擦或阻力的场景，因此它在强化学习中被用作探索机制，尤其是在连续动作空间的问题中。

        在使用OU过程时，参数`ou_teta`、`ou_mu`、`ou_sigma`通常是经验性选择的，他们分别控制：

        1. **`ou_teta`（Theta）** - 这个参数决定了系统回到均值（或趋向平稳状态）的速度。较高的`ou_teta`值会使系统更快地返回到均值，即产生的噪声会更快地趋于稳定状态。如果`ou_teta`设置得太高，系统可能会过于迅速地稳定下来，从而减少探索。太低，则可能导致系统的探索过于缓慢和延迟。

        2. **`ou_mu`（Mu）** - 这是长期平均或均值项，OU过程将围绕此均值进行波动。在强化学习的上下文中，`ou_mu`通常设置为0，这意味着没有任何行动的偏好，确保探索不会偏向于任何特定的行动。

        3. **`ou_sigma`（Sigma）** - 这个参数代表了波动的振幅或标准差，决定了噪声的强度或大小。较大的`ou_sigma`增加了探索的范围和随机性，但可能导致智能体行为变得不稳定；较小的值将减少探索性噪声，可能导致智能体过早收敛到次优策略。

        在确定这些参数时，通常需要通过实验调整以找到适合特定任务的值。一些普遍的做法和建议包括：

        - 开始时使用中等范围的值进行初步测试。
        - 观察智能体的表现，如果它行动过于随机，可以降低`ou_sigma`。
        - 如果智能体似乎探索得不够，或者过早收敛到一个行动上，可以增加`ou_sigma`。
        - 调整`ou_teta`以控制噪声的“记忆”效果，即噪声持续影响行动的程度。

        调整这些参数通常需要反复试验，因为理想的值很大程度上取决于具体问题、智能体的体系结构、奖励结构以及训练过程的其他方面。此外，有时会随着训练的进展而逐步减少噪声的影响，这是通过减少`ou_epsilon`来实现的，这种做法类似于退火。

        OU过程中的参数值ou_mu、ou_theta、ou_sigma和ou_dt通常根据具体问题来确定,没有一个通用的设置方法。下面是一些确定这些参数的常见思考:

        1. ou_mu - 长期均值可以设置为目标值或期望的平均奖励值。

        2. ou_theta - 回归速度可以先设置为一个较小的值(例如0.15),表示轻度回归到均值。可以通过调参来获得更好的回归速度。

        3. ou_sigma - 随机扰动标准差可以先设置较小(例如0.3),然后通过调参确定合适的探索强度。

        4. ou_dt - 时间步长可以设置得较小(例如0.01),然后确保dt足够小以维持迭代稳定性。

        5. ou_x0 - 通常初始化为0即可。

        另外,这些参数还需要互相平衡。例如theta越大,则需要更大的sigma来增加探索。

        一个通用的思路是:

        - 先用较小的回归速度θ,较大的扰动σ开始,更偏向探索。

        - 然后观察过程的变化,如果变化太随机可以适当增大θ或减小σ。

        - 调节参数直到得到既能利用已知信息又能适度探索的平衡效果。

        所以确定OU过程的参数需要根据具体问题,通过试错来找到一个合适的组合。
        '''
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

class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        '''
        :param obs_size: 观测的环境维度
        :param act_size: 执行的动作的维度
        '''
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        # 作用 看名称像是方差：是用来控制动作的探索程度的
        # 怎么更新？：在训练的过程中，会不断的更新这个参数，更新的逻辑就在于计算熵损失以及计算动作优势大小的时候会参与计算，然后在梯度更新的时候，会自动更新这个参数到合适的大小
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)


class ModelCritic(nn.Module):
    '''
    trop信赖域策略优化评价网络
    ACKTR算法中使用的critic网络
    ppt优化评价网络
    '''
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


class PPO_ModelCritic_With_ACTION(nn.Module):
    '''
    trop信赖域策略优化评价网络
    ACKTR算法中使用的critic网络
    ppt优化评价网络
    '''
    def __init__(self, obs_size, action_size):
        super(PPO_ModelCritic_With_ACTION, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size + action_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x, action):
        return self.value(torch.cat([x, action], dim=1))



class AgentTRPO(ptan.agent.BaseAgent):
    '''
    创建代理器
    '''
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        '''
        states: 观测的环境状态
        agent_states：智能体自己的状态，在这里是没有使用的
        '''
        # 创建环境预处理器，将环境状态转换为float32类型
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)

        # 通过环境状态预测执行的动作
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        # 该动作的作用，是对预测的动作添加随机噪音，实现动作的探索
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        # 将执行的动作压缩到-1到1中，可能是因为输入给网络的值不能超过-1和1
        actions = np.clip(actions, -1, 1)
        return actions, agent_states