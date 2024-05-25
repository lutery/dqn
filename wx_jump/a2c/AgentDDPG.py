import numpy as np
import torch

def float32_preprocessor(states):
    '''
    将状态矩阵转换为float32数值类型的tensor
    '''
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class AgentDDPG():
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

    def call(self, states, agent_states):
        '''
        states：当前的环境状态
        agent_states: 内部智能体的状态,之前的代理器里面这个基本无用,一开始的时候，agent_states是空的，但在这里因为要使用OU过程对执行的动作进行噪音干扰，所以需要使用了智能体的内部状态
        '''
        # 将环境转换为目标的数据类型
        states_v = float32_preprocessor(states).to(self.device)
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
        actions = np.clip(actions, 0, 1.1)
        return actions, new_a_states