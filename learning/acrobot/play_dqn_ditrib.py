'''
'''
import os
import gymnasium as gym
import torch
import torch.nn as nn
from lib import dqn_model


Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

class DistributionalDQN(nn.Module):
    '''
    概率分布DQN网络
    '''

    def __init__(self, obs_size, n_actions):
        super(DistributionalDQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, N_ATOMS * n_actions)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        # 不同之处: 应该是用于计算交叉熵
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        param x: 输入的环境状态
        """
        # 获取总共有多少个训练样本
        batch_size = x.size()[0]
        # 进行计算推理
        out = self.net(x)
        # 重新reshape为(batch_size, n_classes, N_ATOMS)
        return out.view(batch_size, -1, N_ATOMS)

    def both(self, x):
        """
        param x: 输入的环境状态
        return: 返回cat_outhe(原始分布) res（下一个状态执行动作的Q值）
        """
        cat_out = self(x)
        # 在这段代码中，probs 是模型预测出的、在离散化的奖励值（也就是supports）上的概率分布。这是通过对网络输出应用 softmax 函数得到的，因为 softmax 函数可以将任何一组数值转化为有效的概率分布（所有概率之和为1，且每个概率都在0和1之间）。
        #
        # 然后，weights 是将这个概率分布（probs）与离散化的奖励值（supports）相乘得到的。这实际上是在计算每个离散奖励值的加权平均值，其中权重就是该奖励值的概率。这就产生了一个预期奖励值，这个预期奖励值是考虑了每个可能的奖励以及其对应的概率后的结果。
        #
        # 换句话说，weights 计算的是每个可能的奖励和对应概率的乘积，这也可以看作是每个可能奖励的"期望贡献"。将所有的 "期望贡献" 相加，就可以得到总的期望奖励，这是对网络预测的奖励分布的一种总结性描述。
        probs = self.apply_softmax(cat_out)
        # 将等差数列与probs相乘，因为supports的size为N_ATOMS
        # 加权平均
        weights = probs * self.supports
        # 将最后一维的数值进行求和计算
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        """

        """
        return self.both(x)[1]

    def apply_softmax(self, t):
        """
        计算推理结束后的softmax的值
        通过这样的计算，将每一次的预测结果都转换成了概率分布的形式
        也就是说，当我执行动作后，获取到这个Q值时的概率分布可能性的大小
        """
        # 将输入转换为（-1, N_ATOMS)形状
        # 计算softmax后再将shape还原，根据上下文，shape为(batch_size, n_classes, N_ATOMS)
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())

def play_acrobot():
    # Set up the environment
    env = gym.make("Acrobot-v1", render_mode="human")

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the DQN model
    net = DistributionalDQN(env.observation_space.shape[0], env.action_space.n).to(device)

    # Load the trained model
    save_path = "saves/dqn_distrib"
    model_path = os.path.join(save_path, "net.pth")

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    else:
        print("No saved model found")
        return

    # Play the game
    state, _ = env.reset()
    total_reward = 0
    done = False
    total_steps = 0

    while not done:
        state_v = torch.tensor(state).float().unsqueeze(0).to(device)
        q_vals = net.qvals(state_v)
        _, act_v = torch.max(q_vals, dim=1)
        action = int(act_v.item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        state = next_state
        done = terminated or truncated

        env.render()

    print(f"Game finished with total reward: {total_reward} and total steps: {total_steps}")
    env.close()


if __name__ == "__main__":
    play_acrobot()