#!/usr/bin/env python3
'''
未验证
'''
import gymnasium as gym
import ptan
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import common

SAVE_STATES_IMG = False
SAVE_TRANSITIONS_IMG = False

if SAVE_STATES_IMG or SAVE_TRANSITIONS_IMG:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pylab as plt

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


class DistributionalDQN(nn.Module):
    '''
    概率分布DQN网络
    '''

    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # 不同之处：输出口数量有n_actions * N_ATOMS
        # 输出执行每个动作的Q值概率分布
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

        # 不同之处：注册了一个supports遍历
        # 创建一个范围在[Vmin, Vmax]范围内的数值区间，以DELTA_Z为差值得到等差数列
        # 这里supports可以看作所有动作的Q值分布
        # 如果每个动作的分布在Q值分布上相同，那么他们最后得到的Q值及Q值的加权平均值也是相同的
        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        # 不同之处: 应该是用于计算交叉熵
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        param x: 输入的环境状态
        """
        # 获取总共有多少个训练样本
        batch_size = x.size()[0]
        # 归一化
        fx = x.float() / 256
        # 进行计算推理
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        # 重新reshape为(batch_size, n_classes, N_ATOMS)
        return fc_out.view(batch_size, -1, N_ATOMS)

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


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net.qvals(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def save_state_images(frame_idx, states, net, device="cpu", max_states=200):
    ofs = 0
    p = np.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_prob = net.apply_softmax(net(states_v)).data.cpu().numpy()
        batch_size, num_actions, _ = action_prob.shape
        for batch_idx in range(batch_size):
            plt.clf()
            for action_idx in range(num_actions):
                plt.subplot(num_actions, 1, action_idx+1)
                plt.bar(p, action_prob[batch_idx, action_idx], width=0.5)
            plt.savefig("states/%05d_%08d.png" % (ofs + batch_idx, frame_idx))
        ofs += batch_size
        if ofs >= max_states:
            break


def save_transition_images(batch_size, predicted, projected, next_distr, dones, rewards, save_prefix):
    for batch_idx in range(batch_size):
        is_done = dones[batch_idx]
        reward = rewards[batch_idx]
        plt.clf()
        p = np.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)
        plt.subplot(3, 1, 1)
        plt.bar(p, predicted[batch_idx], width=0.5)
        plt.title("Predicted")
        plt.subplot(3, 1, 2)
        plt.bar(p, projected[batch_idx], width=0.5)
        plt.title("Projected")
        plt.subplot(3, 1, 3)
        plt.bar(p, next_distr[batch_idx], width=0.5)
        plt.title("Next state")
        suffix = ""
        if reward != 0.0:
            suffix = suffix + "_%.0f" % reward
        if is_done:
            suffix = suffix + "_done"
        plt.savefig("%s_%02d%s.png" % (save_prefix, batch_idx, suffix))


def calc_loss(batch, net, tgt_net, gamma, device="cpu", save_prefix=None):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    # next state distribution 下一个状态的Q值分布
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    # 获取经过加权计算后的推理预测Q值的最大Q值的动作
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    # 得到下一个状态的Q值概率分布
    next_distr = tgt_net.apply_softmax(next_distr_v).data.cpu().numpy()

    # 得到下一个状态预测到的执行动作的Q值概率分布
    next_best_distr = next_distr[range(batch_size), next_actions]
    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    # 利用预测的最好的概率分布，结合实际的激励，计算出下一个状态的概率分布
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    # 得到当前的动作分布
    distr_v = net(states_v)
    state_action_values = distr_v[range(batch_size), actions_v.data]
    # 得到概率分布，计算公式在书中P147页
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    if save_prefix is not None:
        pred = F.softmax(state_action_values, dim=1).data.cpu().numpy()
        save_transition_images(batch_size, pred, proj_distr, next_best_distr, dones, rewards, save_prefix)

    # 使用kl散度来计算两个分布之间的差异
    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()


if __name__ == "__main__":
    params = {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    }

#    params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-distrib")
    # 不同 网路采用的分类式DQN网络
    net = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    # 不同之处：代理器的第一个参数对推理当前时刻的动作激励进行了装饰
    # 使用qvals进行推理的封装
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    eval_states = None
    prev_save = 0
    save_prefix = None

    best_loss = 100

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.asarray(transition.state) for transition in eval_states]
                eval_states = np.asarray(eval_states)

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])

            save_prefix = None
            # 不同之处：加了一个貌似存储训练过程中的过程图片
            if SAVE_TRANSITIONS_IMG:
                interesting = any(map(lambda s: s.last_state is None or s.reward != 0.0, batch))
                if interesting and frame_idx // 30000 > prev_save:
                    save_prefix = "images/img_%08d" % frame_idx
                    prev_save = frame_idx // 30000

            # 不同之处，使用了自定义的计算损失的函数
            loss_v = calc_loss(batch, net, tgt_net.target_model, gamma=params['gamma'],
                               device=device, save_prefix=save_prefix)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
                best_loss = common.save_model("distrib_dqn", loss_v.item(), best_loss, net.state_dict())
                common.save_model("distrib_dqn_tgt", loss_v.item(), best_loss, tgt_net.target_model.state_dict())

            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, frame_idx)

            # 不同之处，会保存训练过程中的状态图片，感觉这点和训练没什么太多关系
            if SAVE_STATES_IMG and frame_idx % 10000 == 0:
                save_state_images(frame_idx, eval_states, net, device=device)
