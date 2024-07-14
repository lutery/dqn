#!/usr/bin/env python3
'''
整改后，验证通过，可以得到让游戏成功的动作，但是不知道为啥越往后训练效果越差，最好的结果是在中间的某个时候
但是经过实际的模型测试，发现后面的模型虽然最高分比不上中间的模型，但是也是可以完成游戏的，效果还不错

1. 本游戏
2. 其他游戏

参考代码：https://github.com/louisnino/RLcode/blob/master/tutorial_TD3.py#L225
https://blog.csdn.net/hhy_csdn/article/details/107090780


td3算法对比ddpg算法的改进点：
1. 关于动作空间测探索范围不算td3的改进点，所以可以用ddpg中的OU方法进行动作空间的探索
DDPG和TD3在探索机制上没有根本性差别，两者都可以使用动作噪声（如高斯噪声或奥恩斯坦-乌伦贝克（OU）噪声）来促进探索。但是由于TD3的双重评价网络和延迟更新机制，所以可能有更好的探索机制，通常是使用简单的高斯噪声添加到探索动作上。

2.  双重Critic网络
DDPG：使用单一的Critic网络来估计状态-动作对的值函数（Q函数）。
TD3：使用两个Critic网络（双重Q学习），并在训练时取这两个网络输出的较小值来进行梯度更新。这种做法减少了值函数的过估计倾向，从而提高了学习过程的稳定性和策略的最终性能。

3. 策略更新延迟
DDPG：每次收到转换（transition）后都会更新策略网络。
TD3：策略网络的更新频率低于Critic网络，即策略网络的更新是延迟的，比如每两次Critic更新才进行一次策略更新。这种延迟更新策略进一步防止了因Critic估计误差大而导致的策略性能下降。

4. 目标策略平滑
DDPG：在使用目标策略网络生成动作来计算目标Q值时，直接采用目标策略网络的输出。
TD3：在目标策略网络生成的动作上添加噪声（通常是限制在一个小范围内的高斯噪声），以进行目标策略平滑。这种方法通过平滑目标策略的输出来降低Q值估计的方差，进一步增强算法的稳定性。

这里说的平滑是指在更新评价网络时，通过给目标动作网络添加噪声，使得评价网络的能够考虑到范围更大的动作空间内的Q值，从而使得网络减少过拟合增强鲁棒性，不再受微小动作的改变而产生较大的Q值变化（因为它将预测动作的领域内都一起考虑在内得到相类似的Q值）

5. 目标网络的同步时机是在动作网络更新时，将所有目标网络进行同步

6 超参数选择不那么敏感，可能是因为有两个Q网络并且在训练时取最小值，所以对于Q值的估计更加稳定


7. 将奖励归一化应该也不是区别之一，都可以采用


TD3 更加适合高维的动作空间会比ddpg有更好的表现
'''

import os
import ptan
import time
import gymnasium as gym
import argparse
from tensorboardX import SummaryWriter
import numpy as np

from lib import model, common

import torch
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F


ENV_ID = "Pendulum-v1"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000 # 重放缓冲区长度，这么长是为了提高稳定性
REPLAY_INITIAL = 10000 # 重放缓冲区初始化大小

TEST_ITERS = 1000

eval_noise_scale = 0.5
explore_noise_scale = 1.0
reward_scale = 1.
policy_target_update_interval = 3

class TD3():
    def __init__(self, obs_space, action_space, action_range, device="cpu"):
        self.device = device
        self.act_net = model.TD3Actor(obs_space.shape[0], action_space.shape[0], action_range).to(device)
        self.crt1_net = model.DDPGCritic(obs_space.shape[0], action_space.shape[0]).to(device)
        self.crt2_net = model.DDPGCritic(obs_space.shape[0], action_space.shape[0]).to(device)
        self.target_crt1_net = ptan.agent.TargetNet(self.crt1_net)
        self.target_crt2_net = ptan.agent.TargetNet(self.crt2_net)
        self.target_act_net = ptan.agent.TargetNet(self.act_net)

        self.act_opt = optim.Adam(self.act_net.parameters(), lr=LEARNING_RATE)
        self.crt1_opt = optim.Adam(self.crt1_net.parameters(), lr=LEARNING_RATE)
        self.crt2_opt = optim.Adam(self.crt2_net.parameters(), lr=LEARNING_RATE)

        self.action_range = action_range
        self.device = device
        self.update_cnt = 0


    def update(self):
        self.update_cnt += 1


    def evaluate(self, states, eval_noise_scale):
        states_v = states
        mu_v = self.target_act_net.target_model(states_v)
        actions = mu_v.data.cpu().numpy()
        normal = dist.Normal(0, 1)
        eval_noise_clip = 1 * eval_noise_scale
        noise = normal.sample(actions.shape) * eval_noise_scale
        noise = np.clip(noise, -eval_noise_clip, eval_noise_clip)
        actions = np.add(actions, noise)

        return actions.clip(-1 * self.action_range, 1 * self.action_range).to(self.device)


class TD3Agent(ptan.agent.BaseAgent):
    def __init__(self, net, explore_noise_scale, action_range=1., step=0, device="cpu"):
        '''

        :param net:
        :param explore_noise_scale:
        :param action_range: todo 这里的动作范围是否需要考虑到不同的游戏
        '''
        assert isinstance(net, model.TD3Actor)
        self.net = net
        self.explore_noise_scale = explore_noise_scale
        self.action_range = action_range
        self.step = step
        self.device = device
        self.rng = np.random.default_rng(int(time.time()))

    def __call__(self, states, agent_states):
        # todo 这边用原来的OU方法是否也可以？
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        self.step += 1
        # todo 500是否可以不用
        if self.step < 500:
            actions = self.rng.uniform(-1, 1, actions.shape)
            actions = self.action_range * actions
        else:
            noise = self.rng.normal(0, 1, actions.shape) * self.explore_noise_scale
            actions = actions + noise

        return actions.clip(-1 * self.action_range, 1 * self.action_range), agent_states



def test_net(net, env, count=10, device="cpu"):
    '''
    count: 执行游戏的次数（每次都是执行到游戏结束）

    return: （平均奖励，平均步数）
    '''
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            # 根据环境预测动作
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            # 执行动作
            obs, reward, done, truncated, _ = env.step(action)
            rewards += reward
            steps += 1
            if done or truncated:
                break
    return rewards / count, steps / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "td3-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID, g=9.81)
    test_env = gym.make(ENV_ID, g=9.81)

    action_range = float(env.action_space.high[0])

    # 构建动作网络和评价网络
    td3 = TD3(env.observation_space, env.action_space, action_range=action_range, device=device)
    print(td3.act_net)
    print(td3.crt1_net)
    print(td3.crt2_net)

    agent = TD3Agent(td3.act_net, action_range=action_range, explore_noise_scale=explore_noise_scale, device=device)
    if (os.path.exists(os.path.join(save_path, "checkpoint_0.dat"))):
        checkpoint = torch.load(os.path.join(save_path, "checkpoint_0.dat"))
        td3.act_net.load_state_dict(checkpoint["act_net"])
        td3.crt1_net.load_state_dict(checkpoint["crt1_net"])
        td3.crt2_net.load_state_dict(checkpoint["crt2_net"])
        td3.target_act_net.target_model.load_state_dict(checkpoint["target_act_net"])
        td3.target_crt1_net.target_model.load_state_dict(checkpoint["target_crt1_net"])
        td3.target_crt2_net.target_model.load_state_dict(checkpoint["target_crt2_net"])
        td3.update_cnt = checkpoint["update_cnt"]
        agent.step = checkpoint["step"]
        print("加载模型成功")

    writer = SummaryWriter(comment="-td3_" + args.name)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)

    frame_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                # 从经验缓冲区执行一轮游戏或者执行游戏过程中采集到指定长度的游戏数据
                buffer.populate(1)
                # 提取目前位置所有的记录
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    # 记录当前的训练进度并判断是否达到了奖励目标
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                td3.update()
                # 从缓冲区里面采样数据
                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)

                # 归一化奖励
                rewards_v = reward_scale * (rewards_v - rewards_v.mean()) / (rewards_v.std() + 1e-6)
                # 使用目标动作预测网路，根据下一个状态预测执行的动作
                last_act_v = td3.evaluate(last_states_v, eval_noise_scale)
                # 使用目标评测网络，根据下一个状态和下一个状态将要执行的动作得到下一个状态的评价Q值
                q_last_v = torch.min(td3.target_crt1_net.target_model(last_states_v, last_act_v), td3.target_crt2_net.target_model(last_states_v, last_act_v))
                # 如果是结束状态则将奖励置为0
                q_last_v[dones_mask.bool()] = 0.0
                # 计算Q值 bellman公式
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                # 计算预测的当前Q值和Bellman计算的到的Q值之间的差异
                # 并更新梯度 这里的方式就和之前的Q值单元的一致
                # train critic
                td3.crt1_opt.zero_grad()
                # 根据状态和动作，得到评价，这里是根据实际游戏的状态和动作获取评价
                q_v = td3.crt1_net(states_v, actions_v)
                critic_1_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_1_loss_v.backward()
                td3.crt1_opt.step()
                tb_tracker.track("loss_critic_1", critic_1_loss_v, frame_idx)

                td3.crt2_opt.zero_grad()
                q_v = td3.crt2_net(states_v, actions_v)
                critic_2_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_2_loss_v.backward()
                td3.crt2_opt.step()
                tb_tracker.track("loss_critic_2", critic_2_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # 预测动作
                if td3.update_cnt % policy_target_update_interval == 0:
                    # train actor
                    td3.act_opt.zero_grad()
                    cur_actions_v = td3.act_net(states_v)
                    # 根据状态和预测的动作计算Q值的负值
                    # 这里是根据网络预测的动作和实际的状态获取评价
                    # 在评价前取负号，就是简单粗暴的取最小值，从而达到最大值Q值的目的
                    # 由于这里评价网络是固定是，所以最大化Q值，只有更新预测的动作，使得
                    # 评价Q值达到最大值的目的
                    actor_loss_v = -td3.crt1_net(states_v, cur_actions_v)
                    actor_loss_v = actor_loss_v.mean()
                    actor_loss_v.backward()
                    td3.act_opt.step()
                    tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                    # 将训练网路同步到目标网络上，但是这里是每次都同步，与之前每隔n步同步一次不同
                    # 这里之所以这样做，是根据测试可知，每次都同步，并使用较小的权重进行同步
                    # 缓存的同步效果更好，并且能够保持平滑的更新
                    td3.target_act_net.alpha_sync(alpha=1 - 1e-3)
                    td3.target_crt1_net.alpha_sync(alpha=1 - 1e-3)
                    td3.target_crt2_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    # 测试并保存最好测试结果的庶数据
                    ts = time.time()
                    rewards, steps = test_net(td3.act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    checkpoint = {
                        "act_net": td3.act_net.state_dict(),
                        "crt1_net": td3.crt1_net.state_dict(),
                        "crt2_net": td3.crt2_net.state_dict(),
                        "update_cnt": td3.update_cnt,
                        "frame_idx": frame_idx,
                        "best_reward": best_reward,
                        "target_crt1_net": td3.target_crt1_net.target_model.state_dict(),
                        "target_crt2_net": td3.target_crt2_net.target_model.state_dict(),
                        "target_act_net": td3.target_act_net.target_model.state_dict(),
                        "step": agent.step
                    }
                    torch.save(checkpoint, os.path.join(save_path, "checkpoint_%d.dat" % (frame_idx % 10)))
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)

                            torch.save(checkpoint, fname)
                        best_reward = rewards
    pass
