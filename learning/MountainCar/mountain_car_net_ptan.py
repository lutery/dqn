'''
验证未通过
书中到3万多步奖励才上升
'''
import ptan
import gymnasium as gym
from gymnasium import Wrapper
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from lib import dqn_model, common


# 测试多少次游戏轮数
MAX_NUM_EPISODES = 50000
# 每轮游戏的最大步数
STEPS_PER_EPISODE = 200
# 游戏动作选择器（即采用随机值还是学习到的动作进行动作执行）
EPSILON_MIN = 0.005
# 总共的游戏步数
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
# epsilon衰减值，这个值根据最大的游戏步数来计算，表示每个步数epsilon的衰减值
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
# todo
ALPHA = 0.05
# todo
GAMMA = 0.98
# todo
NUM_DISCRETE_BINS = 30

REWARD_STEPS_DEFAULT = 2

class CustomEndConditionWrapper(Wrapper):
    def __init__(self, env):
        super(CustomEndConditionWrapper, self).__init__(env)
        self.step_count = 0

    def step(self, action):
        self.step_count += 1
        observation, reward, done, trun, info = self.env.step(action)
        # 自定义结束条件：例如位置大于0.45或小于-0.45
        if observation[0] >= 0.5 or self.step_count >= STEPS_PER_EPISODE:
            self.step_count = 0
            done = True
        return observation, reward, done, trun, info


if __name__ == "__main__":
#    params['epsilon_frames'] = 200000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建游戏环境
    env = gym.make("MountainCar-v0")
    # env = CustomEndConditionWrapper(env)

    writer = SummaryWriter(comment="-mountain-car-ptan-n-steps")
    net = dqn_model.DQNMountainCar(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    epsilon_tracker = common.EpsilonTracker(selector, 1.0, 0.02, 10**5)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=REWARD_STEPS_DEFAULT)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=10**5)

    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    frame_idx = 0
    best_loss = 100

    with common.RewardTracker(writer, stop_reward=-50) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < 10000:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(32)
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=0.99*REWARD_STEPS_DEFAULT, device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % 1000 == 0:
                tgt_net.sync()
                best_loss = common.save_model("mountain_car_dqn_n_steps", loss_v, best_loss, net)
                common.save_model("mountain_car_tgt_dqn_n_steps", loss_v, best_loss, net)
