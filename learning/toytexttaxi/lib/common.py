import gymnasium as gym
import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

DEFAULT_SEED = 20 # 环境默认的随机种子

NUM_ENVS = 16 # 创建环境的数量
GAMMA = 0.99
REWARD_STEPS = 5 # todo 作用 目前看起来像是记录环境交互结果的步数
ENTROPY_BETA = 0.01
VALUE_LOSS_COEF = 0.5
BATCH_SIZE = REWARD_STEPS * 16
CLIP_GRAD = 0.5


def make_env():
    return gym.make("Taxi-v3")


class AtariA2C(nn.Module):
    '''
    创建A2C网络，用来预测执行动作的概率和执行动作后的直接回报
    '''
    def __init__(self, n_obs, n_actions):
        super(AtariA2C, self).__init__()


        # 特征提取后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        # ## 预测执行动作的大小，后续会将大小转换该概率
        self.policy = nn.Linear(512, n_actions)
        # ## 根据环境特征的提取预测执行动作所能获取的直接回报，也就是环境执行游戏一步后得到的reward，这里就是和之前网络的不同指出，之前都是预测Q值，这里预测的直接回报
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        '''
        return todo (预测的动作， )
        '''
        # 归一化
        fx = x.float()
        # 将卷积输出展平
        fc_out = self.fc(fx)
        return self.policy(fc_out), self.value(fc_out)



def discount_with_dones(rewards, dones, gamma):
    '''
    param rewards: 游戏每一步的回报
    param dones: 游戏每一步是否结束
    param gamma: 折扣因子

    return discounted: 游戏每一步的回报。采用类似bellman的计算公式
    '''
    # 保存游戏最近REWARD_STEPS每一步中的回报
    discounted = []
    r = 0
    # 逆序遍历游戏每一步的回报和是否结束
    for reward, done in zip(rewards[::-1], dones[::-1]):
        # 计算公式：r(t+1) = r(t+1) + gamma * r'(1.0 - done（False=0， True=1）)'
        # 如果游戏已经结束，那么r就等于结束的时候拿到的回报
        # 如果游戏没有结束，那么r等于当前的回报加上下一步的回报乘以折扣因子
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    # 有序计算的时候是从后往前计算的，所以这里需要将计算结果逆序
    return discounted[::-1]


class ProbabilityActionSelector(ptan.actions.ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    概率动作选择器
    """
    def __call__(self, probs_and_info):
        probs, info = probs_and_info
        assert isinstance(probs, np.ndarray)
        actions = []
        valid_actions = [np.where(i['action_mask']==1)[0] for i in info]

        for prob, valid in zip(probs, valid_actions):
            mask = np.zeros_like(prob)
            mask[valid] = 1

            masked_prob = mask * prob
            sum_mask_prob = masked_prob.sum()
            if sum_mask_prob == 0:
                masked_prob[valid] = 1 / len(valid)
            else:
                masked_prob = masked_prob / sum_mask_prob

            action = np.random.choice(len(prob), p=masked_prob)
            actions.append(action)

        return np.array(actions)


def iterate_batches(envs, net, device="cpu"):
    '''
    param envs: 环境列表
    param net: I2A网络

    return 采集到的环境观察数据, 环境得到的Q值回报, 执行的动作, 环境得到的回报奖励,执行动作的概率分布, 记录每一次游戏结束时的总回报, 记录每一次游戏结束时的总步数
    '''
    # 动作维度，动作数量
    n_actions = envs[0].action_space.n
    # 概率动作选择器
    act_selector = ProbabilityActionSelector()
    # 重置环境,这里存储着重置后，获取到的环境状态，由于在配置环境是使用了FrameWrapper，所以这里获取的帧数是FRAMES_COUNT（2），所以这里的维度是（NUM_ENVS, FRAMES_COUNT, 84, 84）
    obs = [e.reset() for e in envs]
    info = [o[1] for o in obs]
    obs = [[o[0]] for o in obs]
    # 记录每个环境是否已经结束
    batch_dones = [[False] for _ in range(NUM_ENVS)]
    # 记录每个环境的总回报
    total_reward = [0.0] * NUM_ENVS
    # 记录每个环境的总步数
    total_steps = [0] * NUM_ENVS
    # todo 作用
    # (NUM_ENVS, REWARD_STEPS) + IMG_SHAPE得到的最终维度是（NUM_ENVS, REWARD_STEPS, FRAMES_COUNT, 84, 84）
    mb_obs = np.zeros((NUM_ENVS, REWARD_STEPS, 1), dtype=np.uint8)
    # 记录游戏每进行一步得到的奖励，总共会记录REWARD_STEPS步
    # 在REWARD_STEPS步数结束后，会将其转换为类似Q值的回报
    mb_rewards = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    # 记录网络预测的环境执行动作后的直接回报奖励
    mb_values = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_actions = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.int32)
    mb_probs = np.zeros((NUM_ENVS, REWARD_STEPS, n_actions), dtype=np.float32)
    # todo

    while True:
        # 获取游戏最后的一次结束状态标识
        # 在进行REWARD_STEPS步探索时，会记录每执行一步后，游戏的状态是否结束
        # 总共记录REWARD_STEPS步，这里取最后一步的结束状态
        batch_dones = [[dones[-1]] for dones in batch_dones]
        # 记录游戏结束的环境的总回报和总步数，不区分是哪个环境产生的回报和步数
        done_rewards = []
        done_steps = []
        for n in range(REWARD_STEPS):
            # 将obslist转换为np array
            obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
            # 将拿到的环境特征放置到mb_obs中
            mb_obs[:, n] = obs_v.data.cpu().numpy()
            # 将环境特征放入网络获取
            # todo logits_v是什么,values_v是什么
            logits_v, values_v = net(obs_v)
            # 使用softmax宇哥每个执行动作的概率
            probs_v = F.softmax(logits_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            # 使用概率动作选择其，选择需要执行的动作
            actions = act_selector((probs, info))
            # 保存每一步预测的动作概率
            mb_probs[:, n] = probs
            # 保存每一步选择动作
            mb_actions[:, n] = actions
            # todo 作用
            mb_values[:, n] = values_v.squeeze().data.cpu().numpy()
            # 遍历每一个环境，执行一步，取得环境的状态，奖励，是否结束，其他信息
            for e_idx, e in enumerate(envs):
                # 将选择执行的动作传入环境，获取环境的状态，奖励，是否结束，其他信息
                o, r, done, trunc, _ = e.step(actions[e_idx])
                # 记录每个环境的总回报
                total_reward[e_idx] += r
                # 记录环境的向前一步
                total_steps[e_idx] += 1
                if done:
                    # 如果游戏结束，则重置环境
                    o = e.reset()
                    # 记录游戏结束的环境的总回报
                    done_rewards.append(total_reward[e_idx])
                    done_steps.append(total_steps[e_idx])
                    # 重置该环境的总回报和总步数
                    total_reward[e_idx] = 0.0
                    total_steps[e_idx] = 0
                # 更新最新的游戏观察数据到缓存中
                obs[e_idx] = [o]
                # 将当前游戏的回报保存到缓存中
                mb_rewards[e_idx, n] = r
                # 记录每一个游戏环境执行当前的一步后，是否结束的标识
                batch_dones[e_idx].append(done)
        # obtain values for the last observation
        # 将最新的游戏状体转换为矩阵
        obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
        # 将最新的游戏状态放入网络中获取 todo value
        _, values_v = net(obs_v)
        # 移除大小为1 的维度,也就是将最里面的维度移除，得到一个list
        # 这里记录的是游戏进行了REWARD_STEPS步，这里之所以还要进行一次最后一步
        # 是为保证在计算游戏的bellman公式时，最后一步的回报也能够被计算到，否则没有结束的游戏
        # 最后一步的下一步游戏回报就是0，这样计算出来的回报就会偏小
        values_last = values_v.squeeze().data.cpu().numpy()

        # 遍历之前每一个环境收集的每一步的回报、每一步后游戏是否结束、todo 最后一步的回报
        # 这里就是将每一步收集到的回报转换为类似Q值，计算公式类似bellman公式
        for e_idx, (rewards, dones, value) in enumerate(zip(mb_rewards, batch_dones, values_last)):
            # 当前观测环境的每一步的回报
            rewards = rewards.tolist()
            if not dones[-1]:
                # 计算最后一个如果不是结束状态的的总回报
                # 这里done的size会比rewards多一个，是因为batch_dones初始化的时候，每个环境都会初始化一个done标识，所以这里需要去掉最后一个
                rewards = discount_with_dones(rewards + [value], dones[1:] + [False], GAMMA)[:-1]
            else:
                # 计算最后一个如果是结束状态的的总回报，采用的计算公式一致
                # 因为最后一个状态是结束状态，所以没有下一次回报，所以就不需要+ [value]了
                rewards = discount_with_dones(rewards, dones[1:], GAMMA)
            # 将转换后的回报保存到mb_rewards中
            mb_rewards[e_idx] = rewards

        # 有所有缓存中有REWARD_STEPS维度全部展        平
        out_mb_obs = mb_obs.reshape((-1,1))
        out_mb_rewards = mb_rewards.flatten()
        out_mb_actions = mb_actions.flatten()
        out_mb_values = mb_values.flatten()
        out_mb_probs = mb_probs.flatten()
        # 用协程的方式返回数据
        yield out_mb_obs, out_mb_rewards, out_mb_actions, out_mb_values, out_mb_probs, \
              np.array(done_rewards), np.array(done_steps)


def train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values, optimizer, tb_tracker, step_idx, device="cpu"):
    '''
    param net: I2A网络
    param mb_obs: 游戏每一步的状态
    param mb_rewards: 游戏每一步的回报,这里不是直接回报，而是转换后的Q值
    param mb_actions: 游戏每一步的动作
    param mb_values: 游戏每一步的状态价值，是网络预测的直接回报
    param optimizer: 优化器
    param tb_tracker: tensorboard记录器
    param step_idx: 当前训练轮数

    '''

    # 重置梯度
    optimizer.zero_grad()
    # 记录优势，也就是Q值是否大于状态价值，如果大于，那么优势就是正数，否则就是负数
    # todo 这里的mb_adv的最小值是否是0？
    # todo mb_values是游戏预测的直接回报，那么这个公式实在做什么？
    mb_adv = mb_rewards - mb_values
    adv_v = torch.FloatTensor(mb_adv).to(device)
    obs_v = torch.FloatTensor(mb_obs).to(device)
    rewards_v = torch.FloatTensor(mb_rewards).to(device)
    actions_t = torch.LongTensor(mb_actions).to(device)
    # 将状态传入net，得到预测的动作大小和状态回报
    logits_v, values_v = net(obs_v)
    # 计算动作的概率
    log_prob_v = F.log_softmax(logits_v, dim=1)
    # 和之前的策略梯度一样，使用优势基线和执行动作的概率相结合，计算损失
    # 如果优势是正数，那么会加大这个动作，如果优势是负数，那么会减小这个动作
    # todo 但是这里会出现负值吗？
    log_prob_actions_v = adv_v * log_prob_v[range(len(mb_actions)), actions_t]

    # 因为要计算最大值，所以这里需要取负数，计算最小值
    # 因为所有的优化器，都基本是最小值优化
    loss_policy_v = -log_prob_actions_v.mean()
    # 计算每一个预测出来的环境回报和实际的环境回报之间的均方误差
    # 使得网路预测的回报能够接近世界回报
    loss_value_v = F.mse_loss(values_v.squeeze(-1), rewards_v)

    # 预测的动作概率，这里是计算信息熵，也就是计算动作的不确定性
    # 这里信息熵取了符号，原本信息熵的计算公式前面是由负号的
    # 所以取最小值也就导致了信息熵最大，目的是让网络能够进行探索
    prob_v = F.softmax(logits_v, dim=1)
    entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()
    # 得到总损失
    loss_v = ENTROPY_BETA * entropy_loss_v + VALUE_LOSS_COEF * loss_value_v + loss_policy_v
    # 计算梯度
    loss_v.backward()
    # 限制梯度的大小，防止梯度爆炸
    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
    # 优化
    optimizer.step()

    #记录训练中的数值
    tb_tracker.track("advantage", mb_adv, step_idx)
    tb_tracker.track("values", values_v, step_idx)
    tb_tracker.track("batch_rewards", rewards_v, step_idx)
    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
    tb_tracker.track("loss_value", loss_value_v, step_idx)
    tb_tracker.track("loss_total", loss_v, step_idx)
    return obs_v


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    这个预处理器的方法就是将list转换为矩阵的形式
    如果state是一维的，那么就将其转换为[1, D]的形式
    如果state是多维的，那么就将其转换为[N, E, D]的形式
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)

class PolicyAgent(ptan.agent.BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    def __init__(self, model, action_selector=ptan.actions.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        '''
            model: 策略动作推理网络
            preprocessor: 将计算的结果转换的数据类型，比如转换为float32
            apply_softmax: 使用对model的计算结果使用softmax计算结果
        '''
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states 在本代理器中，agent_states没有参与计算，仅仅是保证其维度和states一样
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        # 如果定义了预处理器，则进行预处理擦欧总
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        # 计算动作概率
        probs_v = self.model(states.unsqueeze(0))
        # 如果需要使用softmax计算
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        # 将网络得到的动作概率丢给动作选择器进行选择需要执行的动作
        actions = self.action_selector(probs)
        return np.array(actions), agent_states


def test_model(env, net, rounds=3, device="cpu"):
    '''
    param env: 测试游戏环境
    param net: I2A网络
    param rounds: 测试轮数（直到结束的轮数）
    '''

    total_reward = 0.0
    total_steps = 0
    agent = PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)

    for _ in range(rounds):
        obs, info = env.reset()
        while True:
            action = agent([obs])[0][0]
            obs, r, done, trunc, _ = env.step(action)
            total_reward += r
            total_steps += 1
            if done or trunc:
                break

    # 平均每一轮的回报和步数
    return total_reward / rounds, total_steps / rounds
