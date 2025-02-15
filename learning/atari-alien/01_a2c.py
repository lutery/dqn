#!/usr/bin/env python3
'''
未验证
'''
import os
import ptan
import time
import argparse
from tensorboardX import SummaryWriter
import gymnasium as gym
import torch
import torch.optim as optim

from lib import common


LEARNING_RATE = 1e-4
TEST_EVERY_BATCH = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--steps", type=int, default=None, help="Limit of training steps, default=disabled")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", "01_a2c_" + args.name)
    os.makedirs(saves_path, exist_ok=True)

    envs = [common.make_env() for _ in range(common.NUM_ENVS)]

    test_env = common.make_env()
    writer = SummaryWriter(comment="-01_a2c_" + args.name)

    # 创建网络
    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)
    if os.path.exists(os.path.join(saves_path, "net.dat")):
        net.load_state_dict(torch.load(os.path.join(saves_path, "net.dat")))
        print("加载模型成功")

    # todo 这里为什么采用了RMSprop优化器
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    step_idx = 0
    total_steps = 0
    best_reward = None
    ts_start = time.time()
    best_test_reward = None
    with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
        # 遍历所有的游戏环境，使用现有的net，预测出REWARD_STEPS的动作以及动作对应的奖励、状态价值
        for mb_obs, mb_rewards, mb_actions, mb_values, _, done_rewards, done_steps in \
                common.iterate_batches(envs, net, device=device):
            if len(done_rewards) > 0:
                # 如果有收集到数据，那么计算公共执行的步数
                total_steps += sum(done_steps)
                # 计算执行步数的速度
                speed = total_steps / (time.time() - ts_start)
                # 保存记录最好的回报奖励
                if best_reward is None:
                    best_reward = done_rewards.max()
                elif best_reward < done_rewards.max():
                    best_reward = done_rewards.max()
                # 记录到tensorboard中
                tb_tracker.track("total_reward_max", best_reward, step_idx)
                tb_tracker.track("total_reward", done_rewards, step_idx)
                tb_tracker.track("total_steps", done_steps, step_idx)
                print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f, speed=%.2f" % (
                    step_idx, len(done_rewards), done_rewards.mean(), best_reward, speed))

            # 训练网络
            common.train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values,
                             optimizer, tb_tracker, step_idx, device=device)
            step_idx += 1
            # 判断步数结束标识
            if args.steps is not None and args.steps < step_idx:
                break

            if step_idx % TEST_EVERY_BATCH == 0:
                # 测试网络
                test_reward, test_steps = common.test_model(test_env, net, device=device)
                # 记录测试结果
                writer.add_scalar("test_reward", test_reward, step_idx)
                writer.add_scalar("test_steps", test_steps, step_idx)
                if best_test_reward is None or best_test_reward < test_reward:
                    # 根据测试结果，保存最好的模型
                    if best_test_reward is not None:
                        fname = os.path.join(saves_path, "best_%08.3f_%d.dat" % (test_reward, step_idx))
                        torch.save(net.state_dict(), fname)
                    best_test_reward = test_reward
                print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                    step_idx, test_reward, test_steps, best_test_reward))

                torch.save(net.state_dict(), os.path.join(saves_path, "net.dat"))
