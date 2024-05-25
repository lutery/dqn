from QNetwork import GameNetwork
from WxJump import WxJump
from collections import deque
import numpy as np
import cv2
import os
import random
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from tensorboardX import SummaryWriter
import time
import traceback
from EvalNet import CriticNetwork
import torchvision.transforms as transforms

GAME = 'wx_jump'
ACTIONS = 1
OBS_SHAPE = 17
GAMMA = 0.99
MINI_BATCH = 10
BATCH = MINI_BATCH * 250
TRAIN_EPOCH = 32
TRAIN_BATCH = MINI_BATCH
ENTROPY_BETA = 0.01
# 增加了梯度裁剪值，防止梯度变得太大
CLIP_GRAD = 0.1

REWARD_STEPS = 1 # 计算Q值观测展开的步数
single_t = 1

STATUS_SHAPE = (14)
SAVE_MODEL_NAME = "wx_game_net.pth"
CHECK_POINT_PATH = "./checkpoints"
SAVE_MODEL_PATH = ".//save"

def float32_preprocessor(states):
    '''
    将状态矩阵转换为float32数值类型的tensor
    '''
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

def unpack_batch_a2c(batch, device="cpu"):
    """
    Convert batch into training tensors
    :param batch: 收集的游戏数据
    :param net:
    :return: states variable, actions tensor, reference values variable（游戏环境状态、执行的动作、评价的Q值）
    """
    states = [] # 每一步的游戏状态
    actions = [] # 每一步执行的动作
    rewards = [] # 每一步执行动作后获取的奖励
    for idx, exp in enumerate(batch):
        states.append(exp[0])
        actions.append(exp[1])
        rewards.append(exp[2])
    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.FloatTensor(np.array(actions)).to(device)
    ref_vals_v = torch.FloatTensor(rewards).to(device)
    return states_v, actions_v, ref_vals_v


def train_net_batch(game_net, eval_net, optimizer, eval_optimizer, device, batch_data):
    list_batch_data = list(batch_data)
    count_loss_v = 0
    count_critic_loss_v = 0
    count = 0
    EPOCH = TRAIN_EPOCH if TRAIN_EPOCH < (len(list_batch_data) // MINI_BATCH) else (len(list_batch_data) // MINI_BATCH)
    print("train epoch: ", EPOCH)
    for _ in range(EPOCH):
        batch = random.sample(list_batch_data, TRAIN_BATCH)
        loss_v, critic_loss_v = train_net(game_net, eval_net, optimizer, eval_optimizer, device, batch)
        count_loss_v += loss_v
        count_critic_loss_v += critic_loss_v
        count += 1
    
    return count_loss_v / count, count_critic_loss_v / count


def train_net(game_net, eval_net, optimizer, eval_optimizer, device, batch_data):
    states_v, actions_v, rewards_v  = unpack_batch_a2c(batch_data, device)

    eval_optimizer.zero_grad()
    values = eval_net(states_v, actions_v)
    critic_loss_v = F.mse_loss(values.squeeze(-1), rewards_v)
    critic_loss_v.backward()
    eval_optimizer.step()

    pred_action = game_net(states_v)
    adv_v = rewards_v - values.detach()
    log_prob_adv_v = log_prob_v[range(len(batch_data)), actions_v.squeeze().long()] * adv_v
    loss_policy_v = -log_prob_adv_v.mean()
    prob_v = F.softmax(logits_v, dim=1)
    entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
    loss_v = ENTROPY_BETA * entropy_loss_v + loss_policy_v
    loss_v.backward()
    # nn_utils.clip_grad_norm_(game_net.parameters(), CLIP_GRAD)
    optimizer.step()

    return loss_policy_v.item(), critic_loss_v.item()


def test_net(game_net, game, test_count, device, writer, step):
    
    reward_count = 0
    for _ in range(test_count):
        _, _, done, obs = game.reset()
        while True:
            pred_action = game_net(torch.FloatTensor(obs).unsqueeze(0).to(device))
            press_action = prob_action_2_press_action(pred_action.max(1)[1].data.cpu().numpy())

            _, r_t, done, obs = game.step_press_up(press_action[0])
            print(f"test_net obs is {obs} press_action is {press_action}, r_t is {r_t}, done is {done}")
            if done:
                break
            else:
                reward_count += r_t
        
    writer.add_scalar('test_reward', reward_count / test_count, global_step=step)


def save_checkpoints(iter, state, checkpoint_dir, keep_last=5):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{iter}.pth')
    torch.save(state, checkpoint_path)

    all_checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    if len(all_checkpoints) > keep_last:
        for old_checkpoint in all_checkpoints[:-keep_last]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
                          transforms.ToTensor(),
                          normalize,
                      ])


def train(game_net, eval_net, game, writer, optimizer, eval_optimizer, scheduler, eval_scheduler, device, D = deque(), MINI_D = deque()):
    global single_t
    done = True
    s_t = None
    best_reward = 0
    reward_count = 0
    step_count = 0
    t = time.time()

    while True:
        inference_t = time.time()
        if done:
            x_t, _, done, obs = game.reset()

        while True:
            action = game_net(torch.tensor(obs).unsqueeze(0).to(device))

            new_s_t, r_t, done, new_obs = game.step_press_up(action[0])
            if (action.item() - 0) < 0.001 and r_t > 0:
                print(f"遇到异常动作，重置游戏,动作：{action.item()}; 奖励：{r_t}")
                x_t, _, done, obs = game.reset()
            else:
                break

        inference_dt = time.time() - inference_t
        print(f"推理时间：{inference_dt:.2f}秒; 动作：{action}; 奖励：{r_t}")
        reward_count += r_t
        step_count += 1
        dt = time.time() - t
        speed_steps = step_count / dt
        print(f"TIMESTEP: {single_t}, 游戏平均速度：{speed_steps:.2f}步/秒")


        MINI_D.append([obs, action, r_t, new_obs, done, 1 if r_t > 0 else 0])
        print(f"MINI_D len is {len(MINI_D)}")
        obs = new_obs
        if len(MINI_D) < MINI_BATCH:
            continue

        D.extend(MINI_D)
        actor_loss_v, critic_loss_v= train_net_batch(game_net, eval_net, optimizer, eval_optimizer, device, D)

        writer.add_scalar('actor_loss_v', actor_loss_v, global_step=single_t)
        writer.add_scalar('critic_loss_v', critic_loss_v, global_step=single_t)
        print("============> start test net <=============")
        test_net(game_net, game, 10, device, writer, single_t)

        if single_t % 10 == 0:
            checkpoint = {
                "net": game_net.state_dict(),
                "eval_net": eval_net.state_dict(),
                "single_t": single_t,
                "optimizer": optimizer.state_dict(),
                "eval_optimizer": eval_optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "eval_scheduler": eval_scheduler.state_dict(),
                "agent_states": agent_states,
            }
            game_net.save_state(checkpoint)
            save_checkpoints(single_t, checkpoint, CHECK_POINT_PATH, keep_last=10)

        if single_t % 150 == 0:
            scheduler.step()
            eval_scheduler.step()
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

        if len(D) > BATCH:
            for _ in range(MINI_BATCH):
                top = D.popleft()
                # top[-1] -= 1
                # if top[-1] > 0:
                #     D.append(top)

        MINI_D.clear()
        game.init_state()
        
        obs = new_obs
        single_t += 1
        print("TIMESTEP", single_t, "/ ACTION", action, "/ REWARD", r_t, "/ reward_count " , reward_count , "/ BEST_REWARD %e" % best_reward)
        writer.add_scalar('REWARD', reward_count, global_step=single_t)
        reward_count = 0


def printf_except_info(e):
    msg = '{}:{} {} '.format(e.__traceback__.tb_frame.f_globals["__file__"], e.__traceback__.tb_lineno, e)
    # QMessageBox.information('错误',msg)
    print(msg)
    # write_log(msg, cache=0)
    return msg


def play_game():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = deque()
    global single_t
    game_net = GameNetwork(OBS_SHAPE=OBS_SHAPE, ACTION_SHAPE=ACTIONS).to(device)
    eval_net = CriticNetwork(OBS_SHAPE=OBS_SHAPE, ACTION_SHAPE=ACTIONS).to(device)
    game = WxJump(web_driver_type="firefox")
    writer = SummaryWriter(logdir="./simple-jump-writer", comment="-a2c-simple-jump")

    optimizer = torch.optim.Adam(game_net.parameters(), lr=1e-3)
    eval_optimizer = torch.optim.AdamW(eval_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    eval_schduler = torch.optim.lr_scheduler.ExponentialLR(eval_optimizer, gamma=0.9)

    if not os.path.exists(CHECK_POINT_PATH):
        os.mkdir(CHECK_POINT_PATH)

    if os.path.exists(CHECK_POINT_PATH) and len(os.listdir(CHECK_POINT_PATH)) > 0:
        checkpoints = sorted(os.listdir(CHECK_POINT_PATH), key=lambda x: int(x.split('_')[1].split('.')[0]))
        checkpoint = torch.load(os.path.join(CHECK_POINT_PATH, checkpoints[-1]), map_location=device)
        single_t = checkpoint['single_t']
        game_net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        eval_net.load_state_dict(checkpoint['eval_net'])
        eval_optimizer.load_state_dict(checkpoint['eval_optimizer'])
        eval_schduler.load_state_dict(checkpoint['eval_schduler'])

        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    while True:
        try:
            train(game_net, eval_net, game, , writer, optimizer, eval_optimizer,scheduler, eval_schduler, device, D)
        except Exception as e:
            printf_except_info(e)
            traceback.print_exc()
            game.init_state()


def main():
    random.seed(time.time())
    play_game()


if __name__ == "__main__":
    main()