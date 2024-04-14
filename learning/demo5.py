'''
游戏已经提示过期了，需要考虑更新去除
'''
import gym
env = gym.make("Qbert-v4", render_mode="human")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
for episode in range(MAX_NUM_EPISODES):
    obs = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, truncated , info = env.step(action)
        obs = next_state
        if done is True:
            print("Episode #{} ended in {} steps.".format(episode, step+1))
            break