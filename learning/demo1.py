import gym

env = gym.make("CartPole-v1", render_mode="human")
observation  = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    env.render()
