import gym

# Initialize the environment
env = gym.make('ALE/Alien-v5', render_mode="human")

# Reset the environment to get the initial state
state = env.reset()
total_reward = 0
# Run a loop to play the game
for _ in range(10000):
    env.render()  # Render the environment

    # Take a random action
    action = env.action_space.sample()

    # Get the next state, reward, done flag, and info from the environment
    state, reward, done, trunc, info = env.step(action)
    total_reward += reward
    if reward != 0:
        print("reward: ", reward)
        print("info: ", info)

    # If done, reset the environment
    if done or trunc:
        state = env.reset()

# Close the environment
env.close()