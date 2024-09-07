import gym

# Initialize the environment
env = gym.make('ALE/Adventure-v5', render_mode="human")

# Reset the environment to get the initial state
state = env.reset()

# Run a loop to play the game
for _ in range(1000):
    env.render()  # Render the environment

    # Take a random action
    action = env.action_space.sample()

    # Get the next state, reward, done flag, and info from the environment
    state, reward, done, trunc, info = env.step(action)

    # If done, reset the environment
    if done or trunc:
        state = env.reset()

# Close the environment
env.close()