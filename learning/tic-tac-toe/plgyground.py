import gymnasium as gym

env = gym.make('gym_tictactoe:tictactoe-v0')
env.reset()

env.render()
# | | | |
# | | | |
# | | | |