import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import StepAPICompatibility, TimeLimit


def gymnasium_reset(self, **kwargs):
    return self.env.reset(), {}

def make_super_mario_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    steps = env._max_episode_steps  # get the original max_episode_steps count
    env = JoypadSpace(env.env, SIMPLE_MOVEMENT)  # set the joypad wrapper
    # overwrite the old reset to accept `seeds` and `options` args
    env.reset = gymnasium_reset.__get__(env, JoypadSpace)

    # set TimeLimit back
    env = TimeLimit(StepAPICompatibility(env, output_truncation_bool=True), max_episode_steps=steps)

    return env

env = make_super_mario_env()

done = True
for step in range(5000):
    if done:
        state, info = env.reset()
    state, reward, done, trunc, info = env.step(env.action_space.sample())
    env.render()

env.close()