import gymnasium as gym

# 列出所有可用的环境
for env_spec in gym.envs.registry:
    print(env_spec)
