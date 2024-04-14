import gym

# 列出所有可用的环境
for env in gym.envs.registry.values():
    print(env.id)
