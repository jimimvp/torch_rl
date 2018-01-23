import gym
import torch_rl.envs

bandit_env = gym.make('BanditsX2-v0')
state = bandit_env.reset()

print(state)