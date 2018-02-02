from torch_rl.memory import HindsightMemory
import gym
import numpy as np
from tqdm import tqdm

memory = HindsightMemory(1000, window_length=1)
env = gym.make("Pendulum-v0")
env.reset()
num_transitions = 4
hind_size = 8


for i in range(1000):
    # Warmup
    action = env.action_space.sample()
    goal = env.observation_space.sample()
    observation, reward, done, _ = env.step(action)
    reward = 1 if np.sum(np.abs(observation-goal)**2) < 1. else -1
    if done:
        env.reset()
    memory.append(observation, env.observation_space.sample(), action=action, terminal=done, reward=reward)


for i in tqdm(range(100000)):
    action = env.action_space.sample()
    goal = env.observation_space.sample()
    observation, reward, done, _ = env.step(action)
    reward = 1 if np.sum(np.abs(observation - goal) ** 2) < 1. else -1

    memory.append(observation, env.observation_space.sample(), action=action, terminal=done, reward=reward)

    state0_batch, goal_batch, action_batch, reward_batch, \
    state1_batch,terminal1_batch = memory.sample_and_split(num_transitions*hind_size + num_transitions)

    if done:
        env.reset()

    assert len(state0_batch) == num_transitions*hind_size+num_transitions






