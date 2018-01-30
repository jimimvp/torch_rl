import gym
from torch_rl.models import Reservoir
import numpy as np


env = gym.make("Pendulum-v0")
state = env.reset()
reservoir = Reservoir(0.4, 0.01, spectral_radius=0.9, network_size=200, input_size=env.observation_space.shape[0], recursive=True)

state_prev = np.zeros(200)

for i in range(100):

    state, reward, done, _ = env.step(env.action_space.sample())
    state = reservoir.forward(state)
    print(state - state_prev)
    state_prev = state