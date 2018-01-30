import gym
from torch_rl.models import Reservoir
import numpy as np


env = gym.make("Pendulum-v0")
state = env.reset()
reservoir = Reservoir(0.1, 0.005, spectral_radius=0.3, network_size=200, input_size=env.observation_space.shape[0], recursive=True)

state_prev = np.zeros(200)

for i in range(100):

    state, reward, done, _ = env.step(env.action_space.sample())
    state = reservoir.forward(state)
    print(state - state_prev)
    state_prev = state