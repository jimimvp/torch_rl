import gym
from torch_rl.models import Reservoir
import numpy as np

class ReservoirObservationWrapper(gym.Wrapper):

    """
        Class wraps an environment and applies a reservoir
        transformation to the observation.
    """

    def __init__(self, env, reservoir):
        super(ReservoirObservationWrapper, self).__init__(env)
        self.reservoir = reservoir



    def step(self, action):
        
        obs, r, done, i = self.env.step(action)
        self.obs = obs
        spiking_obs = self._obsfilt(obs)
        return spiking_obs, r, done, i

    def _obsfilt(self, obs):
        spiking_obs = self.reservoir.forward(obs)
        return spiking_obs.reshape(-1)

    def reset(self):
        obs = self.env.reset()
        self.reservoir.reset()
        spiking_obs = self.reservoir.forward(obs)
        return np.asarray(spiking_obs, dtype=np.float32).flatten()




if __name__ == '__main__':
    import gym
    import torch_rl
    env = gym.make('Pendulum-v0')
    reservoir = Reservoir(dt=0.03, sim_dt=0.01, input_size=env.observation_space.shape[0], network_size=800, recursive=False,
                 spectral_radius=1.,noise=False, synapse_filter=15e-4)
    env = ReservoirObservationWrapper(env, reservoir)

    obs = env.reset()
    for i in range(100):

        print(env.step(env.action_space.sample()))