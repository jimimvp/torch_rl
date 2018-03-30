"""
    Wrappers for logging, useful when environments undergo
    certain transformations to keep track of different rewards
    for example and other environment info
"""


from torch_rl.utils import logger
import gym
from collections import deque
import numpy as np

class EnvLogger(gym.Wrapper):

    def __init__(self, env, level=logger.INFO, track_attrs=[]):
        super(EnvLogger, self).__init__(env)
        self.env = env
        self.rdeque = deque(maxlen=100)
        self.level = level
        self.obs, self.reward, self.info, self.done = None, None, None, False
        self.episode_reward = 0
        self.track_attrs = track_attrs
    
    def step(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        self.episode_reward += self.reward
        self.log()
        if self.done:
            self.rdeque.append(self.episode_reward)
            self.episode_reward = 0
        logger.logkv('avg_episode_reward', np.mean(self.rdeque))

        return  self.obs, self.reward, self.done, self.info

    def log(self):

        for attr in self.track_attrs:
            val = getattr(self, attr)
            logger.logkv(attr, val, self.level)

    def reset(self):
        self.episode_reward = 0
        return self.env.reset()


    def close(self):
        self.env.close()



def _test():

    env = EnvLogger(gym.make('Pendulum-v0'))
    obs = env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
        logger.dumpkvs()



if __name__ == '__main__':

    _test()