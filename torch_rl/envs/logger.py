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

    def __init__(self, env, level=logger.INFO, track_attrs=[], pref=''):
        super(EnvLogger, self).__init__(env)
        self.env = env
        self.rdeque = deque(maxlen=100)
        self.level = level
        self.obs, self.reward, self.info, self.done = None, None, None, False
        self.episode_reward = 0
        self.track_attrs = track_attrs
        self.steps = 0
        self.pref = pref
    
    def step(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        self.episode_reward += self.reward
        self.steps+=1
        if self.done:
            self.rdeque.append(self.episode_reward)
            logger.logkv(self.pref+'eaccreward', self.episode_reward)
            logger.logkv(self.pref+'avgereward', np.mean(self.rdeque))
            logger.logkv(self.pref+'esteps', self.steps)
            self.episode_reward = 0
            self.steps = 0

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



class GoalEnvLogger(gym.Wrapper):
    """
    Wrapper that logs goal distance and relevenat information for goal environment
    """

    def __init__(self, env, target_indices, curr_indices, precision=1e-2):
        super(GoalEnvLogger, self).__init__(env)
        assert len(target_indices) == len(curr_indices)

        self.target_indices = target_indices
        self.curr_indices = curr_indices
        self.precision = precision
        self.episode_success = False
        self.successes = 0.
        self.episodes = 0.

    def step(self, action):
        obs, reward, done, inf = self.env.step(action)
        success = np.allclose(obs[self.curr_indices], obs[self.target_indices], atol=self.precision)
        if not self.episode_success and success:
            self.successes+=1.
            self.episode_success = True
        if done:
            self.episodes += 1.
            self.episode_success = False
            logger.logkv('success_rate', self.successes / self.episodes)
            logger.logkv('errtarget', np.sum(np.abs(obs[self.target_indices] - obs[self.curr_indices])))
            #logger.logkv('sparse_reward', np.sum(np.abs(obs[self.target_indices] - obs[self.curr_indices])))
        return obs, reward, done, inf


def _test():

    env = EnvLogger(gym.make('Pendulum-v0'))
    obs = env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
        logger.dumpkvs()



if __name__ == '__main__':

    _test()