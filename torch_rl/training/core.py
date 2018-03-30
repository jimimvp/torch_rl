import torch as tor
from torch_rl.utils import *
from torch_rl.utils import logger
from collections import deque
import time

def random_process_action_choice(random_process):
    def func(actor_output, epsilon):
        action = actor_output + epsilon * random_process()
        return action

    return func


def mse_loss(input, target):
    return tor.mean(tor.sum((input - target) ** 2))



class Trainer(object):


    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.mvavg_reward = deque(maxlen=100)
        self.callbacks = []
        self.acc_reward = 0

        self.estep = 0
        self.episode = 0

        self.verbose = True
        self.render = False

    def train(self, num_episodes, max_episode_len, render=False, verbose=True, callbacks=[]):
        mvavg_reward = deque(maxlen=100)
        self._warmup()
        self.verbose = True
        for episode in range(num_episodes):
            self.state = self.env.reset()
            t_episode_start = time.time()
            acc_reward = 0
            self._episode_start()

            for step in range(max_episode_len):
                s, r, d, i = self._episode_step(episode, acc_reward)
                for callback in callbacks:
                    callback.step(episode=episode, step=step, reward=r, **i)
                if render:
                    self.env.render()
                if d:
                    break
                acc_reward += r

            #TODO implement these callbacks a bit better
            for callback in callbacks:
                if hasattr(callback, "episode_step"):
                    callback.episode_step(episode=episode, step=step, episode_reward=acc_reward)
            self._episode_end(episode)

            mvavg_reward.append(acc_reward)
            episode_time = time.time() - t_episode_start
            if verbose:
                prRed("#Training time: {:.2f} minutes".format(time.clock() / 60))
                prGreen("#Episode {}. Mvavg reward: {:.2f} Episode reward: {:.2f} Episode steps: {} Episode time: {:.2f} min"\
                        .format(episode, np.mean(mvavg_reward), acc_reward, step + 1, episode_time / 60))

    def _episode_step(self):
        raise NotImplementedError()

    def _episode_start(self):
        raise NotImplementedError()

    def _episode_end(self):
        raise NotImplementedError()

    def _async_episode_step(self):
        raise NotImplementedError()

    def _warmup(self):
        pass


from multiprocessing import Lock

class HorizonTrainer(Trainer):
    """
    Trainer that is not episode based but horizon based, just another way of
    implementing the functionality in case that information has to be used between episodes
    consecutively for a more intuitive implementation.
    """
    def __init__(self, env, num_threads=1):
        super(HorizonTrainer, self).__init__(env)
        self.hstep = 0
        self.horizon = 1000000
        self.num_threads = num_threads
        self.l = Lock()
        self.le = Lock()
        self.async_steps = 0
        self.async_episode_steps = 0

    def train(self, horizon, max_episode_len, render=False, verbose=True, callbacks=[]):
        self.callbacks = callbacks
        self.estep = 0
        self.episode = 0

        self._warmup()
        self.verbose = True
        self.render = render

        self.state = self.env.reset()
        for self.hstep in range(horizon):

            self._horizon_step()
            self._horizon_step_end()

    def _horizon_step(self):
        raise NotImplementedError()

    def _async_step(self, **kwargs):
        self.l.acquire()
        self.async_steps+=1
        for callback in self.callbacks:
            callback.step(**kwargs)
        self.l.release()

    def _async_episode_step(self, **kwargs):
        self.le.acquire()
        self.mvavg_reward.append(acc_reward)
        self.async_episode_steps+=1
        
        self.le.release()


    def _horizon_step_end(self, **kwargs):

        logger.logkv('horizon_step', self.hstep)
        logger.logkv('episodes', self.episode)
        for callback in self.callbacks:
            callback.step(episode=self.episode, step=self.hstep, reward=self.acc_reward, **kwargs)

        
def reward_transform(env, obs, r, done, **kwargs):
    goal = kwargs.get('goal', None)
    if not goal is None:
        return -np.mean((obs - goal) ** 2)
    else:
        return r
