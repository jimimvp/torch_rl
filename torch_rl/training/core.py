import torch as tor
from torch_rl.utils import *
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

    def train(self, num_episodes, max_episode_len, render=False, verbose=True, callbacks=[]):
        mvavg_reward = deque(maxlen=100)
        self._warmup()
        self.verbose = True
        for episode in range(num_episodes):
***REMOVED***elf.env.reset()
            t_episode_start = time.time()
            acc_reward = 0
            self._episode_start()

            for step in range(max_episode_len):
                s, r, d, _ = self._episode_step(episode, acc_reward)
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

***REMOVED***
        raise NotImplementedError()

    def _episode_end(self):
        raise NotImplementedError()

***REMOVED***
***REMOVED***



class HorizonTrainer(Trainer):
    """
    Trainer that is not episode based but horizon based, just another way of
    implementing the functionality in case that information has to be used between episodes
    consecutively for a more intuitive implementation.
    """
    def __init__(self, env):
        super(HorizonTrainer, self).__init__(env)

    def train(self, horizon, max_episode_len, render=False, verbose=True, callbacks=[]):
        self.mvavg_reward = deque(maxlen=100)
        episode = 0

        self._warmup()
        self.verbose = True

        estep = 1
        acc_reward = 0


        self.state = self.env.reset()
        self._episode_start()
        for hstep in range(horizon):

            s, r, d, _ = self._horizon_step()
            acc_reward += r
            if render:
                    self.env.render()
            if not d or estep > max_episode_len:
                self._episode_end()
    ***REMOVED***elf.env.reset()
                estep = 0
                acc_reward = 0
                t_episode_start = time.time()
                self._episode_start()

                # TODO implement these callbacks a bit better
                for callback in callbacks:
                    if hasattr(callback, "episode_step"):
                        callback.episode_step(episode=episode, step=estep, episode_reward=acc_reward)
                self._episode_end(episode)

            if verbose:
                prRed("#Training time: {:.2f} minutes".format(time.clock() / 60))
                prGreen(
                    "#Horizon step {}/. Mvavg reward: {:.2f}" \
                    .format(hstep, np.mean(self.mvavg_reward)))


    def _horizon_step(self):
***REMOVED***



def reward_transform(env, obs, r, done, **kwargs):
    goal = kwargs.get('goal', None)
    if not goal is None:
        return -np.mean((obs - goal) ** 2)
    else:
        return r
