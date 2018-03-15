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

    def _episode_start(self):
        raise NotImplementedError()

    def _episode_end(self):
        raise NotImplementedError()

    def _warmup(self):
        pass



class HorizonTrainer(Trainer):
    """
    Trainer that is not episode based but horizon based, just another way of
    implementing the functionality in case that information has to be used between episodes
    consecutively for a more intuitive implementation.
    """
    def __init__(self, env):
        super(HorizonTrainer, self).__init__(env)
        self.hstep = 0
        self.horizon = 1000000

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


    def _episode_end(self):

        # TODO implement these callbacks a bit better
        for callback in self.callbacks:
            if hasattr(callback, "episode_step"):
                callback.episode_step(episode=self.episode, step=self.estep, episode_reward=self.acc_reward)

        if self.verbose:
            prRed("#Training time: {:.2f} minutes".format(time.clock() / 60))
            prGreen(
                "#Horizon step {}/{} Episode {} Mvavg reward: {:.2f}" \
                    .format(self.hstep, self.horizon, self.episode, np.mean(self.mvavg_reward)))

    def _episode_step(self, s, a, r, d):
        if self.render:
            self.env.render()
        self.acc_reward += r
        self.mvavg_reward.append(self.acc_reward)
        if d:
            self.acc_reward = 0
            self._episode_end()
            self.estep = 0
            self.acc_reward = 0
            self.episode+=1

    def _horizon_step(self):
        pass


def reward_transform(env, obs, r, done, **kwargs):
    goal = kwargs.get('goal', None)
    if not goal is None:
        return -np.mean((obs - goal) ** 2)
    else:
        return r
