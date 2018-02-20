import gym
from torch.optim import Adam
from torch_rl.utils import *
import time

from torch_rl.models import SimpleNetwork
from torch_rl.core import ActorCriticAgent
from torch_rl.envs import NormalisedActionsWrapper
from torch_rl.memory import HindsightMemory, SequentialMemory
from torch_rl.stats import RLTrainingStats
import copy

"""
    Implementation of deep deterministic policy gradients with soft updates.

"""


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


def reward_transform(env, obs, r, done, **kwargs):
    goal = kwargs.get('goal', None)
    if not goal is None:
        return -np.mean((obs - goal) ** 2)
    else:
        return r

class DDPGTrainer(Trainer):

***REMOVED***

    def __init__(self, env, actor, critic, num_episodes=2000, max_episode_len=500, batch_size=32, gamma=.99,
***REMOVED***
***REMOVED***
***REMOVED***
        super(DDPGTrainer, self).__init__(env)
        if exploration_process is None:
            self.random_process = OrnsteinUhlenbeckActionNoise(self.env.action_space.shape[0])
***REMOVED***
            self.random_process = exploration_process
        self.action_choice_function = random_process_action_choice(self.random_process)
        self.tau = tau
        self.lr_critic = lr_critic
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.replay_memory = replay_memory
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.depsilon = depsilon
        self.warmup = warmup
        self.gamma = gamma
        self.target_critic = copy.deepcopy(critic)
        self.target_actor = copy.deepcopy(actor)
        self.optimizer_actor = Adam(actor.parameters(), lr=lr_actor) if optimizer_actor is None else optimizer_actor
        self.optimizer_critic = Adam(critic.parameters(), lr=lr_critic) if optimizer_critic is None else optimizer_critic

        self.goal_based = hasattr(env, "goal")

        self.target_agent = ActorCriticAgent(self.target_actor,self.target_critic)
        self.agent = ActorCriticAgent(actor, critic)

    def add_to_replay_memory(self,s,a,r,d):
***REMOVED***
            self.replay_memory.append(self.state, self.env.goal, a, r, d, training=True)
***REMOVED***
            self.replay_memory.append(self.state, a, r, d, training=True)

***REMOVED***

        for i in range(self.warmup):
            a = self.env.action_space.sample()
***REMOVED***
            self.add_to_replay_memory(self.state, a, r, d)
***REMOVED***

***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***
            action = self.agent.action(np.hstack((self.state, self.env.goal))).cpu().data.numpy()
***REMOVED***
            action = self.agent.action(self.state).cpu().data.numpy()

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

        self.add_to_replay_memory(self.state, action, reward, done)
***REMOVED***

***REMOVED***
***REMOVED***
            s1, g, a1, r, s2, terminal = self.replay_memory.sample_and_split(self.batch_size)
***REMOVED***
***REMOVED***
***REMOVED***
            s1, a1, r, s2, terminal = self.replay_memory.sample_and_split(self.batch_size)


***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***




class***REMOVED***DDPGTrainer(DDPGTrainer):

***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***
