from torch_rl.training.core import HorizonTrainer, mse_loss
import copy
from torch_rl.memory import GeneralisedSequentialMemory
from torch.optim import Adam
from torch_rl.utils import to_tensor
from torch.autograd import Variable
import torch as tor
import numpy as np
from tqdm import tqdm

class PPOTrainer(HorizonTrainer):


    """
        Implementation of proximal policy optimization.

        A = Q(s,a) - V(s)
        ratio = pi(s)/pi_old(s)

    """

***REMOVED***

    def __init__(self, env, network, network_old, num_episodes=2000, max_episode_len=500, batch_size=32, gamma=.99,
              replay_memory=GeneralisedSequentialMemory(1000000, window_length=1), lr=1e-4, memory_fill_steps=300,
                 epsilon=1.,optimizer=None, gae_param=0.95):
        super(PPOTrainer, self).__init__(env)
        self.lr = lr
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.replay_memory = replay_memory
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.gamma = gamma
        self.gae_param = gae_param
        self.optimizer_actor = Adam(network.parameters(), lr=lr) if optimizer is None else optimizer
        self.goal_based = hasattr(env, "goal")
        self.memory_fill_steps = memory_fill_steps
        self.network = network
        self.network_old = network_old
        self.N = 10

    def add_to_replay_memory(self,s,a,r,d):
***REMOVED***
            self.replay_memory.append(self.state, self.env.goal, a, r, d, training=True)
***REMOVED***
            self.replay_memory.append(self.state, a, r, d, training=True)

    def _episode_end(self):
***REMOVED***
***REMOVED***
***REMOVED***
    def _episode_step(self):
***REMOVED***

    def gather_experience_and_calc_advantages(self):
        states = []
        actions = []
        rewards = []
        values = []
        returns = []
        advantages = []
        terminal = []
        acc_reward  = 0
        self.episode_length = 0
        for i in tqdm(range(self.memory_fill_steps)):
            states.append(self.state)
            action, value = self.network(to_tensor(self.state).view(1,-1))
            actions.append(action)
            values.append(value)

            env_action = action.data.squeeze().numpy()
            state, reward, done, _ = self.env.step(env_action)
            self.episode_length += 1
            acc_reward += reward

***REMOVED***tate
            done = (done or self.episode_length >= self.max_episode_len)
            terminal.append(done)
            if done:
                self.env.reset()

            reward = max(min(reward, 1), -1)
            rewards.append(reward)

            R = tor.zeros(1, 1)
            if not done:
                self.mvavg_reward.append(acc_reward)
                acc_reward = 0
                action, value = self.network(to_tensor(state).view(1,-1))
                R = value.data
            R = Variable(R)
            values.append(R)

            A = Variable(tor.zeros(1, 1))
            for i in reversed(range(len(rewards))):
                td = rewards[i] + self.gamma * values[i + 1].data - values[i].data
                A = float(td) + self.gamma * self.gae_param * A
                advantages.insert(0, A)
                R = A + values[i]
                returns.insert(0, R)

            for i, s in enumerate(states):
                self.replay_memory.append(s, actions[i], rewards[i], terminal[i], [returns[i], advantages[i]])


    def sample_and_update(self):
        av_loss = 0
        self.network_old.load_state_dict(self.network.state_dict())
        for step in tqdm(range(self.N)):
            # cf https://github.com/openai/baselines/blob/master/baselines/pposgd/pposgd_simple.py
            batch_states, batch_actions, batch_rewards, batch_states1, batch_terminals, extra = \
                self.replay_memory.sample_and_split(self.batch_size)


            batch_returns = extra[:, 0]
            batch_advantages = extra[:, 1]

            # old probas
            actions_old, v_pred_old = self.network_old(batch_states.detach())
            probs_old = self.network_old.log_prob(batch_actions)

            # new probabilities
            actions, v_pred = self.network(batch_states)
            probs = self.agent.policy_network.log_prob(batch_actions)

            # ratio
            ratio = probs / (1e-15 + probs_old)
            # clip loss
            surr1 = ratio * tor.stack([batch_advantages] * batch_actions.shape[1],
                                      1)  # surrogate from conservative policy iteration
            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * tor.stack(
                [batch_advantages] * batch_actions.shape[1], 1)
            loss_clip = -tor.mean(tor.min(surr1, surr2))
            # value loss
            vfloss1 = (v_pred - batch_returns) ** 2
            v_pred_clipped = v_pred_old + (v_pred - v_pred_old).clamp(-self.epsilon, self.epsilon)
            vfloss2 = (v_pred_clipped - batch_returns) ** 2
            loss_value = 0.5 * tor.mean(tor.max(vfloss1, vfloss2))  # also clip value loss
            # entropy
            loss_ent = -self.ent_coeff * tor.mean(np.e * probs * probs)
            # total
            total_loss = (loss_clip + loss_value + loss_ent)
            av_loss += total_loss.data[0] / float(self.num_episodes)
            # before Adam step, update old_model:

            # model_old.load_state_dict(model.state_dict())
            # step
            self.optimizer.zero_grad()
            # model.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.replay_memory.clear()

    def _horizon_step(self):

        self.gather_experience_and_calc_advantages()
        self.sample_and_update()


from torch_rl.envs.wrappers import *
import gym
from torch_rl.models.ppo import PPOGaussianPolicy

env = NormalisedObservationsWrapper(
    NormalisedActionsWrapper(gym.make("Pendulum-v0")))

network = PPOGaussianPolicy([env.observation_space.shape[0],30, env.action_space.shape[0]*2+1])
network_old = PPOGaussianPolicy([env.observation_space.shape[0],30, env.action_space.shape[0]*2+1])
trainer = PPOTrainer(network=network, network_old=network_old, env=env)
trainer.train(horizon=100000, max_episode_len=500)

