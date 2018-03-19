from collections import deque

import gym
import torch_rl.envs

from torch.optim import Adam
from tqdm import tqdm
from torch_rl.utils import *

from torch_rl.models import SimpleNetwork
from torch_rl.core import Agent
import random

"""
    Implementation of vanilla policy gradient with experience replay.

"""


# Training parameters
num_episodes = 40000
episode_length = 500
batch_size = 8
# How many from the same episode
batch_size_episode = 4
epsilon = 0.6
edecay = 0.99
gamma = 0.9

# Tracks the mean reward over a certain amount of episodes
mvr_tracker = deque(maxlen=20)
replay_memory = deque(maxlen=2000)

env = gym.make("MountainCar-v0")
num_actions = env.action_space.shape[0]
num_observations = env.observation_space.shape[0]

policy = SimpleNetwork([num_observations, 128, 64, 32, num_actions], activation_functions=[tor.nn.ReLU(),tor.nn.ReLU(),
                                                                                           tor.nn.ReLU(),tor.nn.Softmax()])
policy.apply(gauss_weights_init(0, 0.02))
policy = cuda_if_available(policy)

optimizer = Adam(policy.parameters(), lr=1e-4)
possible_actions = np.arange(num_actions)

def sample_action(distribution, epsilon):
    """
    Choose random action with probability epsilon else sample from estimated distribution
    :param distribution:
    :param epsilon:
    :return:
    """
    if np.random.choice([False, True], p=[1-epsilon, epsilon]):
        action = random.choice(possible_actions)
        return action

    action = np.random.choice(distribution, p=distribution)
    action = np.argmax(distribution == action)
    return action


agent = Agent(policy_network=policy, action_choice_function=sample_action)


def onehot(num):
    a = np.zeros(num_actions)
    a[num] = 1
    return a


# Keeps track of the current episode
episode_steps = [0] * episode_length

state_prev = env.reset()
prev_reward = 0
while len(replay_memory) < 500:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    reward = state[0] - 0.6 - (state_prev[0] - 0.6) + prev_reward * gamma
    action_distribution = agent.action(state, requires_grad=True)
    replay_memory.append(Transition(state_prev, action_distribution[action], state, reward))
    if done:
        break
    env.render()

for i in tqdm(range(num_episodes)):

    # The reward accumulated in the episode
    acc_reward = 0
    acc_distance = 0
    actions = [None] * episode_length
    goal_occurances = {}
    prev_reward = 0

    state = env.reset()
    episode_buffer = [None] * episode_length


    for j in range(episode_length):

        # Normal step
        action_distribution = agent.action(state, requires_grad=True)
        action = sample_action(action_distribution.cpu().data.numpy(), epsilon)

        state_prev = state
        state, reward, done, _ = env.step(action)
        reward = state[0]-0.6 - (state_prev[0]-0.6) + prev_reward*gamma

        prev_reward = reward

        env.render()

        episode_buffer[j] = Transition(state_prev, action_distribution[action], state, reward)
        acc_reward += reward

         # Calculation of gradient
        pg = 0
        # Sample from replay memory
        if len(replay_memory) < 8:
            continue

        batch_raw = random.sample(replay_memory, batch_size)
        batch = [tor.cat([to_tensor(x.state), x.action]).view(-1,num_observations+1) for x in batch_raw]
        batch = tor.cat(batch)
        batch_rewards = to_tensor(np.asarray([x.reward for x in batch_raw]).reshape(-1,1))

        batch_states = batch[:, 0:num_observations]
        batch_actions = batch[:, num_observations:num_observations+num_actions]
        # Calculate gradient
        pg = -tor.mean(tor.log(batch_actions * batch_rewards))
        pg.backward(retain_graph=True)


        optimizer.step()
        optimizer.zero_grad()
        policy.zero_grad()

    replay_memory.extend(episode_buffer)
    epsilon *= edecay

    mvr_tracker.append(acc_reward)
    print(i, "Episode reward:", acc_reward)
    if i % 10 == 0:
        print(i,". Moving Average Reward:", np.mean(mvr_tracker), "Acc reward:", acc_reward)
