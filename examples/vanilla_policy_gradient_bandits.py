from collections import deque

import gym
import torch_rl.envs

from torch.optim import Adam
from tqdm import tqdm
from torch_rl.utils import *

from torch_rl.models import SimpleNetwork
from torch_rl.core import Agent

"""
    Implementation of vanilla policy gradient with experience replay.

"""


# Training parameters
num_episodes = 40000
episode_length = 20
batch_size = 8
# How many from the same episode
batch_size_episode = 4

# Tracks the mean reward over a certain amount of episodes
mvr_tracker = deque(maxlen=20)
replay_memory = deque(maxlen=5000)

env = gym.make("BanditsX2-v0")
num_actions = env.action_space.shape[0]
num_observations = env.observation_space.shape[0]

policy = SimpleNetwork([num_observations, 32, 16, num_actions], activation_functions=[tor.nn.ReLU(),tor.nn.ReLU(),tor.nn.Softmax()])
policy.apply(gauss_weights_init(0, 0.02))
optimizer = Adam(policy.parameters())

def sample_action(distribution):
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

print("Distribution: ", env.bandit_distributions)
for i in tqdm(range(num_episodes)):

    # The reward accumulated in the episode
    acc_reward = 0
    acc_distance = 0
    actions = [None] * episode_length
    goal_occurances = {}
    prev_reward = 0

    state = env.reset()
    episode_buffer = [None] * episode_length
    episode_actions = np.zeros(num_actions)
    for j in range(episode_length):

        # Normal step
        action_distribution = agent.action(np.ones(2), requires_grad=True)
        action = sample_action(action_distribution.data.numpy())

        state_prev = state.copy()
        state, reward, done, _ = env.step(action)

        prev_reward = -0.5 + state[0]

        episode_buffer[j] = Transition(state_prev, action_distribution * to_tensor(onehot(action)), state, reward.astype(np.float32))
        acc_reward += reward

        episode_actions[action] +=1
         # Calculation of gradient
        pg = 0
        # Sample from replay memory
        if len(replay_memory) < 8:
            continue

        batch_raw = random.sample(replay_memory, batch_size)
        batch = [tor.cat([to_tensor(x.state), x.action]).view(-1,num_observations+num_actions) for x in batch_raw]
        batch = tor.cat(batch)
        batch_rewards = to_tensor(np.asarray([x.reward for x in batch_raw]).reshape(-1,1))

        batch_states = batch[:, 0:num_observations]
        batch_actions = batch[:, num_observations:num_observations+num_actions]
        # Calculate gradient
        pg = -tor.mean(tor.log(batch_actions[(batch_actions>0).detach()] * batch_rewards))
        pg.backward(retain_graph=True)


        optimizer.step()
        optimizer.zero_grad()
        policy.zero_grad()

    replay_memory.extend(episode_buffer)

    mvr_tracker.append(acc_reward)

    if i % 10 == 0:
        print(i,". Moving Average Reward:", np.mean(mvr_tracker), "Acc reward:", acc_reward)
        print("Episode actions: ", episode_actions)
