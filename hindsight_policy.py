import torch as tor
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from envs import BitFlippingEnv
from utils import to_tensor
from models import PolicyAHG

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

def to_input(state, goal):

    x = np.hstack((state,goal))
    x = np.expand_dims(x, 0)
    return to_tensor(x)

def to_numpy(out):
    return out.data.numpy()


"""
    Implementation of the hindsight policy gradients.

"""



# Training
num_bits = 6
num_episodes = 10000
episode_length = 16

env = BitFlippingEnv(num_bits)


policy = PolicyAHG(num_bits*2, num_bits+1)
# Initialization of weights
policy.apply(weights_init)
policy.zero_grad()
optimizer = Adam(policy.parameters(), lr=0.001)

# Keeps track of the current episode
episode_steps = [0] * episode_length
for i in range(num_episodes):

    # The reward accumulated in the episode
    acc_reward = 0
    acc_distance = 0
    actions = []
    goal_occurances = {}

    for j in range(episode_length):

        # Normal step
        state = env.get_observation()
        goal = env.goal

        hgoal = tuple(state)
        goal_occurances[hgoal] = goal_occurances[hgoal] +1 if hgoal in goal_occurances else 1

        x = to_tensor(np.expand_dims(np.hstack((state, goal)),0))
        action = policy.forward(x).data.numpy()
        action = action.reshape(-1)

        episode_steps[j] = (state, action)

        action = np.argmax(action)


        state, reward, done, _ = env.step(action)

        acc_reward += reward
        acc_distance += env.distance()

    if i % 20 == 0:
        print(i ,". Episode reward: ", acc_reward, "distance: ", acc_distance )


    # Calculation of gradient
    pg  = 0
    for goal, c in goal_occurances.items():

        goal_prob = c / episode_length
        goal_grad = 0

        for state, action in episode_steps:
            if tuple(state) == goal:
                c -= 1
                if c == 0:
                    break
            action_ = policy.forward(to_input(state, goal))
            goal_grad += tor.log(action_[0][np.argmax(action)])*c

        pg += goal_prob * goal_grad
    pg.backward()
    optimizer.step()

    policy.zero_grad()
    env.reset()


