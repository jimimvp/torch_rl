import torch as tor
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from envs import BitFlippingEnv
from utils import to_tensor


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



class Policy(nn.Module):

    def __init__(self, obs_size, act_size):
        super(Policy, self).__init__()

        self.f1 = nn.Linear(obs_size*2,64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, act_size+1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, x):

        out = self.f1(x)
        out = self.relu(out)
        out = self.f2(out)
        out = self.relu(out)
        out = self.f3(out)
        out = self.softmax(out)
        return out



# Training
num_bits = 8
num_episodes = 40000
episode_length = 20

env = BitFlippingEnv(num_bits)
policy = Policy(num_bits, num_bits)
policy.apply(weights_init)
policy.zero_grad()

optimizer = Adam(policy.parameters(), lr=0.001)

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
    pg  = 0
    for goal, c in goal_occurances.items():

        goal_prob = c / episode_length
        goal_grad = 0

        for state, action in episode_steps:
            if tuple(state) == goal:
                c-=1

            action_ = policy.forward(to_input(state, goal))
            goal_grad += tor.log(action_[0][np.argmax(action)])*c

        pg += goal_prob * goal_grad
    pg.backward()
    optimizer.step()

    policy.zero_grad()
    env.reset()


