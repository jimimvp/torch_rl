from collections import deque

import numpy as np
import torch as tor
from torch.optim import Adam
from torch_rl.utils import to_tensor, gauss_weights_init

from torch_rl.envs import BitFlippingEnv
from torch_rl.models import PolicyAHG

"""
    Implementation of the hindsight policy gradient paper: https://arxiv.org/pdf/1711.06006.pdf.

"""



# Training parameters
num_bits = 8
num_episodes = 10000
episode_length = 16
mvr_tracker = deque(maxlen=400)


env = BitFlippingEnv(num_bits)


policy = PolicyAHG(num_bits*2, num_bits+1)

# Initialization of weights
policy.apply(gauss_weights_init(0,0.02))
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
    goal_occurances[tuple(env.goal)] = 1
    state_and_goal = np.zeros((1, num_bits*2))


    for j in range(episode_length):

        # Normal step
        state = env.get_observation()
        goal = env.goal

        hgoal = tuple(state)
        goal_occurances[hgoal] = goal_occurances[hgoal] + 1 if hgoal in goal_occurances else 1
        state_and_goal[0][0:num_bits] = state
        state_and_goal[0][num_bits::] = goal

        x = to_tensor(state_and_goal,0)

        action_distribution = policy.forward(x)
        action = policy.sample_action()

        episode_steps[j] = (state, action)

        state, reward, done, _ = env.step(action)

        acc_reward += reward

    mvr_tracker.append(acc_reward)

    if i % 200 == 0:
        print(i,". Moving Average Reward:", np.mean(mvr_tracker), "Acc reward:", acc_reward)

    # Calculation of gradient
    pg = 0
    for goal, c in goal_occurances.items():

        goal_prob = c / episode_length
        goal_grad = 0

        for state, a in episode_steps:
            if tuple(state) == goal:
                c -= 1
                if c == 0:
                    break
            state_and_goal[0][0:num_bits] = state
            state_and_goal[0][num_bits::] = goal
            action_ = policy.forward(to_tensor(state_and_goal))
            goal_grad += tor.log(action_[0][a])*c

        pg += goal_prob * goal_grad
    # To do gradient ascent
    pg = -pg
    pg.backward()

    optimizer.step()
    optimizer.zero_grad()
    policy.zero_grad()
    env.reset()


