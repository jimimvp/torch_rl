from collections import deque

import gym
import torch as tor
from torch.optim import Adam
from torch_rl.utils import *

from torch_rl.models import SimpleNetwork, Reservoir, SimpleSpikingAgent

"""
    Implementation of the hindsight experience replay with spiking reservoir and
    ANN readout.

"""



# Training parameters
num_episodes = 40000
episode_length = 2000
batch_size = 64
# How many from the same episode
batch_size_episode = 4

# Tracks the mean reward over a certain amount of episodes
mvr_tracker = deque(maxlen=50)
replay_memory = deque(maxlen=5000)

env = gym.make("MountainCar-v0")
env.reset()
num_actions = 3
num_observations = env.observation_space.shape[0]

spiking_net = Reservoir(0.1, 0.01,num_observations*2,200)
readout = SimpleNetwork([200, 32, 16, num_actions], activation_functions=[tor.nn.ReLU(),tor.nn.ReLU(),tor.nn.Softmax()])
readout.apply(gauss_weights_init(0, 0.02))

def sample_action(distribution):
    action = np.random.choice(distribution, p=distribution)
    action = np.argmax(distribution == action)
    return action


agent = SimpleSpikingAgent(spiking_net=spiking_net, readout_net=readout)


# Initialization of weights
readout.apply(gauss_weights_init(0,0.02))
readout.zero_grad()
optimizer = Adam(readout.parameters(), lr=0.001)
goal = np.asarray([0.5,0])

# Keeps track of the current episode
episode_steps = [0] * episode_length
for i in range(num_episodes):

    # The reward accumulated in the episode
    acc_reward = 0
    acc_distance = 0
    actions = []
    goal_occurances = {}
    goal_occurances[tuple([0.5,0])] = 1
    s1 = env.reset()
    for j in range(episode_length):

        # Normal step
        hgoal = tuple(s1.tolist())
        goal_occurances[hgoal] = 1

        a = agent.action(s1, goal)
        a = sample_action(a.data.numpy().reshape(-1))

        episode_steps[j] = (s1, a)
        state_prev = s1.copy()
        s2, r, done, _ = env.step(a)
        env.render()

        episode_steps[j] = Transition(s1, a, s2, r)

        acc_reward += r

    mvr_tracker.append(acc_reward)

    if i % 50 == 0:
        print(i,". Moving Average Reward:", np.mean(mvr_tracker), "Acc reward:", acc_reward)

    # Calculation of gradient
    pg = 0

    for i, transition_start in enumerate(episode_steps[:-batch_size_episode-1]):

        # The replay memory is a pair transition, all transitions after it in episode
        transitions_after = episode_steps[i+1:]
        for j, transition in enumerate(transitions_after):
            transitions_after[j] = Transition(transition.state, transition.action, transition.next_state, 1)
        replay_memory.append((transition_start, transitions_after))


    # Sample from replay memory
    if len(replay_memory) < 200:
        continue
    batch_raw = random.sample(replay_memory, int(batch_size/batch_size_episode))

    batch_raw = [random.sample(x[1], batch_size_episode) + [x[0]] for x in batch_raw]
    batch_raw = sum((x for x in batch_raw), [])


    def onehot(num):
        a = np.zeros(num_actions)
        a[num] = 1
        return a

    batch = np.asarray([np.hstack([x.state, x.next_state, onehot(x.action), x.reward]) for x in batch_raw], dtype=np.float32)
    batch_targets = to_tensor(batch[:,num_observations*2:num_observations*2+1])
    batch_input = batch[:,:2*num_observations]

    # print("Batch target shape: ", batch_targets.data.shape, "Batch input shape: ",  batch_input.data.shape)
    out = agent.action(batch_input)

    # Calculate gradient
    pg = out * batch_targets
    pg = pg[pg > 0]
    pg = tor.log(pg)
    pg = -tor.mean(pg)
    pg.backward()

    optimizer.step()
    optimizer.zero_grad()
    agent.policy_network.zero_grad()
    env.reset()

