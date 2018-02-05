from collections import deque

from torch.optim import Adam
from torch_rl.utils import *

from torch_rl.envs import BitFlippingEnv
from torch_rl.models import PolicySPG
from torch_rl.memory import SequentialMemory

"""
    Implementation of the hindsight experience replay: https://arxiv.org/pdf/1711.06006.pdf.

"""



# Training parameters
num_bits = 2
num_episodes = 80000
episode_length = 10
batch_size = 8

# Tracks the mean reward over a certain amount of episodes
mvr_tracker = deque(maxlen=100)
replay_memory = SequentialMemory(100000, window_length=1)

env = BitFlippingEnv(num_bits)

policy = cuda_if_available(PolicySPG(num_bits*2, num_bits+1))

# Initialization of weights
policy.apply(gauss_init(0,0.02))
policy.zero_grad()
optimizer = Adam(policy.parameters(), lr=0.001)



# Keeps track of the current episode
episode_steps = [0] * episode_length
for i in range(num_episodes):

    # The reward accumulated in the episode
    acc_reward = 0
    acc_distance = 0
    actions = [None] * episode_length

    for j in range(episode_length):

        # Normal step
        state = env.get_observation()
        goal = env.goal

        hgoal = tuple(state)

        x = to_tensor(np.hstack((state, goal)))

        action_distribution = policy.forward(x)
        action = policy.sample_action()
        state, reward, done, _ = env.step(action)
        reward = -1 if reward == 0 else reward

        if j == episode_length or done:
            replay_memory.append(state,action,reward, terminal=True)
            break
        else:
            replay_memory.append(state,action,reward, terminal=False)


        acc_reward+=reward

    mvr_tracker.append(acc_reward)

    if i % 200 == 0:
        print(i,". Moving Average Reward:", np.mean(mvr_tracker), "Acc reward:", acc_reward)

    # Sample from replay memory
    if replay_memory.observations.length < 200:
        continue
    s1, a, r, s2, t = replay_memory.sample_and_split(batch_size)
    g = np.repeat(env.goal, s1.shape[0], axis=0).reshape(s1.shape[0], -1)
    def onehot(num):
        a = np.zeros(num_bits+1)
        a[num] = 1
        return a
    a_mask = np.apply_along_axis(onehot, 1, a)


    out = policy.forward(to_tensor(np.hstack((s1, g))))

    out *= to_tensor(a_mask)

    # Calculate gradient
    idxs = tor.nonzero(out)
    out = out[idxs]
    pg = tor.log(out) * to_tensor(r)
    pg = -tor.mean(pg)
    pg.backward()

    optimizer.step()
    optimizer.zero_grad()
    policy.zero_grad()
    #env.reset()

