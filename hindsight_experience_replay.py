from collections import deque

from torch.optim import Adam
from envs import BitFlippingEnv
from utils import *
from models import PolicySPG

"""
    Implementation of the hindsight experience replay: https://arxiv.org/pdf/1711.06006.pdf.

"""



# Training parameters
num_bits = 8
num_episodes = 40000
episode_length = 16
batch_size = 16
# How many from the same episode
batch_size_episode = 4

# Tracks the mean reward over a certain amount of episodes
mvr_tracker = deque(maxlen=400)
replay_memory = deque(maxlen=300)

env = BitFlippingEnv(num_bits)

policy = PolicySPG(num_bits*2, num_bits+1)

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

    for j in range(episode_length):

        # Normal step
        state = env.get_observation()
        goal = env.goal

        hgoal = tuple(state)
        goal_occurances[hgoal] = 1

        x = to_input_state_goal(state, goal)

        action_distribution = policy.forward(x)

        action = policy.sample_action()

        episode_steps[j] = (state, action)
        state_prev = state
        state, reward, done, _ = env.step(action)

        episode_steps[j] = Transition(state_prev, action, state, reward)

        acc_reward += reward

    mvr_tracker.append(acc_reward)

    if i % 200 == 0:
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
    if len(replay_memory) < 300:
        continue
    batch_raw = random.sample(replay_memory, int(batch_size/batch_size_episode))

    batch_raw = [random.sample(x[1], batch_size_episode) + [x[0]] for x in batch_raw]
    batch_raw = sum((x for x in batch_raw), [])


    def onehot(num):
        a = np.zeros(num_bits+1)
        a[num] = 1

    batch = np.asarray([np.hstack([x.state, x.next_state, onehot(x.action), reward]) for x in batch_raw], dtype=np.float32)
    batch_targets = to_tensor(batch[:,num_bits*2:num_bits*2+1])
    batch_input = to_tensor(batch[:,:2*num_bits])

    # print("Batch target shape: ", batch_targets.data.shape, "Batch input shape: ",  batch_input.data.shape)
    out = policy.forward(batch_input)
    pg = tor.log(out * batch_targets)

    # To do gradient ascent
    pg = -tor.sum(pg)
    pg.backward()

    optimizer.step()
    optimizer.zero_grad()
    policy.zero_grad()
    env.reset()

