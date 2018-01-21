from torch.optim import Adam
from utils import *
from models import SimpleNetwork
import gym
from envs import NormalisedActions
from core import ActorCriticAgent
import torch.nn.functional as F
"""
    Implementation of deep deterministic policy gradients with soft updates.

"""

def random_process_action_choice(random_process):

    def func(actor_output, epsilon):
        action = actor_output + to_tensor(epsilon*random_process())
        return action
    return func

def mse_loss(input, target):
    return tor.mean(tor.sum((input - target)**2))

# Training parameters
num_bits = 8
num_episodes = 80000
batch_size = 32
tau = 0.9
epsilon = 1.0
depsilon = 0.0001
gamma = 0.9
replay_capacity = 100
max_episode_length = 500
learning_rate = 1e-4


replay_memory = ReplayMemory(capacity=replay_capacity)
moving_avg_reward = deque(maxlen=replay_capacity)


env = NormalisedActions(gym.make("Pendulum-v0"))
env.reset()
num_actions = env.action_space.shape[0]
num_observations = env.observation_space.shape[0]


action_choice_function = random_process_action_choice(OrnsteinUhlenbeckActionNoise(num_actions))

policy = SimpleNetwork([num_actions, 32, 16, 1])
target_policy = SimpleNetwork([num_actions, 32, 16, 1])
critic = SimpleNetwork([num_actions+num_observations, 32, 16, 1])
target_critic = SimpleNetwork([num_actions+num_observations, 32, 16, 1])

hard_update(target_policy, policy)
hard_update(target_critic, critic)

target_agent = ActorCriticAgent(target_policy, target_critic)
agent = ActorCriticAgent(policy, critic)


optimizer_critic = Adam(agent.critic_network.parameters(), lr=learning_rate)
optimizer_policy = Adam(agent.policy_network.parameters(), lr=learning_rate)
critic_criterion = mse_loss


# Warmup phase
state = env.reset()
for i in range(replay_capacity):
    action = agent.action(state)
    state_prev = state.copy()
    state, reward, done, info = env.step(action.data.numpy())
    replay_memory.push(state_prev, action, state, reward)

env.reset()
for episode in range(num_episodes):

    state = env.reset()
    acc_reward = 0
    for i in range(max_episode_length):

        action = agent.action(state)

***REMOVED***
        action = action_choice_function(action, epsilon)
        epsilon-=depsilon

        state_prev = state.copy()
        state, reward, done, info = env.step(action.data.numpy())

        replay_memory.push(state_prev, action, state, reward)

        acc_reward += reward
***REMOVED***

        batch = replay_memory.sample(batch_size)
        s1 = np.asarray([x.state for x in batch])
        s2 = np.asarray([x.next_state for x in batch])
        a1 = np.asarray([x.action for x in batch]).reshape((-1,1))
        r = to_tensor(np.asarray([x.reward for x in batch]).reshape(-1,1))

        # Critic optimization
        a2 = target_agent.actions(s1).data.numpy()

        q2 = target_agent.values(s2,a2)

        q_expected = r + gamma*q2
        q_predicted = agent.values(s1, a1, requires_grad=True)

        critic_loss = critic_criterion(q_expected, q_predicted)
        critic_loss.backward()
        optimizer_critic.step()
        optimizer_critic.zero_grad()

***REMOVED***

        pred_a1 = agent.actions(s1, requires_grad=True).data.numpy()
        loss_actor = -1 * tor.sum(agent.values(s1, pred_a1, requires_grad=True))
        loss_actor.backward()
        optimizer_policy.step()
        optimizer_policy.zero_grad()

        soft_update(target_agent.policy_network, agent.policy_network, tau)
        soft_update(target_agent.critic_network, agent.critic_network, tau)

    print("Episode", episode, " Accumulated Rewrad: ", acc_reward)

    if episode % 100 == 0 and episode != 0:
        print("#" * 100)
        print("Episode", episode, ". Average reward: ", np.mean(moving_avg_reward))
        print("#" * 100)








