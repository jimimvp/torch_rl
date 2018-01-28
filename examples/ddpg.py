import gym
from torch.optim import Adam
from torch_rl.utils import *
from tqdm import tqdm

from torch_rl.models import SimpleNetwork
from torch_rl.core import ActorCriticAgent
from torch_rl.envs import NormalisedActions
from collections import deque
from torch_rl.utils import gauss_weights_init

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
tau = 0.001
epsilon = 1.0
depsilon = 0.0001
gamma = 0.99
replay_capacity = 100000
max_episode_length = 1000
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3

replay_memory = ReplayMemory(capacity=replay_capacity)
moving_avg_reward = deque(maxlen=replay_capacity)


env = NormalisedActions(gym.make("Pendulum-v0"))
env.reset()
num_actions = env.action_space.shape[0]
num_observations = env.observation_space.shape[0]
relu, tanh = tor.nn.ReLU(), tor.nn.Tanh()

action_choice_function = random_process_action_choice(OrnsteinUhlenbeckActionNoise(num_actions))

policy = cuda_if_available(SimpleNetwork([num_observations, 32, 16, num_actions],
                           activation_functions=[relu,relu,tanh]))
target_policy =  cuda_if_available(SimpleNetwork([num_observations, 32, 16, num_actions],
                           activation_functions=[relu,relu,tanh ]))
critic =  cuda_if_available(SimpleNetwork([num_observations+num_actions, 32, 16, num_actions],
                           activation_functions=[relu,relu]))
target_critic =  cuda_if_available(SimpleNetwork([num_observations+num_actions, 32, 16, num_actions],
                           activation_functions=[relu,relu]))

policy.apply(gauss_init(0,.2))
critic.apply(gauss_init(0,.2))

hard_update(target_policy, policy)
hard_update(target_critic, critic)

target_agent = ActorCriticAgent(target_policy, target_critic)
agent = ActorCriticAgent(policy, critic)


optimizer_critic = Adam(agent.critic_network.parameters(), lr=critic_learning_rate)
optimizer_policy = Adam(agent.policy_network.parameters(), lr=actor_learning_rate)
critic_criterion = mse_loss


# Warmup phase
state = env.reset()
for i in tqdm(range(replay_capacity)):
    action = agent.action(state)
    state_prev = state
    state, reward, done, info = env.step(action.cpu().data.numpy())
    replay_memory.push(state_prev, action, state, reward)

env.reset()
for episode in range(num_episodes):

    state = env.reset()
    acc_reward = 0
    for i in range(max_episode_length):
        env.render()
        action = agent.action(state)

***REMOVED***
        action = action_choice_function(action, epsilon)
        epsilon-=depsilon

        state_prev = state
        state, reward, done, info = env.step(action.cpu().data.numpy())

        replay_memory.push(state_prev, action, state, reward)

        acc_reward += reward
***REMOVED***

        batch = replay_memory.sample(batch_size)
        s1 = np.asarray([x.state for x in batch])
        s2 = np.asarray([x.next_state for x in batch])
        a1 = tor.cat([x.action for x in batch])
        r = to_tensor(np.asarray([x.reward for x in batch]).reshape(-1,1))

        # Critic optimization
        a2 = target_agent.actions(s2).cpu().data.numpy()

        q2 = target_agent.values(s2,a2)

        q_expected = r + gamma*q2
        q_predicted = agent.values(to_tensor(s1),a1.view((-1,1)), requires_grad=True)

        critic_loss = critic_criterion(q_expected, q_predicted)
***REMOVED***
        optimizer_critic.step()
        optimizer_critic.zero_grad()

***REMOVED***

        pred_a1 = agent.actions(s1, requires_grad=True).cpu().data.numpy()
        loss_actor = -1 * tor.mean(agent.values(s1, pred_a1, requires_grad=True))
***REMOVED***
        optimizer_policy.step()
        optimizer_policy.zero_grad()

        soft_update(target_agent.policy_network, agent.policy_network, tau)
        soft_update(target_agent.critic_network, agent.critic_network, tau)

    print("Episode", episode, " Accumulated Reward: ", acc_reward)

    if episode % 100 == 0 and episode != 0:
        print("#" * 100)
        print("Episode", episode, ". Average reward: ", np.mean(moving_avg_reward))
        print("#" * 100)








