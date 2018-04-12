
from torch_rl.envs.wrappers import *
import gym

# Use Monitor to write the environment rewards to file
from gym.wrappers import Monitor

# Actor-critic model to be used in training with PPO
from torch_rl.models.ppo import ActorCriticPPO
# Trainer with PPO training algorithm
from torch_rl.training.ppo import GPUPPOTrainer
from torch_rl.utils import *
from torch_rl.utils import xavier_uniform_init
from torch_rl.models import Reservoir

# Use for logging of moving average episode rewards to console
from torch_rl.envs import EnvLogger, ReservoirObservationWrapper, ShrinkEnvWrapper
from torch_rl import config
import roboschool


import sys
import os
import random
import numpy as np

env_name = 'RoboschoolReacher-v1'
output_dir = env_name.lower().split("-")[0]

pomdp_indices = .6 #% original observations
recursive_reservoir = True

np.random.seed(456)
tor.manual_seed(456)
# config.start_tensorboard()

reservoir_size = 200
spectral_radius = .6

env = EnvLogger(NormalisedActionsWrapper(gym.make(env_name)))
if pomdp_indices:
    output_dir+='_pomdp{}'.format(pomdp_indices)
    pomdp_indices = int(pomdp_indices*env.observation_space.shape[0])
    pomdp_indices = np.random.choice(np.arange(env.observation_space.shape[0]), pomdp_indices)
    print('Chosen observation indices for pomdp: ', pomdp_indices)
    env = ShrinkEnvWrapper(env, pomdp_indices)

if not recursive_reservoir:
    print('Using non-recursive reservoir')
    output_dir+='_nor'
else:
    output_dir+='_recursive{}'.format(spectral_radius)



config.set_root(output_dir, force=True)
config.configure_logging(clear=False, output_formats=['tensorboard', 'stdout', 'json'])



# Creating the reservoir for observation transformation
reservoir = Reservoir(dt=0.01, sim_dt=0.005, input_size=env.observation_space.shape[0], network_size=reservoir_size, recursive=recursive_reservoir,
                 spectral_radius=spectral_radius,noise=False, synapse_filter=15e-3)

# This wrapper transforms a normal observation to a spiking observation
env = BaselinesNormalize(env)
env = ReservoirObservationWrapper(env, reservoir)


with tor.cuda.device(0):
    network = ActorCriticPPO([reservoir_size, 64, 64, env.action_space.shape[0]])
    network.apply(xavier_uniform_init())

    trainer = GPUPPOTrainer(network=network, env=env, n_update_steps=10, 
        n_steps=2048, n_minibatches=32, lmda=.95, gamma=.99, lr=3e-4, epsilon=0.1, ent_coef=0.0)
    trainer.train(horizon=100000, max_episode_len=500)

