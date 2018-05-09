import gym

from torch_rl.models import SimpleNetwork
from torch_rl.envs import NormalisedActionsWrapper
from torch_rl.memory import HindsightMemory, SequentialMemory
from torch_rl.training import DDPGTrainer
from torch_rl.utils.stats import TrainingStatsCallback
from torch_rl.utils import cuda_if_available, gauss_init
from torch_rl.utils.callbacks import CheckpointCallback
from torch_rl.utils import timestamp
from torch_rl.envs.wrappers import RunningMeanStdNormalize
from torch_rl.envs import EnvLogger, GoalEnv
from gym.wrappers import Monitor

from torch_rl import config
import numpy as np
import os

import torch as tor


import datetime
"""
    Implementation of deep deterministic policy gradients with soft updates.

"""

# Training parameters
num_episodes = 2000
batch_size = 64
tau = 0.001
epsilon = 1.0
depsilon = 1. / 3000
gamma = 0.99
replay_capacity = 1000000
warmup = 2000
max_episode_length = 500
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
middle_layer_size = [64, 64]
weight_init_sigma = 0.003
env_name = 'TRLRoboschoolReacher-v1'
goal_indices = [0,1]
sparse = True
hindsight = True


replay_memory = HindsightMemory(limit=1000000, goal_indices=goal_indices) if hindsight else SequentialMemory(limit=1000000)

# Interpolation parameter v * ppo_gradient + (1-v) * off_policy_gradient
np.random.seed(456)
tor.manual_seed(456)

root_dir = 'torch_rl_her_ddpg_' + env_name.lower().split("-")[0]
if sparse:
  root_dir += '_sparse'

if hindsight:
  root_dir += '_hindsight'


config.set_root(root_dir, force=True)
config.configure_logging(clear=False, output_formats=['tensorboard', 'stdout', 'json'])
# config.start_tensorboard()

env = EnvLogger(NormalisedActionsWrapper(gym.make(env_name)))
env = GoalEnv(RunningMeanStdNormalize(env), target_indices=[0, 1], curr_indices=[2, 3], sparse=sparse, precision=1e-1)
#Make it a sparse goal environment, precision is absolute error tolerance to target

env.reset()

num_actions = env.action_space.shape[0]
num_observations = env.observation_space.shape[0]
relu, tanh = tor.nn.ReLU(), tor.nn.Tanh()

actor = cuda_if_available(SimpleNetwork([num_observations, middle_layer_size[0], middle_layer_size[1], num_actions],
                                        activation_functions=[relu, relu, tanh]))

critic = cuda_if_available(
    SimpleNetwork([num_observations + num_actions, middle_layer_size[0], middle_layer_size[1], 1],
                  activation_functions=[relu, relu]))

actor.apply(gauss_init(0, weight_init_sigma))
critic.apply(gauss_init(0, weight_init_sigma))


# Training
trainer = DDPGTrainer(env=env, actor=actor, critic=critic,
                      tau=tau, epsilon=epsilon, batch_size=batch_size, depsilon=depsilon, gamma=gamma,
                      lr_actor=actor_learning_rate, lr_critic=critic_learning_rate, warmup=warmup, replay_memory=replay_memory
                      )

checkpoint_callback = CheckpointCallback(save_path=config.root_path(), models={"actor": actor, "critic" : critic})



trainer.train(2000, max_episode_len=500, verbose=True, callbacks=[checkpoint_callback])
