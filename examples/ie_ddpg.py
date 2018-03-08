import gym
from torch_rl.utils import *

from torch_rl.models import SimpleNetwork, GaussianPolicy
from torch_rl.envs import NormalisedActionsWrapper, NormalisedObservationsWrapper
from torch_rl.memory import SequentialMemory,  HindsightMemory
from torch_rl.training.ie_ddpg import IEDDPGTrainer
from torch_rl.envs import SparseRewardGoalEnv
from torch_rl.stats import RLTrainingStats
from torch_rl.utils import ParameterGrid
import datetime
import argparse
import os
"""
    Implementation of deep deterministic policy gradients with soft updates.

"""

ROOT_DIR = "/disk/no_backup/vlasteli/Projects/torch_rl/examples/training_stats/grid_training_mc"
EPISODES = 2000

goal_indices = np.asarray([0, 1])
env = NormalisedObservationsWrapper(
    NormalisedActionsWrapper(gym.make("Pendulum-v0")))
def grid_training(p):

    hindsight = p.hindsight
    suff = "_her" if hindsight else ""
    suff = suff + "_reservoir" if p.reservoir else suff


    env.reset()
    num_actions = env.action_space.shape[0]
    num_observations = env.observation_space.shape[0]
    relu, tanh = tor.nn.ReLU(), tor.nn.Tanh()

    c_act_functions = [relu] * len(p.c_middle_layer_size)
    a_act_functions = [relu] * len(p.a_middle_layer_size)
    a_act_functions.extend([tanh])


    replay_memory = HindsightMemory(limit=p.replay_capacity, window_length=1, hindsight_size=p.hindsight_size,
                                    goal_indices=goal_indices) if False else SequentialMemory(p.replay_capacity)

    actor = cuda_if_available(
        GaussianPolicy([num_observations, *p.a_middle_layer_size, num_actions*2],
                      activation_functions=a_act_functions))

    critic = cuda_if_available(
        SimpleNetwork([num_observations + num_actions, *p.c_middle_layer_size, 1],
                      activation_functions=c_act_functions))

    actor.apply(gauss_init(0, p.wsigma))
    critic.apply(gauss_init(0, p.wsigma))

    # Training
    trainer = IEDDPGTrainer(env=env, actor=actor, critic=critic,
                          tau=p.tau, epsilon=p.epsilon, batch_size=p.batch, depsilon=p.epsilon, gamma=p.gamma,
                          lr_actor=p.actor_learning_rate, lr_critic=p.critic_learning_rate, warmup=p.warmup,
                          replay_memory=replay_memory)



    output_dir = "{}/ddpg".format(ROOT_DIR) + suff + "_" + str(
        datetime.datetime.now()).replace(" ", "_")

    stats = RLTrainingStats(save_destination=output_dir)
    p.to_json(os.path.join(output_dir, "parameters.json"))

    trainer.train(EPISODES, max_episode_len=p.max_episode_length, verbose=True, callbacks=[stats])


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--config', "-c", default="grid.json",
                    help='JSON file that specifies the parameter grid that is to be used for training')
parser.add_argument('--config-spiking', "-cs", default="grid_spiking.json",
                    help='JSON file that specifies the parameter grid that is to be'
                         ' used for training of the reservoir version.')

args = parser.parse_args()

grid = ParameterGrid.from_config(args.config)
grid_spiking = ParameterGrid.from_config(args.config_spiking)


import numpy as np
import torch as tor
import random
def seed(s):
    np.random.seed(s)
    tor.manual_seed(s)
    random.seed(s)

c1, c2 = 0, 0
for i in range(1000000):
    print("####", "GRID TRAINING ANN {}/{}.".format(c2+1, len(grid)), "####")
    if c2 < len(grid):
        c2+=1
        grid_training(next(grid.__iter__()))
