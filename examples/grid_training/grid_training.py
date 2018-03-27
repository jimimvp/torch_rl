import gym
from torch_rl.utils import *

from torch_rl.models import SimpleNetwork
from torch_rl.envs import NormalisedActionsWrapper, NormalisedObservationsWrapper
from torch_rl.memory import SequentialMemory, SpikingHindsightMemory,  HindsightMemory
from torch_rl.training import DDPGTrainer, SpikingDDPGTrainer
from torch_rl.envs import SparseRewardGoalEnv
from torch_rl.utils.stats  import TrainingStatsCallback
from torch_rl.models import Reservoir
from torch_rl.utils import ParameterGrid
import datetime
import argparse
import os
"""
    Implementation of deep deterministic policy gradients with soft updates.

"""



assert "TRL_DATA_PATH" in os.environ, "Set TRL_DATA_PATH variable to the path where you want to store the training statistics."

ROOT_DIR = os.path.join(os.environ['TRL_DATA_PATH'], "training_data")
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EPISODES = 3000



def grid_training(p):

    hindsight = p.hindsight
    suff = "_her" if hindsight else ""
    suff = suff + "_reservoir" if p.reservoir else suff

    goal_indices = np.asarray([0, 1])
    env = SparseRewardGoalEnv(NormalisedObservationsWrapper(
        NormalisedActionsWrapper(gym.make("Pendulum-v0"))), precision=1e-1, indices=goal_indices)
    env.reset()
    num_actions = env.action_space.shape[0]
    num_observations = env.observation_space.shape[0] + 2
    relu, tanh = tor.nn.ReLU(), tor.nn.Tanh()

    c_act_functions = [relu] * len(p.c_middle_layer_size)
    a_act_functions = [relu] * len(p.a_middle_layer_size)
    a_act_functions.extend([tanh])

    if p.reservoir:
        print("Using reservoir...")
        reservoir = Reservoir(0.1, 0.01, env.observation_space.shape[0], p.reservoir, recursive=True, spectral_radius=p.spectral_radius)
        replay_memory = SpikingHindsightMemory(limit=p.replay_capacity, hindsight_size=p.hindsight_size,
                                               goal_indices=goal_indices, window_length=1)

        actor = cuda_if_available(
            SimpleNetwork([p.reservoir + len(goal_indices), *p.c_middle_layer_size, num_actions],
                          activation_functions=a_act_functions))

        critic = cuda_if_available(
            SimpleNetwork([p.reservoir + len(goal_indices) + num_actions, *p.c_middle_layer_size, 1],
                          activation_functions=c_act_functions))

        actor.apply(gauss_init(0, p.wsigma))
        critic.apply(gauss_init(0, p.wsigma))

        # Training
        trainer = SpikingDDPGTrainer(env=env, actor=actor, critic=critic, reservoir=reservoir,
                                     tau=p.tau, epsilon=p.epsilon, batch_size=p.batch, depsilon=p.epsilon, gamma=p.gamma,
                                     lr_actor=p.actor_learning_rate, lr_critic=p.critic_learning_rate, warmup=p.warmup,
                                     replay_memory=replay_memory)


    else:
        replay_memory = HindsightMemory(limit=p.replay_capacity, window_length=1, hindsight_size=p.hindsight_size,
                                        goal_indices=goal_indices) if hindsight else SequentialMemory(p.replay_capacity)

        actor = cuda_if_available(
            SimpleNetwork([num_observations, *p.a_middle_layer_size, num_actions],
                          activation_functions=a_act_functions))

        critic = cuda_if_available(
            SimpleNetwork([num_observations + num_actions, *p.c_middle_layer_size, 1],
                          activation_functions=c_act_functions))

        actor.apply(gauss_init(0, p.wsigma))
        critic.apply(gauss_init(0, p.wsigma))

        # Training
        trainer = DDPGTrainer(env=env, actor=actor, critic=critic,
                              tau=p.tau, epsilon=p.epsilon, batch_size=p.batch, depsilon=p.epsilon, gamma=p.gamma,
                              lr_actor=p.actor_learning_rate, lr_critic=p.critic_learning_rate, warmup=p.warmup,
                              replay_memory=replay_memory)



    output_dir = "{}/ddpg".format(ROOT_DIR) + suff + "_" + str(
        datetime.datetime.now()).replace(" ", "_")

    stats = TrainingStatsCallback(save_destination=output_dir)
    p.to_json(os.path.join(output_dir, "parameters.json"))

    trainer.train(EPISODES, max_episode_len=p.max_episode_length, verbose=True, callbacks=[stats])


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--config', "-c", default=os.path.join(SCRIPT_DIR, "grid.json"),
                    help='JSON file that specifies the parameter grid that is to be used for training')
parser.add_argument('--config-spiking', "-cs", default=os.path.join(SCRIPT_DIR, "grid_spiking.json"),
                    help='JSON file that specifies the parameter grid that is to be'
                         ' used for training of the reservoir version.')
parser.add_argument('--did', "-id", default=0,
                    help='ID of GPU to be used for training', type=int)

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


c1 = 0; c2 = 0;

with tor.cuda.device(args.did):

    for i, parameters in enumerate(grid_spiking):
        seed(666)
        print("####", "GRID TRAINING RESERVOIR {}/{}.".format(i, len(grid)), "####")
        grid_training(parameters)

        if i < len(grid):
            seed(666)
            print("####", "GRID TRAINING {}/{}.".format(i, len(grid)), "####")
            grid_training(grid[i])
