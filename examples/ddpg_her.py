import gym
from torch_rl.utils import *

from torch_rl.models import SimpleNetwork
from torch_rl.envs import NormalisedActionsWrapper, NormalisedObservationsWrapper
from torch_rl.memory import SequentialMemory, HindsightMemory
from torch_rl.training import DDPGTrainer
from torch_rl.envs import SparseRewardGoalEnv
from torch_rl.stats import RLTrainingStats
import datetime
import argparse
import numpy as np
"""
    Implementation of deep deterministic policy gradients with soft updates.

"""

# Training parameters
num_episodes = 2000
batch_size = 8*8 + 8
tau = 0.001
epsilon = 1.0
depsilon = 1. / 50000
gamma = 0.99
replay_capacity = 1000000
warmup = 2000
max_episode_length = 500
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
middle_layer_size = [400, 300]
weight_init_sigma = 0.003


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hindsight', "-hi", action="store_true", default=False,
                        help='Use hindsight replay buffer.')
    parser.add_argument('--epsilon', "-e", default=1.0, type=float,
                        help='Epsilon for exploration with linear decay.')
    parser.add_argument('--depsilon', "-de", default=5000., type=float,
                        help='Slope of linear decay of epsilon per step.')
    parser.add_argument('--gamma', "-g", default=.99, type=float,
                        help='Reward discount factor')
    parser.add_argument('--batch', "-bs", default=16*4+16, type=int,
                        help='Batch size, in case of hindsight replay has to be equal to hindsight_size*transitions + transitions')
    parser.add_argument('--replay_capacity', "-rc", default=1000000, type=int,
                        help="Capacity of the replay buffer.")
    parser.add_argument('--max_episode_length', "-mel", default=500, type=int,
                        help='Max number of steps per episode')
    parser.add_argument('--actor_learning_rate', "-aler", default=1e-4, type=float,
                        help='Learning rate of actor.')
    parser.add_argument('--critic_learning_rate', "-cler", default=1e-3, type=float,
                        help='Learning rate of critic.')
    parser.add_argument('--warmup', "-w", default=2000, type=int,
                        help='Number of steps to use for warmup.')
    parser.add_argument('--wsigma', "-ws", default=1e-3, type=float,
                        help='Sigma to use for weight initialization Gauss distribution.')
    parser.add_argument('--tau', "-t", default=1e-3, type=float,
                        help='Tau for soft updates of target actor and critic.')
    parser.add_argument('--hindsight_size', "-hs", default=4, type=int,
                        help='Size of hindsight per transition.')
    parser.add_argument('--config', "-c", default=None, type=str,
                        help='Path to config file for the training.')
    p = Parameters.from_args(parser.parse_args())

    hindsight = p.hindsight
    suff = "_her" if hindsight else ""

    goal_indices = np.asarray([0,1])

    replay_memory = HindsightMemory(limit=p.replay_capacity, window_length=1, hindsight_size=p.hindsight_size,
                                    goal_indices=goal_indices) if hindsight else SequentialMemory(p.replay_capacity, window_length=1)

    env = SparseRewardGoalEnv(NormalisedObservationsWrapper(
        NormalisedActionsWrapper(gym.make("Pendulum-v0"))), precision=1e-1, indices=goal_indices)

    env.reset()
    num_actions = env.action_space.shape[0]
    num_observations = env.observation_space.shape[0]+2
    relu, tanh = tor.nn.ReLU(), tor.nn.Tanh()

    actor = cuda_if_available(SimpleNetwork([num_observations, middle_layer_size[0], middle_layer_size[1], num_actions],
                                            activation_functions=[relu, relu, tanh]))

    critic = cuda_if_available(
        SimpleNetwork([num_observations + num_actions, middle_layer_size[0], middle_layer_size[1], 1],
                      activation_functions=[relu, relu]))

    actor.apply(gauss_init(0, p.wsigma))
    critic.apply(gauss_init(0, p.wsigma))

    # Training
    trainer = DDPGTrainer(env=env, actor=actor, critic=critic,
                          tau=p.tau, epsilon=p.epsilon, batch_size=p.batch, depsilon=p.epsilon, gamma=p.gamma,
                          lr_actor=p.actor_learning_rate, lr_critic=p.critic_learning_rate, warmup=p.warmup, replay_memory=replay_memory
                          )

    output_dir = "/disk/no_backup/vlasteli/Projects/torch_rl/examples/ddpg"+ suff + "_" + str(datetime.datetime.now()).replace(" ", "_")

    stats = RLTrainingStats(save_destination=output_dir)
    p.to_json(os.path.join(output_dir, "config.json"))

    trainer.train(8000, max_episode_len=p.max_episode_length, verbose=True, callbacks=[stats])


