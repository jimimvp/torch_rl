
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


# Use for logging of moving average episode rewards to console
from torch_rl.envs import EnvLogger
from torch_rl import config


import sys
import os
import roboschool

env_name = 'RoboschoolReacher-v1'

config.set_root('torch_rl_ppo_' + env_name.lower().split("-")[0], force=False)
config.configure_logging(clear=False, output_formats=['tensorboard', 'stdout'])
# config.start_tensorboard()

monitor = Monitor(EnvLogger(NormalisedActionsWrapper(gym.make(env_name))), 
    directory=os.path.join(config.root_path(), 'stats'), force=True, 
    video_callable=False, write_upon_reset=True)
env = RunningMeanStdNormalize(monitor)
print(env.observation_space.shape)



with tor.cuda.device(1):
    network = ActorCriticPPO([env.observation_space.shape[0], 64, 64, env.action_space.shape[0]])
    network.apply(xavier_uniform_init())

    trainer = GPUPPOTrainer(network=network, env=env, n_update_steps=5, 
        n_steps=2048, n_minibatches=32, lmda=.95, gamma=.99, lr=3e-4, epsilon=0.2, ent_coef=0.0)
    trainer.train(horizon=100000, max_episode_len=500)

