from torch_rl.envs import GoalEnv, GoalEnvLogger, EnvLogger, RunningMeanStdNormalize
import gym
import roboschool
import torch_rl.envs
from torch_rl import config
from torch_rl.utils import logger
from tqdm import tqdm
import numpy as np
import argparse
from torch_rl.utils import *



addarg('--env', type=bool, default='OsimArm2D-v1')
p = parser.parse_args()


env_name = p.env
if 'Osim' in env_name:
    env = make_osim(env_name)
else:
    env = gym.make(env_name)
steps = 600

target_indices = [0,1]
curr_indices = [1,2]


# Configure logging, all data will automatically be saved to root_dir in the TRL_DATA_PATH
root_dir = 'random_' + str(env_name[:-3]).lower()

config.set_root(root_dir, force=True)
config.configure_logging(clear=False, output_formats=['tensorboard', 'json'])

env = GoalEnv(GoalEnvLogger(EnvLogger(env), target_indices=target_indices, curr_indices=curr_indices, precision=1e-2), target_indices=target_indices, curr_indices=curr_indices, precision=1e-2, sparse=True)
env.reset()

np.random.seed(666)

# Do 50000 random environment steps
for i in tqdm(range(steps)):
    done = False
    while not done:
        _,_,done,_ = env.step(env.action_space.sample())
    env.reset()
    logger.dumpkvs()

