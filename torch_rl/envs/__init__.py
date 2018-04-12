from gym.envs.registration import register
from .wrappers import *
from .logger import *
from .envs import *
from .spiking import *
from .roboschool_envs import *

register(
    id='BanditsX2-v0',
    kwargs = {'num_bandits' : 2},
    entry_point='torch_rl.envs:BanditEnv',
)


register(
    id='BanditsX4-v0',
    kwargs = {'num_bandits' : 4},
    entry_point='torch_rl.envs:BanditEnv',
)

register(
    id='BanditsX8-v0',
    kwargs = {'num_bandits' : 8},
    entry_point='torch_rl.envs:BanditEnv',
)



register(
    id='TRLRoboschoolReacher-v1',
    kwargs = {},
    entry_point='torch_rl.envs:RoboschoolReacher',
    max_episode_steps=150,
    reward_threshold=18.0,
    tags={ "pg_complexity": 1*1000000 },
)

