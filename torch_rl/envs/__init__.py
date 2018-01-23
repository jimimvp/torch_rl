from gym.envs.registration import register

from .envs import *

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