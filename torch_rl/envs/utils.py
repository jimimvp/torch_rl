


def wrapped_by(env, cls):
    """
    Checks if the environment is wrapped by a given wrapper
    :param env: The env
    :param cls: The wrapping class
    :return:
    """
    if isinstance(env, cls):
        return True

    while hasattr(env, "env"):
        env = env.env
        if isinstance(env, cls):
            return True


def env_info(env):
    spacing = 40
    top = "#"*spacing, "ENV_INFO", "#"*spacing
    bottom = "#"*len(top)
    print(top)
    print("Observation space high/low: ", env.observation_space.low, env.observation_space.high)
    print("Action space high/low: ", env.observation_space.low, env.observation_space.high)
    env.reset()
    print("Few steps:")
    for i in range(5):
        print(env.step(env.action_space.sample()))
    print(bottom)



import numpy as np
import gym

def potential_goal_indices(env):
    """
        Function does 100 environment steps and finds indices that dont change, those
        are potential goals of the environment.
    """

    obs = env.reset()

    #We should come to zero in the obs array on the end
    for i in range(99):
        obs_next = env.step(env.action_space.sample())[0]
        obs = obs - obs_next if i%2 == 0 else obs + obs_next

    return np.isclose(obs, 0, atol=1e-6)


