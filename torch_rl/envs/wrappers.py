import gym
import numpy as np

class NormalisedActionsWrapper(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class NormalisedObservationsWrapper(gym.ObservationWrapper):

    def _observation(self, observation):
        observation -= self.observation_space.low
        observation /= (self.observation_space.high - self.observation_space.low)
        observation = observation*2-1
        return observation


class GoalEnvWrapper(gym.RewardWrapper):
    """
    Wrapper to create a goal based environment.
    TODO add posibility to take part of the observation as a goal, this makes more sense
    """

    def __init__(self,*args,**kwargs):
        super(GoalEnvWrapper, self).__init__(*args, **kwargs)

    def reset(self, **kwargs):
        self._s = super(GoalEnvWrapper, self).reset(**kwargs)
        self._goal = self.observation_space.sample()
        return self._s

    def _step(self, action):
        info = super(GoalEnvWrapper, self)._step(action)
        self._s = info[0]
        return info

    def _reward(self, reward):
        return - np.mean(np.abs(self._goal - self._s))

    @property
    def goal(self):
        return self._goal


class SparseRewardGoalEnv(GoalEnvWrapper):
    """
    Wrapper that creates sparse rewards 0 and 1 for the environment.
    """

    def __init__(self, *args, **kwargs):
        super(SparseRewardGoalEnv, self).__init__(*args,**kwargs)
        #TODO add precision, currently it is hardcoded to 1e-3 %
        self.normalising_factor = self.observation_space.high - self.observation_space.low
        self.precision = 1e-3

    def _reward(self, reward):
        if np.any(np.abs(self._goal - self._s)/self.normalising_factor > 1e-3):
            return 0
***REMOVED***
            return 1


