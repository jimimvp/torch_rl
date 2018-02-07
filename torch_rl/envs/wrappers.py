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
    """
    def __init__(self,*args,**kwargs):
        if "indices" in kwargs:
            self.indices = kwargs.get("indices")
            del kwargs['indices']
***REMOVED***
            self.indices = None
        super(GoalEnvWrapper, self).__init__(*args, **kwargs)
        if self.indices is None:
            self.indices = np.arange(0, self.observation_space.shape[0])

    def reset(self, **kwargs):
        self._s = super(GoalEnvWrapper, self).reset(**kwargs)
        self._goal = self.observation_space.sample()[self.indices]
        return self._s

    def _step(self, action):
        info = super(GoalEnvWrapper, self)._step(action)
        self._s = info[0]
        return info

    def _reward(self, reward):
        return - np.mean(np.abs(self._goal - self._s[self.indices]))

    @property
    def goal(self):
        return self._goal


class SparseRewardGoalEnv(GoalEnvWrapper):
    """
    Wrapper that creates sparse rewards 0 and 1 for the environment.
    """

    def __init__(self, *args, **kwargs):
        self.precision = kwargs.get("precision", 1e-2)
        del kwargs['precision']
        super(SparseRewardGoalEnv, self).__init__(*args,**kwargs)
        self.normalising_factor = self.observation_space.high - self.observation_space.low

    def _reward(self, reward):
        if np.any((np.abs(self._goal - self._s[self.indices])/self.normalising_factor) > self.precision):
            return 0
***REMOVED***
            return 1


