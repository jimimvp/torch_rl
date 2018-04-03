import gym
import numpy as np
from torch_rl.envs.utils import wrapped_by
from torch_rl.utils.mpi_running_mean_std import RunningMeanStd



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



from torch_rl.envs.utils import potential_goal_indices

class GoalEnvWrapper(gym.RewardWrapper, NormalisedActionsWrapper):
    """
    Wrapper to create a goal based environment.
    """
    def __init__(self, *args, **kwargs):
        if "indices" in kwargs:
            self.indices = kwargs.get("indices")
            del kwargs['indices']
        else:
            self.indices = None

        if "separate" in kwargs:
            self.separate = kwargs["separate"]
        else:
            #By default separate the goal from the state
            self.separate  = True


        super(GoalEnvWrapper, self).__init__(*args, **kwargs)
        if self.indices is None:
            self.indices = potential_goal_indices(self)

    def reset(self, **kwargs):
        self._s = super(GoalEnvWrapper, self).reset(**kwargs)
        self._goal = self.observation_space.sample()[self.indices] if not hasattr(self, 'new_target') else self.new_target()
        return self._s

    def _step(self, action):
        info = super(GoalEnvWrapper, self)._step(action)
        if self.separate:
            self._s = info[0][not self.indices]
            self._goal = info[0][self.indices]
        else:
            self._s = info[0]
        info[0] = self._s

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
        if not wrapped_by(self.env, NormalisedObservationsWrapper):
            self.normalising_factor = self.observation_space.high - self.observation_space.low
        else:
            self.normalising_factor = 2.

    def _reward(self, reward):
        if np.any((np.abs(self._goal - self._s[self.indices])/self.normalising_factor) > self.precision):
            return 0
        else:
            return 1


class NormalisedRewardWrapper(gym.RewardWrapper):


    def _reward(self, reward):
        low, high = self.reward_range
        return ((reward-low)/(high-low)-0.5)*2



class ShapedRewardGoalEnv(GoalEnvWrapper):
    """
    Wrapper that creates sparse rewards 0 and 1 for the environment.
    """

    def __init__(self, *args, **kwargs):
        self.precision = kwargs.get("precision", 1e-2)
        del kwargs['precision']
        super(ShapedRewardGoalEnv, self).__init__(*args, **kwargs)
        if not wrapped_by(self.env, NormalisedObservationsWrapper):
            self.normalising_factor = self.observation_space.high - self.observation_space.low
        else:
            self.normalising_factor = 2.

    def _reward(self, reward):
        if np.any((np.abs(self._goal - self._s[self.indices])/self.normalising_factor[self.indices]) > self.precision):
            return -np.sum(np.abs(self._goal - self._s[self.indices])/self.normalising_factor[self.indices]/len(self.indices))
        else:
            return 1




class BaselinesNormalize(gym.Wrapper):
    """
        Normalization wrapper by running mean and standard deviation, as done in the OpenAI baselines
        implementation. This wrapper is made only for one environment, not a vector
        of environments as in the baselines implementation.
    """
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(BaselinesNormalize, self).__init__(env)
        self.env = env
        self.ob_rms = RunningMeanStd(shape=()) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, news, infos = self.env.step(action)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.std**2 + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def observation(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / (self.ob_rms.std + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.std**2 + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs




class OsimArmWrapper(gym.ObservationWrapper):
    """
    Wrapper that wraps the Stanford OpenSim environment to make it gym standard.
    """



    def _observation(self, observation):
        return observation[2:]




# import gym
# env = BaselinesNormalize(gym.make("MountainCar-v0"))
# env.reset()
# for i in range(10):
#     obs, _,_,_ = env.step(env.action_space.sample())
#     print(obs)






