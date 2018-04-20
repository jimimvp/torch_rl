import gym
import numpy as np
from torch_rl.envs.utils import wrapped_by
from torch_rl.utils.mpi_running_mean_std import RunningMeanStd
from gym import spaces



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

from torch_rl.utils import logger

class GoalEnv(gym.Wrapper):
    """
    Wrapper that creates sparse rewards 0 and 1 for the environment or a distance
    reward in case we want it shaped.
    """

    def __init__(self, env, target_indices, curr_indices, precision=1e-2, sparse=False, log=False):
        super(GoalEnv, self).__init__(env)
        assert len(target_indices) == len(curr_indices)

        self.target_indices = target_indices
        self.curr_indices = curr_indices
        self.precision = precision
        self.sparse = sparse
        self.log = log

    def step(self, action):
        obs, reward, done, inf = self.env.step(action)
        sparse_reward = float(np.allclose(obs[self.curr_indices], obs[self.target_indices], atol=self.precision))-1.
        if self.sparse:
            reward = sparse_reward
        else:
            reward = .5 * np.mean((obs[self.curr_indices] - obs[self.target_indices])**2)
            
        return obs, reward, done, inf


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



class ShrinkEnvWrapper(gym.ObservationWrapper):
    """
        The wrapper takes indices of the observation that are going to be used
        in the end and hides the rest i.e. forcing a POMDP process.
    """

    def __init__(self, env, indices):
        super(ShrinkEnvWrapper, self).__init__(env)
        self.indices = indices
        self.observation_space = spaces.Box(env.observation_space.high[indices], env.observation_space.low[indices])


    def observation(self, obs):
        return obs[self.indices]




if __name__ == '__main__':


    env = ShrinkEnvWrapper(gym.make('Pendulum-v0'), indices=[0,1])
    obs = env.reset()
    print('Observation space: ', env.observation_space.shape)
    print('Observation: ', obs.shape)
    assert env.observation_space.shape[0] == 2 and obs.shape[0] == 2, 'Should be shrinked to shape of 2'
    env.step(env.action_space.sample())

from multiprocessing import Process, Queue, Pipe
from time import sleep

class EnvProcess(Process):

    def __init__(self, env, dt=1e-2):
        self.queue = Queue()
        self.info_pipe, info_pipe_child = Pipe()
        self.reset_pipe, reset_pipe_child = Pipe()
        self.env = env
        super(EnvProcess, self).__init__(target=EnvProcess.env_loop, args=(env, 
            self.queue, info_pipe_child, reset_pipe_child, dt))
        self.start()

    @staticmethod
    def env_loop(env, action_queue, info_pipe, reset_pipe, dt):

        action = np.zeros_like(env.action_space.sample())
        state = env.reset()
        info_pipe.send((state, 0, False, {}))
        while(True):
            action = action_queue.get() if not action_queue.empty() else action
            inf = env.step(action)
            info_pipe.send(inf)
            sleep(dt)
            if reset_pipe.poll(1e-4):
                reset_pipe.recv()
                state = env.reset()
                reset_pipe.send(state)
                action = np.zeros_like(env.action_space.sample())


    def reset(self):
        self.reset_pipe.send(True)
        return self

    def step(self, action):
        if not action is None:
            self.queue.put(action)
        return self.info_pipe.recv()



class AsyncEnvWrapper(gym.Wrapper):
    """
        Creates an asynchronous environment. The environment
        reacts instantly when receiving the action but also
        reacts on the last action received when not receiving
        the action.
    """
    def __init__(self, env, dt=1e-2):
        super(AsyncEnvWrapper, self).__init__(env)
        # Queue for sending actions to the process
        self.env_process = EnvProcess(env, dt)



    def step(self, action):

        return self.env_process.step(action)

    def reset(self):

        return self.env_process.reset()




# import gym
# env = AsyncEnvWrapper(gym.make("MountainCar-v0"))
# env.reset()
# for i in range(100):
#     obs, _,_,_ = env.step(env.action_space.sample())
#     print(obs)






