import numpy as np
import gym

class Environment(object):


    def __init__(self):
***REMOVED***


    def step(self, action):
***REMOVED***


class BitFlippingEnv(object):


    def __init__(self, num_bits=10):
        self.num_bits = num_bits
        self.state = np.zeros(num_bits, dtype=np.uint8)
        # Maybe too hard?
        self.goal = np.random.choice([0,1], num_bits).astype(np.uint8)


    def step(self, action):
        if action < self.state.size:
            self.state[action] ^= 1
        self.done = np.array_equal(self.state, self.goal)

        return self.state, int(self.done), self.done, {}

    def reset(self):
        self.goal = np.random.choice([0,1], self.num_bits)
        self.state = np.zeros(self.num_bits, dtype=np.uint8)
        self.done = False

    def get_observation(self):
        return self.state

    def distance(self):
        return np.sum(np.abs(self.state-self.goal))




class NormalisedActions(gym.ActionWrapper):

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


