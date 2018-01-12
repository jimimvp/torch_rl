import numpy as np


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
        self.done = np.alltrue(self.state == self.goal)

        return self.state, int(self.done), self.done, {}

    def reset(self):
        self.goal = np.random.choice([0,1], self.num_bits)
        self.state = np.zeros(self.num_bits)
        self.done = False

    def get_observation(self):
        return self.state

    def distance(self):
        return np.sum(np.abs(self.state-self.goal))



env = BitFlippingEnv(10)

# for i in range(10):
#     obs, rew, done, info = env.step(i)
#     print(obs, rew, done, info)
#
