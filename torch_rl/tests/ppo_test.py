from unittest import TestCase
import pytest
import gym
import sys
import numpy as np

class PPOTest(TestCase):
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.episode = 1
        cls.T = 10
        cls.env = gym.make('Pendulum-v0')
        cls.gamma = .99
        cls.lmda = .95

        cls.def_reward = 1.
        cls.def_value = 4.


    def test_advantage_calculation(self):

        values = np.zeros(self.T)
        advantages = np.zeros(self.T)
        rewards = np.zeros(self.T)
        returns = np.zeros(self.T)

        for i in range(self.T):

            rewards[i] = self.def_reward
            values[i] = self.def_value

            lastgaelem = 0
            for j in reversed(range(i)):
                td = self.def_reward + values[j+1]*self.gamma - values[j]
                A = lastgaelem = self.lmda * self.gamma * lastgaelem + td
                advantages[j] = A
                returns[j] = A + values[j]

        print(advantages)
        print(returns)




if __name__ == '__main__':
    pytest.main([sys.argv[0]])
