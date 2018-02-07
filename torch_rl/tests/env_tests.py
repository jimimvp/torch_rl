import gym
import torch_rl.envs
import unittest
from unittest import TestCase
import numpy as np
import pytest
import sys

from torch_rl.envs import SparseRewardGoalEnv

class TestBanditEnv(TestCase):

    @classmethod
    def setup_class(cls):
        cls.env = gym.make('BanditsX2-v0')

    def test_step(self):
        obs, reward, done, info = self.env.step(self.env.action_space.sample())
        self.assertTrue(not obs is None)
        self.assertTrue(not reward is None)

    def test_reset(self):
        state = self.env.reset()
        self.assertTrue(isinstance(state, np.ndarray), "State is instance numpy array")
        self.assertTrue(state.shape[0] == self.env.observation_space.shape[0], "State shape is equal to observation space shape")


class TestSparseRewardWrapper(TestCase):

    @classmethod
    def setup_class(cls):
        cls.env = gym.make("MountainCarContinuous-v0")
        cls.env = SparseRewardGoalEnv(cls.env)

    def test_step(self):
        obs, reward, done, info = self.env.step(self.env.action_space.sample())
        self.assertTrue(not obs is None)
        self.assertTrue(not reward is None)
        self.assertTrue(reward in [0,1])

    def test_reset(self):
        state = self.env.reset()
        self.assertTrue(isinstance(state, np.ndarray), "State is instance numpy array")
        self.assertTrue(state.shape[0] == self.env.observation_space.shape[0], "State shape is equal to observation space shape")
        self.assertTrue(not self.env.goal is None)


if __name__ == '__main__':
    pytest.main([sys.argv[0]])