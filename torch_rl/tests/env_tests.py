import gym
import torch_rl.envs
import unittest
from unittest import TestCase
import numpy as np

class TestBanditEnv(TestCase):

    def setUp(self):
        self.env = gym.make('BanditsX2-v0')

    def testStep(self):
        obs, reward, done, info = self.env.step(self.env.action_space.sample())
        self.assertTrue(not obs is None)
        self.assertTrue(not reward is None)

    def testReset(self):
        state = self.env.reset()
        self.assertTrue(isinstance(state, np.ndarray), "State is instance numpy array")
        self.assertTrue(state.shape[0] == self.env.observation_space.shape[0], "State shape is equal to observation space shape")


