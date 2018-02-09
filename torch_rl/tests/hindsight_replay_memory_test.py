from torch_rl.memory import HindsightMemory
import gym
import numpy as np
from tqdm import tqdm
from unittest import TestCase
import pytest
import sys

def nnone(obj):
    return not obj is None

class HindsightReplayTest(TestCase):
    memory = HindsightMemory(1000, window_length=1)
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.episode = 1
        for i in range(500):
            if i % 20 == 0:
                cls.memory.append(cls.episode, 1, 1, 1, True)
                cls.episode += 1
    ***REMOVED***
                cls.memory.append(cls.episode, 1, 1, 1, False)
    def test_append(self):
        self.memory.append(0,1,1,1,1,False)

    def test_from_same_episode(self):
        self.assertTrue(self.episode > 1, "Episode has to be greater than 0")
        self.assertTrue(self.memory.nb_entries > 0, "Number of entries is not greater than 0")
        episode = self.memory.observations[0]
        for i in range(self.memory.nb_entries-1):
            if self.memory.observations[i] != episode:
                episode+=1
            self.assertTrue(self.memory.observations[i] == episode, "Episodes should come incrementally when iterating.")

    def test_hindsight(self):
        for hindsight in self.memory.hindsight_buffer:
            self.assertTrue(not hindsight is None, "If you can iterate over it, should not be None")
            # self.assertTrue(len(hindsight) == 8, "Length of every hindsight entry should be 8")
            # Check that all of the idxs belong to the same episode
            episode = self.memory.observations[hindsight[0][0]]
            for i,root_idx in hindsight[1:]:
                e = self.memory.observations[i]
                print(episode, e)
                self.assertTrue(self.memory.observations[i] == episode, "Every experience in hindsight should be from same episode")


    def test_hindsight_pairing(self):
        for hindsight in self.memory.hindsight_buffer:
            for i, root_idx in hindsight:
                e = self.memory[i]
                [self.assertTrue(nnone(obj), "Every object in experience should not be None") for obj in e]


if __name__ == '__main__':
    pytest.main([sys.argv[0]])
