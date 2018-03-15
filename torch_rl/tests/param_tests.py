from torch_rl.utils import ParameterGrid
import pytest
import unittest
import sys

class TestParameterGrid(unittest.TestCase):


    def setup_class(cls):
        cls.grid = ParameterGrid.from_config('grid.json')


    def test_from_config(self):

        grid = ParameterGrid.from_config('grid.json')


    def test_iter(self):
        for parameters in self.grid:
            pass
        assert False in self.grid.reservoir
        assert 400 in self.grid.reservoir




if __name__ == '__main__':
    pytest.main([sys.argv[0]])

