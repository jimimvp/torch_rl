import pytest
import unittest
import sys
import os
from torch_rl.models import SimpleNetwork


class ModelSavingTest(unittest.TestCase):


    def setup_class(cls):
        cls.model = SimpleNetwork([100, 10, 1])
        cls.save_path = "./model.ckpt"


    def test_saving_and_loading(cls):

        # Save the model
        cls.model.save(path=cls.save_path)
        assert os.path.isfile(cls.save_path), "Model should be stored in a file"

        # Load the model
        model = SimpleNetwork.load(cls.save_path)
        assert isinstance(model, SimpleNetwork), "Model should be loaded and of the same class type "

        # Remove the file created
        os.remove(cls.save_path)


if __name__ == '__main__':
    pytest.main([sys.argv[0]])

