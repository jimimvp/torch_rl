import os
import json
from collections import namedtuple
import torch_rl
from torch_rl.utils import compact_timestamp
import shutil
import numpy as np

module_path = os.path.abspath(os.path.dirname(torch_rl.__file__))


default_config={ 
    "paths": {
        "tensorboard" : "tensorboard",
        "logging" : "logs",
        "benchmark" : "benchmark",
        "video" : "video"
    }
}

default_config_path = os.path.join(module_path, 'default.config')

try:
    with open(default_config_path) as f:
        default_config = json.load(f)
except Exception as e:
    pass

def check_main_path_defined(func):
    def check(*args, **kwargs):
        assert 'TRL_DATA_PATH' in os.environ, \
        'TRL_DATA_PATH environment variable has to be defined to know where to store logs'
        return func(*args, **kwargs)

    return check

Paths = namedtuple('Paths', ['tensorboard', 'logging', 'benchmark', 'video'])


class Paths(object):


    def __init__(self, tensorboard, logging, benchmark, video):
        self.tensorboard = tensorboard
        self.benchmark = benchmark
        self.video = video
        self.logging = logging

    def __getitem__(self, key):
        """
            Create absoulte path from the config
        """
        return os.path.join(Config.CURRENT.target_path, getattr(self,key))





def configure_path(var, val):
 for k,v in Config.CURRENT.paths.items():
        Config.CURRENT.paths[k] = v.replace(var, val)


def merge_hierarchical_dicts(d1, dd):
    """
        Sets values to d1 from dd which are not set
    """

    if not isinstance(d1, dict) and not isinstance(dd, dict):
        #Arrived at a leaf
        return None
    elif isinstance(dd, dict)  ^ isinstance(dd, dict):
        raise Exception("Config dictionary formed wrong")

    else:
        for key, val in dd.items():
            if not key in d1:
                d1[key] = dd[key]
            else:
                merge_hierarchical_dicts(d1[key], dd[key])


def tensorboard_path():
    return Config.CURRENT.paths['tensorboard']

def logging_path():
    return Config.CURRENT.paths['logging']

def benchmark_path():
    return Config.CURRENT.paths['benchmark']

def video_path():
    return Config.CURRENT.paths['video']

def root_path():
    return Config.CURRENT.target_path  

@check_main_path_defined
def data_path():
    """
        Top level directory for storing torch_rl data.
    """
    return os.environ['TRL_DATA_PATH']



class Config(object):
    """
        Class keeps the directory tree structure for logging of
        a training session.
    """

    DEFAULT_DICT= default_config
    CURRENT = default_config

    def __init__(self, **kwargs):

        self.paths = Paths(**kwargs['paths'])


    @classmethod
    def __getitem__(cls, key):
        
        #If current config has this attribute, use it, else fall back to default config
        if hasattr(cls.CURRENT, key):
            return getattr(cls.CURRENT, key)
        else:
            raise Exception("Config key [{}] unknown".format(key))

import torch as tor
import numpy as np
import random

def set_global_seed(seed):
    np.random.seed(seed)
    tor.manual_seed(seed)
    random.seed(seed)
    Config.CURRENT.seed = seed


def load_config(config_path=default_config_path):
    
    with open(config_path) as f:
        config_dict = json.load(f)

    # Check which files are to be taken from default

    merge_hierarchical_dicts(config_dict, Config.DEFAULT_DICT)

    Config.CURRENT = Config(**config_dict)



@check_main_path_defined
def set_root(root, force=False):
    """
        Set the root to save all info regarding this training session to
        the specified path. If root is not an absolute path, then
        the directory is saved in TRL_DATA_PATH.
    """

    Config.CURRENT.target_path = root if os.path.isabs(root) else os.path.join(os.environ['TRL_DATA_PATH'], root)
    if force:
        if os.path.isdir(Config.CURRENT.target_path):
            shutil.rmtree(Config.CURRENT.target_path)
        os.makedirs(Config.CURRENT.target_path)
    elif os.path.isdir(Config.CURRENT.target_path):
        Config.CURRENT.target_path = Config.CURRENT.target_path + '_' + compact_timestamp()
        os.makedirs(Config.CURRENT.target_path)

Config.CURRENT = Config(**default_config)
set_root('training_data_' + compact_timestamp())


@check_main_path_defined
def configure_logging(clear=False, output_formats=['stdout'], force=False, root_dir=None):
    """
        Main method to configure logging. Supported output formats are
        tensorboard, stdout, csv, json. root_dir is the directory that
        will contain other outputs from logging as subdirs. 
    """
    from torch_rl.utils import logger
    if root_dir:
        set_root(root_dir, force=force)

    logger.configure(root_path(), clear=clear, output_formats=output_formats)


def start_tensorboard():
    from subprocess import Popen
    p = Popen(['tensorboard', '--logdir', tensorboard_path()])


set_global_seed(int(np.random.uniform(0, 100000)))

def get_global_seed():
    return Config.CURRENT.seed



