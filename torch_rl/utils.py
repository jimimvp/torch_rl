import torch as tor
from torch.autograd import Variable
import numpy as np
import random
from collections import namedtuple, deque
import os



def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=tor.FloatTensor, cuda=True):
    if cuda:
        return cuda_if_available(Variable(
            tor.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
        ).type(dtype))
    return Variable(
        tor.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def cuda_if_available(model):
    if tor.cuda.is_available():
        try:
            return model.cuda()
        except Exception as e:
             #prRed("GPU memory allocation failed, using CPU.")
            return model
    else:
        return model


def gauss_weights_init(mu, std):
    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mu, std)

    return init


def gauss_init(mu, std):
    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mu, std)
            m.bias.data.normal_(mu, std)

    return init


def to_input_state_goal(state, goal):
    x = np.hstack((state, goal))
    x = np.expand_dims(x, 0)
    return to_tensor(x)


def to_numpy(out):
    return out.data.numpy()


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


"""
    Color printing functions.
"""

def prRed(prt, flush=False): print("\033[91m {}\033[00m" .format(prt), flush=flush)
def prGreen(prt, flush=False): print("\033[92m {}\033[00m" .format(prt), flush=flush)
def prYellow(prt, flush=False): print("\033[93m {}\033[00m" .format(prt),flush=flush)
def prLightPurple(prt, flush=False): print("\033[94m {}\033[00m" .format(prt), flush=flush)
def prPurple(prt, flush=False): print("\033[95m {}\033[00m" .format(prt), flush=flush)
def prCyan(prt, flush=False): print("\033[96m {}\033[00m" .format(prt), flush=flush)
def prLightGray(prt, flush=False): print("\033[97m {}\033[00m" .format(prt), flush=flush)
def prBlack(prt, flush=False): print("\033[98m {}\033[00m" .format(prt), flush=flush)

def loop_print(prt, r):
    for i in r:
        print("\r\033[96m {}\033[00m".format(prt.format(i+1, len(r))), end="", flush=True)
        yield i






# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

    def __call__(self, *args, **kwargs):
        return self.sample()


import json

class Parameters(object):

    def __init__(self):
        super(Parameters, self).__init__()

    def to_named_tuple(self):
        d = self.__dict__
        return namedtuple('Parameters', d.keys())(**self.__dict__)

    @classmethod
    def from_config(cls, path):
        with open(path, "r") as f:
            config = f.read()
            config_dict = json.laods(config)
            p = Parameters.from_args(config_dict)
        return p

    @classmethod
    def from_args(cls, args):
        p = Parameters()
        d = vars(args)
        for k, i in d.items():
            setattr(p, k, i)
        return p

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(vars(self), f)



class Callback(object):

    def step(self, *args, **kwargs):
***REMOVED***

    def episode_step(self, *args, **kwargs):
***REMOVED***

