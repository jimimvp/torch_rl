import torch as tor
from torch.autograd import Variable
import numpy as np
from collections import namedtuple
import datetime

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


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def compact_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")



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
            config_dict = json.loads(config)
            p = Parameters.from_args(config_dict)
        return p

    @classmethod
    def from_args(cls, args):
        p = Parameters()
        d = args
        for k, i in d.items():
            setattr(p, k, i)
        return p

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(vars(self), f)


import itertools

def cart_product(d, i=0):
    """
    Generator function for cartesian product of dictionary when each entry in key is a list.
    :param d: Dictionary
    :param i: Index of key that is being iterated over
    :return: Dictionary of parameters
    """
    # If iterated over all parameters, return empty dict
    if i >= len(d.keys()):
        yield {}
        return None

    k = list(d.keys())[i]

    for p in d[k]:
        # Combine this parameter with another parameter
        p = {k : p}
        for p_next in cart_product(d, i+1):
            yield {**p, **p_next}




class ParameterGrid(Parameters):

    @classmethod
    def from_args(cls, args):
        p = ParameterGrid()
        d = args
        setattr(p, "grid", d)
        setattr(p, "cart_product", list(cart_product(p.grid)))
        return p

    @classmethod
    def from_config(cls, path):
        with open(path, "r") as f:
            config = f.read()
            config_dict = json.loads(config)
            p = ParameterGrid.from_args(config_dict)
        return p

    def __iter__(self):
        """
        Iterate over parameter grid
        :return:
        """
        for params in self.cart_product:
            yield Parameters.from_args(params)

    def __len__(self):
        return len(self.cart_product)



class Callback(object):

    def __init__(self, episodewise=True, stepwise=False):
        self.stepwise=stepwise
        self.episodewise = episodewise

    def step(self, *args, **kwargs):
        if self.stepwise:
            self._step()

    def episode_step(self, *args, **kwargs):
        if self.episodewise:
            self._episode_step(*args, **kwargs)

    def _step(self, *args, **kwargs):
        pass

    def _episode_step(self, *args, **kwargs):
        pass



def gauss_weights_init(mu, std):
    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mu, std)

    return init


def uniform_init(l=-1, r=1):
    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.uniform_(l, r)

    return init


def xavier_normal_init(gain=1.):

    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            tor.nn.init.xavier_normal(m.weight.data, gain)
            m.bias.data.zero_()


def xavier_uniform_init():

    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            tor.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()

    return init


def gauss_init(mu, std):
    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mu, std)
            m.bias.data.normal_(mu, std)

    return init



import argparse
cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)



class UniversalParser(object):
    """
        Store everything as a string except bool arguments
        so that you can evaluate them
    """
    parser = cparser
    def __init__(self, parser):
        self.parser = parser

    def parse_args(self):
        self.args = self.parser.parse_args()
        for arg in var(self.args):
            try:
                setattr(arg, eval(str(getattr(self.args, arg))))
            except TypeError as err:
                setattr(arg, getattr(self.args, arg))

        return self.args

    def add_argument(self, name, default):
        if isinstance(default, bool):
            self.parser.add_argument('--'+name, default=default, action='store_false' if default else 'store_true')
        else:
            print('Adding ' , name)
            self.parser.add_argument('--'+name, default=str(default), type=str)

def addarg(name, type, default=None, info=None):
    cparser = UniversalParser.parser

    if isinstance(default, list):
        cparser.add_argument('--'+name, default=default, nargs='+', type=type, help=info)
    elif type == bool and not default:
        cparser.add_argument('--'+name, default=default, action='store_true', help=info)
    elif type == bool and default:
        cparser.add_argument('--'+name, default=default, action='store_false', help=info)
    else:
        cparser.add_argument('--'+name, default=default, type=type, help=info)


def init_parser(description=""):

    UniversalParser.parser = argparse.ArgumentParser(description=description)

def cmdl_args():
    return UniversalParser.parser.parse_args()

def adduarg(name, default=None):
    parser = UniversalParser.parser
    addarg(name, default=str(default), type=str)


#Init default parser without description
init_parser()
parser = UniversalParser.parser



