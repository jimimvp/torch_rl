import torch as tor
from torch.autograd import Variable
import numpy as np
import random
from collections import namedtuple, deque


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=tor.FloatTensor, cuda=True):
    if cuda:
        return cuda_if_available(Variable(
            tor.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
        ).type(dtype))
    return Variable(
        tor.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def cuda_if_available(model):
    if tor.cuda.device_count() > 0:
        return model.cuda()
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

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))




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


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#TODO Implement Hindsight Replay Memory
