import torch as tor
from torch.autograd import Variable
import numpy as np
import random
from collections import namedtuple, deque

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=tor.FloatTensor):
    return Variable(
        tor.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def gauss_weights_init(mu, std):

    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mu, std)

    return init


def to_input_state_goal(state, goal):

    x = np.hstack((state,goal))
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