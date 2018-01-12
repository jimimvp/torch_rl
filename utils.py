import torch as tor
from torch.autograd import Variable
import numpy as np

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=tor.FloatTensor):
    return Variable(
        tor.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)