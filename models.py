import torch as tor
from torch import nn
import numpy as np



class StochasticPolicy(nn.Module):

    def __init__(self):
        super(StochasticPolicy, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def sample_action(self, action_distribution=None):

        if not action_distribution:
            action_distribution = self.out[0]
        action_distribution = action_distribution.data.numpy()
        action = np.random.choice(action_distribution, p=action_distribution)
        action = np.argmax(action_distribution == action)
        return action




class PolicyAHG(StochasticPolicy):

    def __init__(self,  input_size, output_size):
        super(PolicyAHG, self).__init__()

        self.f1 = nn.Linear(input_size,32)
        self.f2 = nn.Linear(32, output_size)


    def forward(self, x):

        out = self.f1(x)
        out = self.tanh(out)
        out = self.f2(out)
        out = self.softmax(out)
        self.out = out
        return out



class PolicySPG(StochasticPolicy):

    def __init__(self,  input_size, output_size):
        super(PolicySPG, self).__init__()

        self.f1 = nn.Linear(input_size,32)
        self.f2 = nn.Linear(32, output_size)


    def forward(self, x):

        out = self.f1(x)
        out = self.relu(out)
        out = self.f2(out)
        out = self.softmax(out)
        self.out = out
        return out

