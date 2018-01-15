import torch as tor
from torch import nn
import numpy as np
from utils import gauss_weights_init


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def action(self, x):
***REMOVED***

class StochasticPolicy(Policy):

    def __init__(self):
        super(StochasticPolicy, self).__init__()

    def sample_action(self, action_distribution=None):

        if not action_distribution:
            action_distribution = self.out[0]
        action_distribution = action_distribution.data.numpy()
        action = np.random.choice(action_distribution, p=action_distribution)
        action = np.argmax(action_distribution == action)
        return action




class SimpleNetwork(Policy):

    def __init__(self, architecture, weight_init=gauss_weights_init(0,0.02),activation_functions=None):
        super(SimpleNetwork, self).__init__()
        if len(architecture) < 2:
            raise Exception("Architecture needs at least two numbers to create network")

        self.activation_functions = activation_functions
        self.layer_list = []
        self.fc0 = nn.Linear(architecture[0], architecture[1])
        for i in range(1, len(architecture)-1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[i+1]))
            setattr(self, "fc" + str(i), self.layer_list[-1])

        self.apply(weight_init)

    def forward(self, x):
        out = x
        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                out = self.activation_functions[i](self.layer_list[i](out))
***REMOVED***
            if self.activation_functions:
                for i, func in enumerate(self.activation_functions):
                    x = self.relu(self.layer_list[i](x))

        self.out = out
        return out



class DDPGCritic(nn.Module):

    def __init__(self):
***REMOVED***



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


