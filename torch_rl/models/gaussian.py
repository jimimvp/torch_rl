from torch_rl.models.core import NeuralNet, StochasticContinuousNeuralNet
from torch_rl.utils import gauss_weights_init
from torch import nn
from torch.distributions import Normal
import torch as tor
from torch_rl.utils import to_tensor

class GaussianPolicy(StochasticContinuousNeuralNet):

    def __init__(self, architecture, weight_init=gauss_weights_init(0,0.02),activation_functions=None):
        super(GaussianPolicy, self).__init__()
        if len(architecture) < 2:
            raise Exception("Architecture needs at least two numbers to create network")

        self.activation_functions = activation_functions
        self.layer_list = []
        for i in range(len(architecture)-1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[i+1]))
            setattr(self, "fc" + str(i), self.layer_list[-1])

        self.apply(weight_init)

    def forward(self, x):
        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list[i](x))
        else:
            for i, layer in enumerate(self.layer_list[:-1]):
                x = self.relu(layer(x))

        x = self.layer_list[-1](x)

        #Last layer is partially tanh and partially softmax

        self.means = self.tanh(x[None,:int(x.shape[1]/2)])
        self.sigmas = self.softmax(x[None, int(x.shape[1]/2):])
        self.dist = Normal(self.means, self.sigmas)

        self.sampled = self.dist.rsample()
        x = self.sampled
        self.out = x
        return x


    def sigma(self):

        return self.sigmas

    def mu(self):
        return self.means

    def log_prob(self, values):
        return self.dist.log_prob(values)
