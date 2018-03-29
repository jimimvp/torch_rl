from torch_rl.models.core import NeuralNet, StochasticContinuousNeuralNet
from torch_rl.utils import gauss_weights_init
from torch import nn
from torch.distributions import Normal, Uniform
import numpy as np
from torch_rl.utils import to_tensor, cuda_if_available
import torch as tor
from torch import nn


class PPONetwork(StochasticContinuousNeuralNet):


    def __init__(self, architecture, weight_init=gauss_weights_init(0,0.2),activation_functions=None):
        super(RandSigmaPPONetwork, self).__init__()
        if len(architecture) < 2:
            raise Exception("Architecture needs at least two numbers to create network")
        #assert architecture[-1]%2 == 1, "Last layer has to represent 2*actions_space for the Gaussian + 1 for value"
        self.activation_functions = activation_functions
        self.layer_list = []
        self.sigma_log = self.sigma_log_bounds[0]
        for i in range(len(architecture)-1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[i+1]))
            setattr(self, "fc" + str(i), self.layer_list[-1])

        self.apply(weight_init)
        self.siglog = nn.Parameter(self.siglog)



    def forward(self, x):
        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list[i](x))
        else:
            for i, layer in enumerate(self.layer_list[:-1]):
                x = self.relu(layer(x))

        x = self.layer_list[-1](x)

        self.means = self.tanh(x[:,:int((x.shape[1]-1))])
        # Choose sigmas randomly
        self.dist = Normal(self.means, self.siglog)
        self.value = x[:,-1]
        self.sampled = self.dist.rsample()
        x = self.sampled
        self.out = x
        return x


    def __call__(self, state, sigma_log):
        action = self.forward(state)
        self.sigma_log = sigma_log
        self.sigma_log = max(self.sigma_log, self.sigma_log_bounds[0])

        return action, self.value


    def sigma(self):

        return self.sigmas

    def mu(self):
        return self.means

    def log_prob(self, values):
        return self.dist.log_prob(values)


#Abbr.
class ActorCriticPPO(StochasticContinuousNeuralNet):


    def __init__(self, architecture, weight_init=gauss_weights_init(0,0.02),activation_functions=None):
        super(ActorCriticPPO, self).__init__()
        if len(architecture) < 2:
            raise Exception("Architecture needs at least two numbers to create network")
        #assert architecture[-1]%2 == 1, "Last layer has to represent 2*actions_space for the Gaussian + 1 for value"
        self.activation_functions = activation_functions
        self.layer_list = []
        self.layer_list_val = []
        self.siglog = tor.zeros(1, requires_grad=True)


        self.siglog = nn.Parameter(self.siglog)

        for i in range(len(architecture)-1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[i+1]))
            setattr(self, "fc" + str(i), self.layer_list[-1])

        for i in range(len(architecture) - 1):
            self.layer_list_val.append(nn.Linear(architecture[i], architecture[i + 1]))
            setattr(self, "fc_val" + str(i), self.layer_list_val[-1])

        #self.last_value_layer = nn.Linear(architecture[-1], 1)

        self.apply(weight_init)


    def policy_forward(self, x):

        # Policy network

        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list[i](x))
        else:
            for i, layer in enumerate(self.layer_list[:-1]):
                x = self.tanh(layer(x))

        x = self.layer_list[-1](x)

        self.means = self.tanh(x)
        self.dist = Normal(self.means, tor.exp(self.siglog))

        self.sampled = self.dist.rsample()
        x = self.sampled

        return x


    def value_forward(self, x):


        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list_val[i](x))
        else:
            for i, layer in enumerate(self.layer_list_val[:-1]):
                x = self.tanh(layer(x))

        x = self.layer_list_val[-1](x)

        return x

    def forward(self, x):

        # Policy network
        action = self.policy_forward(x)
        value = self.value_forward(x)

        return tor.cat([action, value], dim=1)


    def __call__(self, state):
        #self.sigma_log -= sigma_epsilon
        action, value = self.policy_forward(state), self.value_forward(state)

        return action, value


    def sigma(self):

        return self.sigmas

    def mu(self):
        return self.means

    def logprob(self, values):
        return self.dist.log_prob(values)

    def entropy(self):
        return self.dist.entropy()




#
# #Test
# import numpy as np
# architecture = [6, 11]
# gauss_policy = PPONetwork(architecture=architecture)
#
# loss = 0
# for i in range(100):
#     res = gauss_policy.forward(to_tensor(np.random.normal(3,1,(5,6)), cuda=False))
#     sigma = gauss_policy.sigma()
#     mean = gauss_policy.mu()
#     loss += tor.sum(res)
#     print(sigma, mean)
#
# loss.backward()
