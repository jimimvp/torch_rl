from torch_rl.models.core import Policy, StochasticContinuousPolicy
from torch_rl.utils import gauss_weights_init
from torch import nn
from torch.distributions import Normal
import numpy as np
from torch_rl.utils import to_tensor
import torch as tor

class PPONetwork(StochasticContinuousPolicy):

    def __init__(self, architecture, weight_init=gauss_weights_init(0,0.02),activation_functions=None):
        super(PPONetwork, self).__init__()
        if len(architecture) < 2:
            raise Exception("Architecture needs at least two numbers to create network")
        assert architecture[-1]%2 == 1, "Last layer has to represent 2*actions_space for the Gaussian + 1 for value"
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
***REMOVED***
            for i, layer in enumerate(self.layer_list[:-1]):
                x = self.relu(layer(x))

        x = self.layer_list[-1](x)

        self.means = self.tanh(x[:,:int((x.shape[1]-1)/2)])
        self.sigmas = np.e ** self.tanh(x[:, int((x.shape[1]-1)/2):-1])
        self.dist = Normal(self.means, self.sigmas)
        self.value = x[:,-1]
        self.sampled = self.dist.rsample()
        x = self.sampled
        self.out = x
        return x


    def __call__(self, state):
        action = self.forward(state)
        return action, self.value


    def sigma(self):

        return self.sigmas

    def mu(self):
        return self.means

    def log_prob(self, values):
        return self.dist.log_prob(values)



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
