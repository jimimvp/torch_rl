from torch_rl.models.core import Policy, StochasticContinuousPolicy
from torch_rl.utils import gauss_weights_init
from torch import nn
from torch.distributions import Normal
import torch as tor
from torch_rl.utils import to_tensor

class GaussianPolicy(StochasticContinuousPolicy):

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
***REMOVED***
            for i, layer in enumerate(self.layer_list):
                x = self.relu(layer(x))

        i+=1
        while i < len(self.layer_list):
            x = self.layer_list[i](x)
            i+=1


        self.dist_parameters = x

        res = []

        self.gauss_dists = tor.stack([Normal(x,y) for x,y in x.view(-1,2)])

        for dist in self.gauss_dists:
            res.append(dist.rsample())
        self.sampled = tor.stack(res, -1)
        self.sampled = self.sampled.view(-1, x.shape[1] / 2)
        x = self.sampled
        self.out = x
        return x






# Test
#import numpy as np
#architecture = [100, 10]
#gauss_policy = GaussianPolicy(architecture=architecture)

#loss = 0
#for i in range(20):
#    res = gauss_policy.forward(to_tensor(np.random.normal(0,1,(5,100)), cuda=False))
#    loss += tor.sum(res)
#    print(res.data.cpu())

#loss.backward()




