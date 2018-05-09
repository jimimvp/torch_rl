from torch import nn
from torch_rl.utils import gauss_weights_init

from torch_rl.core import *
import os
import glob

class SaveableModel(object):


    def save(self, path):
        tor.save(self, path)

    @classmethod
    def load(cls, path):
        return tor.load(path)


    @classmethod
    def load_best(cls, path):
        assert os.path.isdir(path)

        best_models = glob.glob(os.path.join(path, "*best*"))

        assert not len(best_models > 1)

        return tor.load(os.path.join(path, best_models[0]))


class NeuralNet(nn.Module, SaveableModel):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def action(self, x):
        pass




class StochasticNeuralNet(NeuralNet):

    def __init__(self):
        super(StochasticNeuralNet, self).__init__()

    def sample_action(self, action_distribution=None):
        if not action_distribution:
            action_distribution = self.out
        action_distribution = action_distribution.cpu().data.numpy()
        action = np.random.choice(action_distribution.squeeze(), p=action_distribution.squeeze())
        action = np.argmax(action_distribution == action)
        return action


class StochasticContinuousNeuralNet(NeuralNet):


    def __init__(self):
        super(StochasticContinuousNeuralNet, self).__init__()


    def sigma(self):
        pass

    def mu(self):
        pass




class SimpleNetwork(NeuralNet):

    def __init__(self, architecture, weight_init=gauss_weights_init(0,0.02),activation_functions=None):
        super(SimpleNetwork, self).__init__()
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
            for i, layer in enumerate(self.layer_list):
                x = self.relu(layer(x))

        i+=1
        while i < len(self.layer_list):
            x = self.layer_list[i](x)
            i+=1

        self.out = x
        return x


class QNetwork(NeuralNet):
    """
        Just adds a call method for simpler state and action passing.
    """

    def __init__(self, architecture, weight_init=gauss_weights_init(0,0.02),
            activation_functions=None):
        super(NeuralNet, self).__init__()
        self.activation_functions = activation_functions
        self.layer_list = []
        for i in range(len(architecture)-1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[i+1]))
            setattr(self, "fc" + str(i), self.layer_list[-1])

        #self.last_linear = nn.Linear(architecture[-1], 1)
        self.apply(weight_init)

    def forward(self, x):
        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list[i](x))
        else:
            for i, layer in enumerate(self.layer_list):
                x = self.relu(layer(x))

        i+=1
        while i < len(self.layer_list):
            x = self.layer_list[i](x)
            i+=1

       # x = self.last_linear(x)

        return x


 
    def __call__(self, s, a):

        x = tor.cat((s,a), 1)
        return self.forward(x)


class PolicyAHG(StochasticNeuralNet):

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



class PolicySPG(StochasticNeuralNet):

    def __init__(self,  input_size, output_size):
        super(PolicySPG, self).__init__()

        self.f1 = nn.Linear(input_size,64)
        self.f2 = nn.Linear(64, output_size)


    def forward(self, x):

        out = self.f1(x)
        out = self.relu(out)
        out = self.f2(out)
        out = self.softmax(out)
        self.out = out
        return out

