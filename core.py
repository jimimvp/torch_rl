import torch as tor
from torch.nn import Module
import numpy as np
from utils import to_tensor

def return_output(action):
    return action


class Agent(Module):
    """
        Wrapper for neural networks for more convenient use
        in RL environments.
    """

    def __init__(self, policy_network, action_choice_function=return_output):
        super(Agent, self).__init__()
        self.policy_network = policy_network
        self.action_choice_function = action_choice_function

    def forward(self, x):

        out = self.policy_network.forward(x)
        return out

    def action(self, *args, requires_grad=False):
        """
        To be called in non-batch call.
        :param args: List of inputs to the network
        :return: Output of the network in non-batch shape
        """

        if len(args) > 1:
            x = np.hstack(args)
            x = np.expand_dims(x, 0)
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.forward(x)
            self.out = out[0]
            return out[0]
***REMOVED***
            x = args[0]
            x = np.expand_dims(x, 0)
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.forward(x)
            self.out = out[0]
            return out[0]


    def actions(self, *args, requires_grad=False):
        """
        To be called in a batch call.
        :param args: List of inputs to the network
        :return: Output of the network in batch shape
        """
        if len(args) > 1:
            x = np.hstack(args)
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.forward(x)
            self.out = out
            return out
***REMOVED***
            x = args[0]
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.forward(x)
            self.out = out
            return out


    def choose_action(self, x):

        action = self.forward(x)
        action = self.action_choice_function(action)
        return action



class ActorCriticAgent(Agent):

    def __init__(self, actor_network, critic_network, action_choice_function=return_output):
        super(ActorCriticAgent, self).__init__(actor_network, action_choice_function=action_choice_function)
        self.critic_network = critic_network

    def critic_forward(self, x):
        critic_out = self.critic_network.forward(x)
        return critic_out

    def value(self, *args, requires_grad=False):
        """
        To be called in non-batch call.
        :param args: List of inputs to the network
        :return: Output of the network in non-batch shape
        """

        if len(args) > 1:
            x = np.hstack(args)
            x = np.expand_dims(x, 0)
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.critic_forward(x)
            self.out = out[0]
            return out[0]
***REMOVED***
            x = args[0]
            x = np.expand_dims(x, 0)
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.critic_forward(x)
            self.out = out[0]
            return out[0]


    def values(self, *args, requires_grad=False):
        """
        To be called in a batch call.
        :param args: List of inputs to the network
        :return: Output of the network in batch shape
        """
        if len(args) > 1:
            x = np.hstack(args)
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.critic_forward(x)
            self.out = out
            return out
***REMOVED***
            x = args[0]
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.forward(x)
            self.out = out
            return out



class***REMOVED***Network(object):
    """
       A similar interface to pytorch for easier plug-in.
    """


    def __init__(self, dt, sim_dt):
        super(SpikingNetwork, self).__init__()
        self.dt = dt
        self.sim_dt = sim_dt

    def forward(self, x):
        raise NotImplementedError("Forward pass has to be implemented for spiking network")



