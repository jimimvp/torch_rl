from torch.nn import Module
import torch as tor
import numpy as np
from torch_rl.utils import to_tensor
import numpy as np
from torch.nn import Module

from torch_rl.utils import to_tensor


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
        print("Agent initialized...")

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
        else:
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
            if len(args) > 1:
                if not tor.is_tensor(args[0]):
                    x = np.hstack(args)
                    x = to_tensor(x, requires_grad=requires_grad)
                else:
                    x = tor.cat(args, 1)
            out = self.forward(x)
            self.out = out
            return out
        else:
            x = args[0]
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.forward(x)
            self.out = out
            return out


    def choose_action(self, *args):

        action = self.action(*args)
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
        else:
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
            if not tor.is_tensor(args[0].data ):
                x = np.hstack(args)
                x = to_tensor(x, requires_grad=requires_grad)
            else:
                x = tor.cat(args, 1)
            out = self.critic_forward(x)
            self.out = out
            return out
        else:
            x = args[0]
            x = to_tensor(x, requires_grad=requires_grad)
            out = self.forward(x)
            self.out = out
            return out



class SpikingNetwork(object):
    """
       A similar interface to pytorch for easier plug-in.
    """


    def __init__(self, dt, sim_dt):
        super(SpikingNetwork, self).__init__()
        self.dt = dt
        self.sim_dt = sim_dt

    def forward(self, x):
        raise NotImplementedError("Forward pass has to be implemented for spiking network")



