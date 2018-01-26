from torch import nn
from torch_rl.utils import gauss_weights_init

from torch_rl.core import *


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

        self.out = x
        return x



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



import nengo
class Reservoir(SpikingNetwork):


    def __init__(self,dt, sim_dt, input_size, network_size=800, recursive=False, spectral_radius=1., noise=False):
        super(Reservoir, self).__init__(dt, sim_dt)
        self.model = nengo.Network(seed=60)
        with self.model as model:
            """
                Network configurations.
            """
            self.input_node = nengo.Node(np.zeros(input_size))
            # Noise
            if noise:
                noise = nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0,0.5),default_size_out=input_size)

            # If specified create reservoir for the state
            state_ensemble = nengo.Ensemble(network_size, dimensions=input_size)

            if recursive:
                # l2 = nengo.Ensemble(200, dimensions=observation_size)
                W = np.random.uniform(-0.2, 0.2, (state_ensemble.n_neurons, state_ensemble.n_neurons))
                eig, eigv = np.linalg.eig(W)
                W = W / np.max(np.abs(eig)) * spectral_radius

                print('Spectral radius of reservoir weights: ', spectral_radius)

                nengo.Connection(state_ensemble.neurons, state_ensemble.neurons, transform=W)

            """
                Connect input to ensemble.
            """
            nengo.Connection(self.input_node, state_ensemble.neurons, transform=np.random.normal(0.1,0.2,(network_size,input_size)))


            """
                This is the output that is going to be read out after simulation ends.
            """
            self.output_probe = nengo.Probe(state_ensemble.neurons)

        self.sim = nengo.Simulator(network=self.model, dt=sim_dt)
        self.dt_steps = int(self.dt / self.sim_dt)

    def forward(self, x, requires_grad=False):
        if tor.is_tensor(x):
            x = x.data.numpy()
        if len(x.shape) >= 2 and x.shape[0] > 1:
            return self.batch_forward(x)
***REMOVED***
            x = x.reshape(-1)
        self.input_node.output = x
        self.sim.run_steps(self.dt_steps, progress_bar=False)

        out = self.sim.data[self.output_probe]
        """
            Take the average of all steps as output.
        """
        out = np.mean(out[-self.dt_steps:-1], axis=0)
        out = out.reshape(1, -1)
        return out


    def batch_forward(self, x):
        out = []
        for i in range(x.shape[0]):
            out.append(self.forward(x[i]))
        return np.vstack(out)

    def reset(self):
        self.sim.reset()





class SimpleSpikingAgent(Agent):
    """
        Implements liquid state machine agent with a neural network for readout.
    """

    def __init__(self, spiking_net, readout_net, action_choice_function=return_output):
        super(SimpleSpikingAgent, self).__init__(policy_network=readout_net, action_choice_function=action_choice_function)
        self.spiking_net = spiking_net
        self.spiking_state = None

    def action(self, *args, requires_grad=False):
        if len(args) > 1:
            x = np.hstack(args)
***REMOVED***
            x = args[0]
        x = self.spiking_net.forward(x)
        self.spiking_net.reset()
        self.spiking_state = x
        out = self.forward(to_tensor(x))
        return out


    def actions(self, *args, requires_grad=False):
        if len(args) > 1:
            x = np.hstack(args)
***REMOVED***
            x = args[0]
        x = self.spiking_net.forward(x)
        self.spiking_net.reset()
        self.spiking_state = x
        out = self.forward(to_tensor(x))
        return out


