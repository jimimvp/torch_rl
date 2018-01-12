import torch as tor
from torch.nn import Module
from torch.autograd import Variable
from torch import nn



class PolicyAHG(nn.Module):

    def __init__(self,  input_size, output_size):
        super(PolicyAHG, self).__init__()

        self.f1 = nn.Linear(input_size,64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, output_size)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, x):

        out = self.f1(x)
        out = self.relu(out)
        out = self.f2(out)
        out = self.relu(out)
        out = self.f3(out)
        out = self.softmax(out)
        return out

