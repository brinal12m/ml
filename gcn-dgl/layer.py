import torch as torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import math


class Layer(nn.Module):
    """Applies a layered data transformation to the incoming data 
    """
    def __init__(self, inputFeat, outputFeat, bias=True):
        super(Layer, self).__init__()
        self.inputFeat = inputFeat
        self.outputFeat = outputFeat
        self.weight = Parameter(torch.FloatTensor(inputFeat, outputFeat))
        if bias:
            self.bias = Parameter(torch.FloatTensor(outputFeat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(adj, torch.mm(input, self.weight))
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        