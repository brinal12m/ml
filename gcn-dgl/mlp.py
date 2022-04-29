import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
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

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Model(nn.Module):
    """First order term only model class applies two layers of transformation on the Graph
    """
    def __init__(self, noFeat, noHidden, noClass, dropout:None):
        super(Model, self).__init__()
        self.layer1 = Layer(noFeat, noHidden)
        self.layer2 = Layer(noHidden, noClass)
        self.dropout = dropout

    def forward(self, in_feat, adj_mat):
        x = F.relu(self.layer1(in_feat))
        # drop out if available in training phase
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, training=self.training)
        return F.softmax(self.layer2(x), dim=1)