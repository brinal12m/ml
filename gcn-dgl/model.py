import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Layer

class Model(nn.Module):
    """GCN model class applies two layers of transformation on the Graph
    """
    def __init__(self, noFeat, noHidden, noClass, dropout):
        super(Model, self).__init__()
        self.layer1 = Layer(noFeat, noHidden)
        self.layer2 = Layer(noHidden, noClass)
        self.dropout = dropout

    def forward(self, in_feat, adj_mat):
        x = F.relu(self.layer1(in_feat, adj_mat))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.softmax(self.layer2(x,adj_mat), dim=1)