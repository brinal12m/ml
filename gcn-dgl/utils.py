import torch as torch
import numpy as np

def calculate_accuracy(model_output, ground_truth, mask):
    logits = model_output[mask]
    labels = ground_truth[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def calculate_adj_cap(graph):
    '''
    Utility function to calculate adjacency cap matrix
    '''
    adj_mat = graph.adjacency_matrix()
    adj_mat_tilda = torch.eye(adj_mat.shape[0]) + adj_mat
    d_tilda_inv_sqrt = torch.diag(torch.pow(torch.sum(adj_mat_tilda, 1), -0.5)) 
    return torch.mm(torch.mm(d_tilda_inv_sqrt, adj_mat_tilda), d_tilda_inv_sqrt)


def customize_mask(size, train_ratio, val_ratio, test_ratio):
    '''
    Creates the mask of the for the given size with different given ratios.
    Output be Tensor masks for train, validation and test respectively
    '''
    masks = torch.from_numpy(np.random.choice(a=[1, 2, 3], size=size, p=[train_ratio, val_ratio, test_ratio]))
    return (masks == 1), (masks == 2), (masks == 3)