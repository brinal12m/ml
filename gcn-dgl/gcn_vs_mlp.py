import dgl
import torch as torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import easydict
from layer import Layer
from model import Model
from utils import calculate_accuracy, calculate_adj_cap, customize_mask
import time


# Training settings
args = easydict.EasyDict({ 'seed': int(42),
         'epochs': int(200),
         'hidden': 16,
         'lr': float(0.01),
         'dropout': float(0.5)
        })

# Set the constant seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Obtain the data set
# dataset = dgl.data.CoraGraphDataset()
dataset = dgl.data.CiteseerGraphDataset()

graph = dataset[0]

features = graph.ndata['feat']
labels = graph.ndata['label']

##Customise sample masks randomly
train_mask, val_mask, test_mask = customize_mask(labels.size(), 0.1, 0.3, 0.6)

adj_mat_cap = calculate_adj_cap(graph)

# Model ,optimizer and loss function
model = Model(features.size()[1], args.hidden,  dataset.num_classes, args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

loss_fcn = F.cross_entropy
# loss_fcn = F.nll_loss

comp_model = torch.nn.Sequential(
                  torch.nn.Linear(features.size()[1], args.hidden),
                  torch.nn.ReLU(),
                  torch.nn.Linear(args.hidden, dataset.num_classes),
                  torch.nn.Softmax()
                )
comp_optimizer = optim.Adam(comp_model.parameters(), lr=args.lr)

def main():
    acc_list=[]
    loss_list=[]
    comp_acc_list=[]
    comp_loss_list=[]
    for epoch in range(args.epochs):
        acc, loss, comp_acc, comp_loss = train(epoch)
   
        acc_list.append(acc)
        loss_list.append(loss)
        comp_acc_list.append(comp_acc)
        comp_loss_list.append(comp_loss)
   
    plot_stats('GCN', 'MLP', acc_list, comp_acc_list, loss_list, comp_loss_list)
    test()

def plot_stats(model_name, comp_model_name, accuracy, comp_accuracy,  loss, comp_loss):
    figure, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.plot(accuracy, label= model_name)
    ax1.plot(comp_accuracy, label= comp_model_name)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.plot(loss)
    ax2.plot(comp_loss)
    
    figure.legend(loc='upper left')
    figure.tight_layout()
    plt.show()


def train(epoch):
    t = time.time()

    model.train()
    comp_model.train()
    optimizer.zero_grad()
    comp_optimizer.zero_grad() 
    

    output = model(features, adj_mat_cap)
    logp = F.log_softmax(output, 1)
    loss = loss_fcn(logp[train_mask], labels[train_mask])

    comp_output = comp_model(features)
    comp_logp = F.log_softmax(comp_output, 1)
    comp_loss = loss_fcn(comp_logp[train_mask], labels[train_mask])


    loss.backward()
    optimizer.step()

    comp_loss.backward()
    comp_optimizer.step()


    output = model(features, adj_mat_cap)
    acc = calculate_accuracy(output, labels, val_mask)

    comp_output = comp_model(features)
    comp_acc = calculate_accuracy(comp_output, labels, val_mask)

    print("Epoch {:03d}: Time: {:.4f}s"
        .format(epoch, time.time() - t))
    
    return acc, loss.item(), comp_acc, comp_loss.item()
    

def test():
    model.eval()
    output = model(features, adj_mat_cap)
    acc = calculate_accuracy(output, labels, test_mask)
    print("GCN Accuracy: %f" % acc)


if __name__ == "__main__":
    main()