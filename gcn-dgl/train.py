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
         'hidden': 16*8,
         'lr': float(0.01),
         'dropout': float(0.5)
        })

# Set the constant seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Obtain the data set
dataset = dgl.data.CoraGraphDataset()
# dataset = dgl.data.CiteseerGraphDataset()

graph = dataset[0]

features = graph.ndata['feat']
labels = graph.ndata['label']

# Sample masks given by DGL lib
train_mask, test_mask, val_mask = (graph.ndata['train_mask'], graph.ndata['test_mask'], graph.ndata['val_mask'])

print( 'DGL dataset info:'
        'No. of classes: {:d}'.format(dataset.num_classes),
        'Train samples: {:d}'.format(torch.sum(train_mask == True)),
        'Validation samples: {:d}'.format(torch.sum(val_mask == True)),
        'Test samples: {:d}'.format(torch.sum(test_mask == True)))


##Customise sample masks randomly
train_mask, val_mask, test_mask = customize_mask(labels.size(), 0.1, 0.3, 0.6)
print( 'Custom dataset info:'
        'Train samples: {:d}'.format(torch.sum(train_mask == True)),
        'Validation samples: {:d}'.format(torch.sum(val_mask == True)),
        'Test samples: {:f}'.format(torch.sum(test_mask == True)))


adj_mat_cap = calculate_adj_cap(graph)

# Model , optimizer and loss function
model = Model(features.size()[1], args.hidden,  dataset.num_classes, args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

loss_fcn = F.cross_entropy
# loss_fcn = F.nll_loss


def main():
    loss_list=[]
    acc_list=[]
    
    for epoch in range(args.epochs):
        acc, loss = train(epoch)
        loss_list.append(loss)
        acc_list.append(acc)

    plot_stats(acc_list, loss_list)
    test()

def plot_stats(accuracy, loss):
    figure, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.plot(accuracy)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.plot(loss)
    
    figure.tight_layout()
    plt.show()


def train(epoch):
    t = time.time()

    model.train()

    optimizer.zero_grad() 
    
    output = model(features, adj_mat_cap)
    logp = F.log_softmax(output, 1)
    loss = loss_fcn(logp[train_mask], labels[train_mask])

    loss.backward()
    optimizer.step()

    output = model(features, adj_mat_cap)
    acc = calculate_accuracy(output, labels, val_mask)

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%} Time: {:.4f}s"
        .format(epoch, loss.item(), acc, time.time() - t))

    return acc, loss.item()
    

def test():
    model.eval()
    output = model(features, adj_mat_cap)
    acc = calculate_accuracy(output, labels, test_mask)
    print("Test Accuracy: %f" % acc)


if __name__ == "__main__":
    main()