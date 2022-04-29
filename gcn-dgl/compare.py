import dgl
import torch as torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import easydict

from gcn import Model as GCNModel
from first_order_term_only import Model as FOTOModel
from single_parameter import Model as SPModel
from mlp import Model as MLPModel

from utils import calculate_accuracy, calculate_adj_cap_gcn, customize_mask, calculate_adj_first_order_term, calculate_adj_single_param
import time


# Training settings
args = easydict.EasyDict({ 'seed': int(42),
         'epochs': int(200),
         'hidden': 16*8,
         'lr': float(0.01),
         'dropout': None
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

##Customise sample masks randomly
train_mask, val_mask, test_mask = customize_mask(labels.size(), 0.1, 0.3, 0.6)

adj_mat_gcn = calculate_adj_cap_gcn(graph)
adj_mat_foto = calculate_adj_first_order_term(graph)
adj_mat_sp = calculate_adj_single_param(graph)


# Model ,optimizer and loss function
model_gcn = GCNModel(features.size()[1], args.hidden,  dataset.num_classes, args.dropout)
model_foto = FOTOModel(features.size()[1], args.hidden,  dataset.num_classes, args.dropout)
model_sp =  SPModel(features.size()[1], args.hidden,  dataset.num_classes, args.dropout)
model_mlp = MLPModel(features.size()[1], args.hidden,  dataset.num_classes, args.dropout)

# model_mlp = torch.nn.Sequential(
#                   torch.nn.Linear(features.size()[1], args.hidden),
#                   torch.nn.ReLU(),
#                   torch.nn.Linear(args.hidden, dataset.num_classes),
#                   torch.nn.Softmax()
#                 )

optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=args.lr)
optimizer_foto = optim.Adam(model_foto.parameters(), lr=args.lr)
optimizer_sp = optim.Adam(model_sp.parameters(), lr=args.lr)
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=args.lr)


loss_fn = F.cross_entropy
# loss_fn = F.nll_loss

def main():
    gcn_acc_list=[]
    gcn_loss_list=[]
    foto_acc_list=[]
    foto_loss_list=[]
    sp_acc_list=[]
    sp_loss_list=[]
    mlp_acc_list=[]
    mlp_loss_list=[]
    for epoch in range(args.epochs):
        acc, loss = train('GCN', model_gcn, optimizer_gcn, features, adj_mat_gcn, epoch)
        gcn_acc_list.append(acc)
        gcn_loss_list.append(loss)

        acc, loss = train('FOTO', model_foto, optimizer_foto, features, adj_mat_foto, epoch)
        foto_acc_list.append(acc)
        foto_loss_list.append(loss)

        acc, loss = train('SP', model_sp, optimizer_sp, features, adj_mat_sp, epoch)
        sp_acc_list.append(acc)
        sp_loss_list.append(loss)

        acc, loss = train('MLP', model_mlp, optimizer_mlp, features, None, epoch)
        mlp_acc_list.append(acc)
        mlp_loss_list.append(loss)
   
    figure, (ax1, ax2) = plt.subplots(2, 1)
    figure.suptitle('Training Comparison Metrics')

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.plot(gcn_acc_list, label= 'GCN')
    ax1.plot(foto_acc_list, label= 'First-Order-Term-Only')
    ax1.plot(sp_acc_list, label= 'Single-Parameter')
    ax1.plot(mlp_acc_list, label= 'MLP')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.plot(gcn_loss_list, label= 'GCN')
    ax2.plot(foto_loss_list, label= 'First-Order-Term-Only')
    ax2.plot(sp_loss_list, label= 'Single-Parameter')
    ax2.plot(mlp_loss_list, label= 'MLP')
    
    plt.legend()
    plt.show()

    test('GCN', model_gcn, features, adj_mat_gcn)
    test('FOTO', model_foto, features, adj_mat_foto)
    test('SP', model_sp, features, adj_mat_sp)
    test('MLP', model_mlp, features, None)

def train(model_name, model, optimizer, features, adj_mat, epoch):
    t = time.time()

    model.train()
    optimizer.zero_grad()

    output = model(features, adj_mat)
    logp = F.log_softmax(output, 1)
    loss = loss_fn(logp[train_mask], labels[train_mask])

    loss.backward()
    optimizer.step()

    output = model(features, adj_mat)
    acc = calculate_accuracy(output, labels, val_mask)
    print("Model {} Epoch {:03d} Accuracy: {:.4f} Time: {:.4f}s"
        .format(model_name, epoch, acc, time.time() - t))
    return acc, loss.item()
    

def test(model_name, model, features, adj_mat):
    model.eval()
    output = model(features, adj_mat)
    acc = calculate_accuracy(output, labels, test_mask)
    print("{} Accuracy: {:.4f}".format(model_name, acc))


if __name__ == "__main__":
    main()