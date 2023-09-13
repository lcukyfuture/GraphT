import torch
import torch.nn as nn
# from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch.nn.functional as F
from model import GCN, ClassificationModel
from kernelmodel import GraphTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import os
import csv
import time 
import argparse

def load_args():
    parser = argparse.ArgumentParser(description='Graph Kernel Transformer Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG', 'PATTERN', 'PROTEINS', 'ogbg-molhiv'],
                        help='Dataset to use')
    parser.add_argument('--num-layers', type=int, default=3, help="number of layers")
    parser.add_argument('--hop', type=int, default=2, help='Hop for subgraph extraction')
    parser.add_argument('--kernel', type=str, default='SP', choices=['SP', 'WL', 'WLSP', 'RW','GRAPHLET'],
                        help='Kernel type')
    parser.add_argument('--fold', type=int, default=10, help='The number of K folds')
    parser.add_argument('--same-attn', type=bool, default=True, help='Use the same ')
    parser.add_argument('--dim_hidden', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--epochs', type=int, default=160,help='number of epochs')
    args = parser.parse_args()
    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
# dataset = GNNBenchmarkDataset(root='/tmp/PATTERN', name='PATTERN')
# dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
# dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')
print(len(dataset), dataset.num_classes, dataset.num_node_features)
# train_dataset = dataset[0:150]
# test_dataset = dataset[150:]
#Kfold 

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
def train(loader, model, warm_up, criterion, optimizer, lr_scheduler, epoch): 
    model.train()
    total_loss = 0
    train_corr = 0
    nums = 0
    strat_time = time.time()
    for i, data in enumerate(loader):
        size = len(data.y)
        if epoch < warm_up:
            iteration = epoch * len(loader) + i
            for param in optimizer.param_groups:
                param["lr"] = lr_scheduler(iteration)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item()*size
        nums += size
        train_pred = out.argmax(dim=-1).cpu()
        train_corr += int((train_pred == data.y.cpu()).sum())
        loss.backward()
        optimizer.step()
    train_avg_loss = total_loss / nums
    train_avg_corr = train_corr / len(loader.dataset)
    end_time = time.time()
    epoch_time = end_time - strat_time
    return train_avg_loss, train_avg_corr, epoch_time


def test(loader, model, criterion):
    model.eval()
    test_loss = 0
    test_nums = 0
    corr = 0
    for data in loader:
        size = len(data.y)
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        test_loss += loss.item()*size
        test_nums += size
        pred = out.argmax(dim=-1).cpu()
        corr += int((pred == data.y.cpu()).sum())
    test_avg_loss = test_loss / test_nums
    test_avg_corr = corr / len(loader.dataset)
    return test_avg_loss, test_avg_corr


def plot_curve(kernel, train_loss_list, test_loss_list, train_acc_list, test_acc_list, fold):
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle(f'Loss And Accuacy Curves of Fold {fold}')
    plt.savefig(f'{kernel}figs/curves_fold_{fold}.png')
    plt.show()


global args
args = load_args()
train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []
csv_file = open(f'{args.kernel}{args.num_layers}layer{args.hop}hops_results.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Fold', 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Best Accuracy'])

def main():
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    indices = list(kf.split(dataset))
    best_acc = 0
    best_accs = []
    Allstart_time = time.time()
    for i in range(5):
        train_acc_list.clear()
        test_acc_list.clear()
        train_loss_list.clear()
        test_loss_list.clear()
        best_acc = 0
        train_indices, test_indices = indices[i]
        train_indices = torch.tensor(train_indices, dtype=torch.long)
        test_indices = torch.tensor(test_indices, dtype=torch.long)
        train_dataset = dataset[train_indices]
        test_dataset = dataset[test_indices]

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
        model = GraphTransformer(in_size=dataset.num_node_features,
                                num_class=dataset.num_classes,
                                d_model=args.dim_hidden,
                                dim_feedforward=2*args.dim_hidden,
                                dropout=0.2,
                                num_layers=args.num_layers,
                                batch_norm=False,
                                use_edge_attr=False,
                                num_edge_features=dataset.num_edge_features,
                                use_global_pool=False,
                                kernel=args.kernel,
                                hop=args.hop,
                                same_attn=True).to(device)
        # for name, param in model.named_parameters():
        #     print(name, param.data)
        # print(model)
        # print(model.parameters)
        epochs = 161
        warm_up = 10
        weight_decay = 1e-4
        lr = 0.0001
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr , weight_decay = weight_decay)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-warm_up)
        lr_steps = lr / (warm_up * len(train_dataloader))
        def warmup_lr_scheduler(s):
            lr = s * lr_steps
            return lr

        for epoch in range(epochs):
            train_loss, train_acc, epoch_time = train(train_dataloader, model, warm_up, criterion, optimizer, warmup_lr_scheduler, epoch)
            test_loss, test_acc = test(test_dataloader, model, criterion)
            if epoch >= warm_up:
                lr_scheduler.step()
            if best_acc < test_acc:
                best_acc = test_acc
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            # torch.save(model.state_dict(), 'best_model_fold_{}.pth'.format(i+1)) 
            print(f'Fold: {i}, epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Test loss:{test_loss:.4f}, Train acc: {train_acc:.4f}, Test acc : {test_acc:.4f}, Best acc: {best_acc:.4f}, Epoch time: {epoch_time}')
            csv_writer.writerow([i+1, epoch, train_loss, train_acc, test_loss, test_acc, best_acc])

        print(f'Best test acc for fold {i+1}:{best_acc:.4f}')
        best_accs.append(best_acc)
        os.makedirs(f'{args.kernel}{args.num_layers}layer{args.hop}hops_figs', exist_ok=True)
        plot_curve(args.kernel, train_loss_list, test_loss_list, train_acc_list, test_acc_list, i+1)

    avg_test_acc = np.mean(best_accs)
    std_error = np.std(best_accs) / np.sqrt(5)
    Allend_time = time.time()
    Gap_time = Allend_time - Allstart_time
    print(f'Average test acc: {avg_test_acc:.4f} +/- {std_error:.4f}, Time: {Gap_time}')

if __name__ == "__main__":
    main()