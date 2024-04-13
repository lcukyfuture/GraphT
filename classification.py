import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch

import torch.nn as nn
from torch.autograd import profiler
# from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader
from torch_geometric import datasets
import torch.nn.functional as F
from model import GraphTransformer
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from torch_geometric.transforms import OneHotDegree

import csv
import time 
import argparse
import copy
from data_idx import SubgraphDataset, GraphDataset
from utils import compute_kernel_for_batch, save_kernel, load_kernel, count_parameters
import pickle

def load_args():
    parser = argparse.ArgumentParser(description='Graph Kernel Transformer Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG', 'PATTERN', 'PROTEINS','NCI1', 'PTC_MR', 'ogbg-molhiv', 'IMDB-BINARY'],
                        help='Dataset to use')
    parser.add_argument('--num-layers', type=int, default=3, help="number of layers")
    parser.add_argument('--hop', type=int, default=2, help='Hop for subgraph extraction')
    parser.add_argument('--kernel', type=str, default='WL_GPU', choices=['SP', 'WL', 'WLSP', 'RW','GL', 'WL_GPU'],
                        help='Kernel type')
    parser.add_argument('--fold', type=int, default=10, help='The number of K folds')
    parser.add_argument('--same-attn', type=bool, default=True, help='Use the same ')
    parser.add_argument('--dim_hidden', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--epochs', type=int, default=1000,help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch_size')
    parser.add_argument('--dropout', type=float, default=0, help='drop out rate')
    parser.add_argument('--outdir', type=str, default='',help='output path')
    parser.add_argument('--wl', type=int, default=3, help='WL_GPU iteration')
    parser.add_argument('--batch-norm', action='store_true', help='use batch norm instead of layer norm')
    args = parser.parse_args()
    
    if args.outdir != '':
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.join(outdir, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.join(outdir,'fold_{}'.format(args.fold))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}'.format(args.kernel, args.wl, args.num_layers, args.hop, args.dropout, args.lr, args.batch_size)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
     
        args.outdir = outdir
    return args
    
device = torch.device('cuda')

def train(loader, model, criterion, optimizer): 
    model.train()
    total_loss = 0.0
    train_corr = 0.0
    strat_time = timer()
    mid_time = 0
    # with profiler.profile(use_cuda=True) as prof:

    gradients = {}
    for i, (data, mask, pe, labels) in enumerate(loader):
        # pe is the kernel matrix
        labels = labels.view(-1)
        
        data = data.to(device)
        mask = mask.to(device)
        pe = pe.to(device)
        label = labels.to(device)
        optimizer.zero_grad()
        #add kernel to model

        out = model(data, mask, pe)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()

        train_pred = out.data.argmax(dim=1)
        total_loss += loss.item()*len(data)
        train_corr += torch.sum(train_pred==label).item()
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    # print("mid time:", mid_time)
    end_time = timer()
    epoch_time = end_time - strat_time
    n_samples = len(loader.dataset)
    train_avg_loss = total_loss / n_samples
    train_avg_corr = train_corr / n_samples
    return train_avg_loss, train_avg_corr, epoch_time
    
def val(loader, model, criterion):
    model.eval()
    val_loss = 0
    val_nums = 0
    corr = 0
    with torch.no_grad():
        for data, mask, pe, labels in loader:
            labels = labels.view(-1)

            size = len(data)
            data = data.to(device)
            mask = mask.to(device)
            pe = pe.to(device)
            label = labels.to(device)
            out = model(data, mask, pe)

            loss = criterion(out, label)
            val_loss += loss.item()*size
            val_nums += size

            pred = out.argmax(dim=-1)
            corr += int((pred == label).sum())
    val_avg_loss = val_loss / val_nums
    val_avg_corr = corr / len(loader.dataset)
    val_avg_loss = round(val_avg_loss, 3)
    return val_avg_loss, val_avg_corr

def plot_curve(train_loss_list, test_loss_list, train_acc_list, test_acc_list, fold):
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

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.suptitle(f'Loss And Accuacy Curves of Fold {fold}')
    # plt.savefig(f'{args.kernel}{args.num_layers}layer{args.hop}hops{args.dropout}dropout_figs/curves_fold_{fold}.png')
    plt.savefig(os.path.join(args.outdir, f'curves_fold_{fold}.png'))
    plt.show()
    
global args

def main():
    
    global args
    args = load_args()
    torch.manual_seed(44)
    np.random.seed(44)
    data_path = '../dataset/TUDataset'
    dataset_name = args.dataset
    # torch.use_deterministic_algorithms(True)
    # dataset = datasets.TUDataset(data_path, dataset_name)
    
    if args.dataset == 'IMDB-BINARY':
        transform = OneHotDegree(max_degree=135)
        dataset = datasets.TUDataset(root=data_path, name='IMDB-BINARY', transform=transform)
    else:
        dataset = datasets.TUDataset(data_path, dataset_name)
        
    classes = dataset.num_classes 

    print(f"{args.num_layers}layers {args.hop}hops")

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    csv_file = open(args.outdir + '/results.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Best Epoch','Best Accuracy'])

    idx_path = 'new_folds/{}/inner_folds/{}-{}-{}.txt'
    test_idx_path = 'new_folds/{}/test_idx-{}.txt'
    inner_idx = 1

    train_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'train_idx', args.fold, inner_idx)).astype(int)).long()
    val_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'val_idx', args.fold, inner_idx)).astype(int)).long()
    test_fold_idx = torch.from_numpy(np.loadtxt(
        test_idx_path.format(args.dataset, args.fold)).astype(int)).long()

    if not os.path.exists("cache/pe/{}".format(args.dataset)):
        try:
            os.makedirs("cache/pe/{}".format(args.dataset))
        except Exception:
            pass
    
    kernel_cache_path = 'cache/pe/{}/{}_{}_{}.pkl'.format(
        args.dataset, args.kernel, args.wl, args.hop)
    
    Subgraph_kernels = load_kernel(kernel_cache_path)

    if Subgraph_kernels is None:
        Subgraph_kernels = []
        SubdDataset = SubgraphDataset(dataset, k_hop = args.hop)
        print("Length of dataset:", len(dataset))
        print("compute subgraph kernel...")
        SubDataloader = PyGDataLoader(SubdDataset, batch_size=1, shuffle=False)
        for data in SubDataloader:
            Subgraph_kernel = compute_kernel_for_batch(data, device, args.wl)
            Subgraph_kernels.extend(Subgraph_kernel)
        save_kernel(Subgraph_kernels, kernel_cache_path)
   
    train_fold_idx = train_fold_idx.tolist()
    val_fold_idx = val_fold_idx.tolist()
    test_fold_idx = test_fold_idx.tolist()
    train_dataset = GraphDataset(dataset[train_fold_idx])
    
    print(len(train_dataset))
    print(train_dataset[0])
    val_dataset = GraphDataset(dataset[val_fold_idx])
    test_dataset = GraphDataset(dataset[test_fold_idx])
    
    train_dataset.pe_list = [Subgraph_kernels[i] for i in train_fold_idx]
    val_dataset.pe_list = [Subgraph_kernels[i] for i in val_fold_idx]
    test_dataset.pe_list = [Subgraph_kernels[i] for i in test_fold_idx]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn())

    print(train_dataset.pe_list[0])

    best_acc = 0
    best_epoch = 9999
    input_size = dataset.num_node_features
    nb_class = dataset.num_classes

    model = GraphTransformer(in_size=input_size,
                            nb_class=nb_class,
                            d_model=args.dim_hidden,
                            dim_feedforward=2*args.dim_hidden,
                            dropout=args.dropout,
                            nb_layers=args.num_layers,
                            batch_norm=False
                            ).to(device)
    
    print("Total number of parameters: {}".format(count_parameters(model)))
    warm_up = 100
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr , weight_decay = weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_loss = float('inf')
    patience_counter = 0
    Allstart_time = time.time()
    epoch_time_list = []
    for epoch in range(args.epochs):

        print(f'Epoch: {epoch}/{args.epochs}, LR: {optimizer.param_groups[0]["lr"]}')
        # train_loss, train_acc, epoch_time = train(train_dataloader, model, warm_up, criterion, optimizer, warmup_lr_scheduler, epoch)
        train_loss, train_acc, epoch_time = train(train_loader, model,criterion, optimizer)
        epoch_time_list.append(epoch_time)
        val_loss, val_acc = val(val_loader, model, criterion)
        lr_scheduler.step()
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weight = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 200:
                break

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f'epoch: {epoch:03d}, Train loss: {train_loss:.4f}, val loss:{val_loss:.4f}, Train acc: {train_acc:.4f}, val acc : {val_acc:.4f}, Best loss: {best_loss:.4f}, Epoch time: {epoch_time}')
        csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, best_epoch, best_loss])

    print(f'Best epoch: {best_epoch}')
    print(f'Best val loss for fold {args.fold}: {best_loss:.4f}')
    model.load_state_dict(best_weight)
    test_loss, test_acc = val(test_loader, model, criterion)
    print(f'Test acc for fold {args.fold}: {test_acc:.4f}')
    csv_writer.writerow([test_loss, test_acc])
    plot_curve(train_loss_list, val_loss_list, train_acc_list, val_acc_list, args.fold)

    Allend_time = time.time()
    Gap_time = Allend_time - Allstart_time
    print(f'Time: {Gap_time}')
    csv_writer.writerow([Gap_time])
    mean_epoch_time = np.mean(epoch_time_list)
    std_epoch_time = np.std(epoch_time_list)/epoch
    print(f'epoch_time: {mean_epoch_time:.4f} +/-{std_epoch_time:.4f}')
    
if __name__ == "__main__":
    # args = load_args()
    # dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    
    main()
