import sys
sys.path.append("./")
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch
import torch.nn as nn
# from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import os
import csv
import time 
import argparse
import copy
from torch_geometric.nn.models import GIN
from DGCNN import DGCNN
from GIN import GIN

def load_args():
    parser = argparse.ArgumentParser(description='Graph Kernel Transformer Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG', 'PATTERN', 'PROTEINS','NCI1', 'PTC_MR', 'ogbg-molhiv'],
                        help='Dataset to use')
    parser.add_argument('--num-layers', type=int, default=5, help="number of layers")
    parser.add_argument('--dim_hidden', type=int, default=64, help="hidden dimension of model")
    parser.add_argument('--fold', type=int, default=1, help='The number of K folds')
    # parser.add_argument('--dim_hidden', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--epochs', type=int, default=300,help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch_size')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('--outdir', type=str, default='',help='output path')
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
        outdir = outdir + '/{}_{}_{}_{}_{}'.format(args.num_layers, args.dropout, args.lr, args.batch_size, args.dim_hidden)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        
        args.outdir = outdir

    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def save_gradients(model, filename="gradients.txt"):
#     with open(filename, "w") as f:
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 grad_data = param.grad.data
#                 f.write(f"Parameter: {name}\nGradient:\n{grad_data}\n\n")

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
        # iteration = epoch * len(loader) + i
        # for param in optimizer.param_groups:
        #     param["lr"] = lr_scheduler(iteration)
        data = data.to(device)
        optimizer.zero_grad()
        #add kernel to model
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item()*size
        nums += size
        train_pred = out.argmax(dim=-1).cpu()
        train_corr += int((train_pred == data.y.cpu()).sum())
        loss.backward()
        # save_gradients(model, filename=f"gradients_epoch_{epoch}.txt")
        optimizer.step()
    train_avg_loss = total_loss / nums
    train_avg_corr = train_corr / len(loader.dataset)
    end_time = time.time()
    epoch_time = end_time - strat_time
    return train_avg_loss, train_avg_corr, epoch_time


def val(loader, model, criterion):
    model.eval()
    val_loss = 0
    val_nums = 0
    corr = 0
    for data in loader:
        size = len(data.y)
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        val_loss += loss.item()*size
        val_nums += size
        pred = out.argmax(dim=-1).cpu()
        corr += int((pred == data.y.cpu()).sum())
    val_avg_loss = val_loss / val_nums
    val_avg_corr = corr / len(loader.dataset)
    return val_avg_loss, val_avg_corr

def plot_curve( train_loss_list, test_loss_list, train_acc_list, test_acc_list, fold):
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
    # plt.show()


global args



def main():
    torch.manual_seed(44)
    np.random.seed(44)
    torch.use_deterministic_algorithms(True)

    Allstart_time = time.time()
    dataset = raw_dataset
    print("Length of dataset:", len(dataset))
    print(f"{args.num_layers}layers")
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    best_acc = 0
    # csv_file = open(f'{args.kernel}{args.num_layers}layer{args.hop}hops{args.dropout}dropout_figs/{args.kernel}{args.num_layers}layer{args.hop}hops_results.csv', 'w', newline='')
    csv_file = open(args.outdir + '/results.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Best Epoch','Best Accuracy'])


    # kf = KFold(n_splits=5, shuffle=True, random_state=1)
    # indices = list(kf.split(dataset))
    # best_acc = 0
    # best_accs = []
    idx_path = 'new_folds/{}/inner_folds/{}-{}-{}.txt'
    test_idx_path = 'new_folds/{}/test_idx-{}.txt'
    inner_idx = 1

    train_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'train_idx', args.fold, inner_idx)).astype(int)).long()
    val_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'val_idx', args.fold, inner_idx)).astype(int)).long()
    test_fold_idx = torch.from_numpy(np.loadtxt(
        test_idx_path.format(args.dataset, args.fold)).astype(int)).long()
    train_acc_list.clear()
    val_acc_list.clear()
    train_loss_list.clear()
    val_loss_list.clear()
    best_acc = 0
    best_epoch = 9999


    # train_dataset = dataset[train_fold_idx]
    # val_dataset = dataset[val_fold_idx]
    # test_dataset = dataset[test_fold_idx]
    train_dataset = [dataset[idx] for idx in train_fold_idx]
    val_dataset = [dataset[idx] for idx in val_fold_idx]
    test_dataset = [dataset[idx] for idx in test_fold_idx]
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.num_layers == 5 and args.dim_hidden == 32:
        hidden_units = [32, 32, 32, 32]
    elif args.num_layers == 5 and args.dim_hidden == 64:
        hidden_units = [64, 64, 64, 64]
    elif args.num_layers == 3 and args.dim_hidden == 32:
        hidden_units = [32, 32]
    elif args.num_layers == 2 and args.dim_hidden == 64:
        hidden_units = [64]

    model = GIN(dim_features = dataset.num_node_features, dim_target = dataset.num_classes, hidden_units = hidden_units, 
                dropout=args.dropout, train_eps=True, aggregation='sum').to(device)
    # model = GIN(in_channels=dataset.num_node_features, hidden_channels=2 * args.dim_hidden,num_layers=args.num_layers,
    #             out_channels=dataset.num_classes,
    #             dropout=args.dropout).to(device)
    # for name, param in model.named_parameters():
    #     print(name, param.data)
    # print(model)
    # print(model.parameters)
    warm_up = 10
    weight_decay = 1e-4
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(args.epochs):

        print(f'Epoch: {epoch}/{args.epochs}, LR: {optimizer.param_groups[0]["lr"]}')
        # train_loss, train_acc, epoch_time = train(train_dataloader, model, warm_up, criterion, optimizer, warmup_lr_scheduler, epoch)
        train_loss, train_acc, epoch_time = train(train_dataloader, model, warm_up, criterion, optimizer, lr_scheduler, epoch)
        val_loss, val_acc = val(val_dataloader, model, criterion)
        lr_scheduler.step()
        if best_acc < val_acc and epoch > args.epochs-50:
            best_acc = val_acc
            best_epoch = epoch
            best_weight = copy.deepcopy(model.state_dict())
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        # torch.save(model.state_dict(), 'best_model_fold_{}.pth'.format(i+1)) 
        print(f'epoch: {epoch:03d}, Train loss: {train_loss:.4f}, val loss:{val_loss:.4f}, Train acc: {train_acc:.4f}, val acc : {val_acc:.4f}, Best acc: {best_acc:.4f}, Epoch time: {epoch_time}')
        csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, best_epoch, best_acc])
    print(f'Best epoch: {best_epoch}')
    print(f'Best val acc for fold {args.fold}:{best_acc:.4f}')
    model.load_state_dict(best_weight)
    test_loss, test_acc = val(test_dataloader, model, criterion)
    print(f'Test acc for fold {args.fold}: {test_acc:.4f}')
    csv_writer.writerow([test_loss, test_acc])
    plot_curve(train_loss_list, val_loss_list, train_acc_list, val_acc_list, args.fold)

    Allend_time = time.time()
    Gap_time = Allend_time - Allstart_time
    print(f'Time: {Gap_time}')
    csv_writer.writerow([Gap_time])

if __name__ == "__main__":
    args = load_args()
    # dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    raw_dataset = TUDataset(root=f'/tmp/{args.dataset}', name=args.dataset)
    print(len(raw_dataset), raw_dataset.num_classes, raw_dataset.num_node_features)
    num_classes = raw_dataset.num_classes
    num_node_features = raw_dataset.num_node_features
    main()