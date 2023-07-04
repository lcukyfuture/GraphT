import torch
import torch.nn as nn
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
from model import GCN, ClassificationModel
from Satmodel import GraphTransformer

dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

print(dataset[0])
# dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')
print(len(dataset), dataset.num_classes, dataset.num_node_features)
train_dataset = dataset[0:150]
test_dataset = dataset[150:]

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# model = GCN(dataset = dataset, hidden_channels=64)
# in_dim, out_dim, num_heads, use_bias, num_layers, num_classes
# model = ClassificationModel(in_dim=dataset.num_node_features, out_dim=64, num_heads=8, num_layers=2, num_classes=dataset.num_classes)
model = GraphTransformer(in_size=dataset.num_node_features,
                            num_class=dataset.num_classes,
                            d_model=64,
                            dim_feedforward=2*64,
                            dropout=0.2,
                            num_heads=8,
                            num_layers=6,
                            batch_norm=False,
                            use_edge_attr=False,
                            num_edge_features=dataset.num_edge_features,
                            use_global_pool=False)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(): 
    model.train()
    for data in train_dataloader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    corr = 0
    for data in test_dataloader:
        out = model(data)
        pred = out.argmax(dim=1)
        corr += int((pred == data.y).sum())
    return corr / len(test_dataloader.dataset)

for epoch in range(1, 121):
    train()
    train_acc = test(train_dataloader)
    test_acc = test(test_dataloader)
    print(f'epoch: {epoch:03d}, Train acc: {train_acc:.4f}, Test acc : {test_acc:.4f}')