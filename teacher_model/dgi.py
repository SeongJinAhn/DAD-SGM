import pickle
import os.path as osp
import time
import torch

import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DeepGraphInfomax, GCNConv
from torch_geometric.datasets import Planetoid, Amazon, AmazonProducts, Flickr, Coauthor, CoraFull, Reddit, WikiCS, Reddit2, LINKXDataset, WikipediaNetwork, WebKB, Actor
from heterodataset import HeterophilousGraphDataset
from torch_geometric.utils import train_test_split_edges
from utils import add_edge_noise, add_feature_noise, LogReg, svm_test2
from torch_geometric.nn import GAE, VGAE, APPNP, GCN
import torch_geometric.transforms as T

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
device = torch.device('cpu')
dataName = dataset = 'Cora'
path = osp.join(osp.dirname("D:\\"), 'data',  dataset)
#path = osp.join('D:\data', 'Reddit1')

if dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(path, dataset, 'public')
if dataset in ['cs', 'physics']:
    dataset = Coauthor(path, dataset)
if dataset in ['computers', 'photo']:
    dataset = Amazon(path, dataset)
if dataset in ['Penn94', 'Reed98']:
    dataset = LINKXDataset(path, name=dataset)
if dataset in ['Texas', 'Cornell', 'Wisconsin']:
    dataset = WebKB(path, name=dataset)
if dataset == 'Reddit':
    dataset = Reddit2(path)
if dataset in ['roman_empire', 'amazon_ratings']:
    dataset = HeterophilousGraphDataset(path, name=dataset)
if dataset in ['Products']:
    path = osp.join(osp.dirname("D:\\"), 'data',  'Products')
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset('ogbn-products', path)
    split_idx = dataset.get_idx_split()
    data = dataset[0].to('cuda')
    label = data.y.squeeze()
    data.train_mask = torch.zeros(data.num_nodes).cuda().bool()
    data.val_mask = torch.zeros(data.num_nodes).cuda().bool()
    data.test_mask = torch.zeros(data.num_nodes).cuda().bool()
    data.train_mask[split_idx['train'].cuda()] = True
    data.val_mask[split_idx['valid'].cuda()] = True
    data.test_mask[split_idx['test'].cuda()] = True

data = dataset[0].to(device)
torch.manual_seed(42)
data.y = data.y.squeeze()
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.prelu1 = torch.nn.PReLU(512)
        self.prelu2 = torch.nn.PReLU(64)
        self.conv1 = GCNConv(in_channels, 64).to('cuda:0')
        self.conv2 = GCNConv(512, 64, cached=False).to('cuda:0')

    def forward(self, x, edge_index):
        x = self.conv1.to('cuda:0')(x, edge_index)
        x = self.prelu1(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0), device=x.device)], edge_index

model = DeepGraphInfomax(
    hidden_channels=64,
    encoder=Encoder(dataset.num_features, 64),
    summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(),
    corruption=corruption,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time
def train():
    model.train()
    optimizer.zero_grad()
    start_time = time.time()
    pos_z, neg_z, summary = model(data.x, data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item(), 123

for epoch in range(1, 2001):
    loss, perturbed_image = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    z, _, _ = model(data.x, data.edge_index)

    if epoch % 100 == 0:
        with torch.no_grad():
            import time
            start_time = time.time()
            z, _, _ = model(data.x, data.edge_index)
            print(time.time() - start_time)
            print('SVM : ', svm_test2(z.cpu().detach().numpy(), data.y.cpu().detach().numpy(), data.train_mask.cpu().detach().numpy(), data.test_mask.cpu().detach().numpy()))
            print('LogReg :',LogReg(z.cpu().detach().numpy(), data.y.cpu().detach().numpy(), data.train_mask.cpu().detach().numpy(), data.test_mask.cpu().detach().numpy()))

text = './DGI_' + dataName + '.txt'
with open(text, 'wb') as f:
    pickle.dump([z, data.edge_index, data.x], f)

