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
dataset = 'Products'
noisy_rate = 0.0
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
#data = dataset[0].to(device)
torch.manual_seed(42)
train_rate = 1
data.test_pos_edge_index = data.test_neg_edge_index = data.train_pos_edge_index = data.edge_index
#if noisy_rate != 0:
#    data.x = add_feature_noise(data.x, noisy_rate)

data.y = data.y.squeeze()
data.train_mask = 1
if data.train_mask == 1:
    idx = torch.randperm(data.x.size()[0])
    data.train_mask = torch.zeros(data.x.size()[0]).bool().cuda()
    data.val_mask = torch.zeros(data.x.size()[0]).bool().cuda()
    data.test_mask = torch.zeros(data.x.size()[0]).bool().cuda()
    data.train_mask[idx[:idx.size()[0]//10]] = True
    data.val_mask[idx[idx.size()[0]//10:idx.size()[0]//10*2]] = True
    data.test_mask[idx[idx.size()[0]//10*2:]] = True

import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
#        self.conv =  GCN(dataset.num_node_features, hidden_channels=128,
#          out_channels=64, num_layers=2).to(device)
        self.prelu1 = torch.nn.PReLU(512)
        self.prelu2 = torch.nn.PReLU(64)
        self.conv1 = GCNConv(in_channels, 64).to('cuda:0')
        self.conv2 = GCNConv(512, 64, cached=False).to('cuda:0')

        self.classifier1 = nn.Linear(in_channels, 128).to('cuda:0')
        self.classifier2 = nn.Linear(128, 128).to('cuda:0')
        self.classifier3 = nn.Linear(128, 128).to('cuda:0')
        self.classifier4 = nn.Linear(128, 128).to('cuda:0')
        self.classifier5 = nn.Linear(128, 64).to('cuda:0')
        torch.nn.init.xavier_uniform_(self.classifier1.weight)
        torch.nn.init.xavier_uniform_(self.classifier2.weight)
        torch.nn.init.xavier_uniform_(self.classifier3.weight)
        torch.nn.init.xavier_uniform_(self.classifier4.weight)
        torch.nn.init.xavier_uniform_(self.classifier5.weight)

    def forward(self, x, edge_index):
#        z = self.classifier1(x)
#        z = F.normalize(z,p=2,dim=1)
#        z = self.classifier2(z)
#        z = F.normalize(z,p=2,dim=1)
#        z = self.classifier3(z)
#        z = F.normalize(z,p=2,dim=1)
#        z = self.classifier4(z)
#        z = F.normalize(z,p=2,dim=1)
#        z = self.classifier5(z)
#        return z

        x = self.conv1.to('cuda:0')(x, edge_index)
#        x = self.prelu1(x)
#        x = self.conv2(x, edge_index)
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
#    x = data.x.clone().detach().requires_grad_(True)
    pos_z, neg_z, summary = model(data.x, data.edge_index)
#    print('Time : ', time.time()-start_time)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
#    gradient = x.grad.data
#    signed_grad = torch.sign(gradient)
#    perturbed_image = x + 0.2 * signed_grad
    optimizer.step()
    return loss.item(), 123

for epoch in range(1, 2001):
    loss, perturbed_image = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    z, _, _ = model(data.x, data.edge_index)

    if epoch % 100 == 0:
        with torch.no_grad():
#            corrupted_x = add_feature_noise(data.x, 0.0)
            import time
            start_time = time.time()
            z, _, _ = model(data.x, data.edge_index)
            print(time.time() - start_time)
            print('SVM : ', svm_test2(z.cpu().detach().numpy(), data.y.cpu().detach().numpy(), data.train_mask.cpu().detach().numpy(), data.test_mask.cpu().detach().numpy()))
            print('LogReg :',LogReg(z.cpu().detach().numpy(), data.y.cpu().detach().numpy(), data.train_mask.cpu().detach().numpy(), data.test_mask.cpu().detach().numpy()))


#with open('./DGI_cora_uni_80.txt', 'wb') as f:
#    pickle.dump([z, data.edge_index, data.x], f)

