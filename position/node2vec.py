import os.path as osp

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from utils import add_edge_noise, add_feature_noise
from torch_geometric.utils import train_test_split_edges
from torch_geometric.datasets import Planetoid, Amazon, AmazonProducts, Flickr, Coauthor, CoraFull, Reddit, WikiCS, Reddit2, LINKXDataset, WikipediaNetwork, WebKB, Actor
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

def main():
    dataName = dataset = 'Reddit'
    path = osp.join(osp.dirname("D:\\"), 'data',  dataset)

    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, dataset, 'public')
    if dataset in ['cs', 'physics']:
        dataset = Coauthor(path, dataset)
    if dataset in ['computers', 'photo']:
        dataset = Amazon(path, dataset)
    if dataset == 'Actor':
        dataset = Actor(path)
    if dataset in ['Penn94', 'Reed98']:
        dataset = LINKXDataset(path, name=dataset)
    if dataset is 'OGB':
        dataset = PygNodePropPredDataset('ogbn-arxiv', path)
        split_idx = dataset.get_idx_split()
    if dataset in ['Texas', 'Cornell', 'Wisconsin']:
        dataset = WebKB(path, name=dataset)
    if dataset == 'Reddit':
        dataset = Reddit(path)
    if dataset is 'Products':
        dataset = PygNodePropPredDataset('ogbn-products', path)
        split_idx = dataset.get_idx_split()
    from heterodataset import HeterophilousGraphDataset
    if dataset in ['roman_empire', 'amazon_ratings']:
        dataset = HeterophilousGraphDataset(path, name=dataset)
        dataName = 'roman'
    if dataset in ['IMDB']:
        import pickle
        with open("TARL_"+dataset+'_input.txt', 'rb') as f:
            data = pickle.load(f)
            edge_index = data[1]

    data = dataset[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dataName is 'Products':
        data.y = data.y.squeeze()


    torch.manual_seed(42)
    model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=20,
                     context_size=5 , walks_per_node=3,
                     num_negative_samples=5, p=2, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:

            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = 0
        return z, acc

    for epoch in range(1, 21):
        loss = train()
        z, acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    import pickle
    with open(dataName+'_pe.txt', 'wb') as f:
        pickle.dump(z, f)

if __name__ == "__main__":
    main()