import torch
from torch_geometric.utils import negative_sampling, to_undirected

def add_edge_noise(edge, noisy_rate=0.05):
    num_nodes = torch.max(edge)+1
    edge = edge[:, edge[0] < edge[1]]
    num_edges = edge.size()[1]

    idx = torch.randperm(num_edges)
    edge = edge[:, idx]
    edge = edge[:, :int((1-noisy_rate)*num_edges)]
    edge = to_undirected(edge, num_nodes=num_nodes)
    neg_edge = negative_sampling(edge, num_nodes=num_nodes, num_neg_samples=int(noisy_rate*num_edges))
    edge = torch.cat((edge, neg_edge), 1)
    edge = to_undirected(edge, num_nodes=num_nodes)
    return edge

def add_feature_noise(x, noisy_rate=0.01):
    x = x.detach()
    return x * (1-noisy_rate) + torch.rand_like(x) * noisy_rate         # Uniform
#    return x * (1-noisy_rate) + torch.randn_like(x) * noisy_rate       # Gaussian

    NM = x.size()[0] * x.size()[1]
    feats =  x.nonzero().t()
    num_feats = feats.size()[1]
    idx = torch.randperm(num_feats)[:int(num_feats * noisy_rate)]
    x[feats[0, idx], feats[1, idx]] = 0
    idx = torch.randperm(NM)[:int(num_feats * noisy_rate)]
    row, col  = idx // x.size()[1], idx % x.size()[1]
    x[row, col] = 1
    return x

def link_pred(z, pos_edge_index, neg_edge_index):
    from torch_geometric.nn.models import InnerProductDecoder
    from sklearn.metrics import average_precision_score, roc_auc_score

    decoder = InnerProductDecoder()
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = decoder(z, pos_edge_index, sigmoid=True)
    neg_pred = decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)

def LogReg(z, y, train_mask, test_mask, solver='lbfgs',
        multi_class='auto', max_iter=150):
    r"""Evaluates latent space quality via a logistic regression downstream
    task."""
    from sklearn.linear_model import LogisticRegression
    train_z, test_z = z[train_mask], z[test_mask]
    train_y, test_y = y[train_mask], y[test_mask]
    clf = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter).fit(train_z,
                                          train_y)
    return clf.score(test_z, test_y)

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np
def svm_test2(X, y, train_nodes, test_nodes, repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []

    macro_f1_list = []
    micro_f1_list = []
    for i in range(repeat):
        X_train, X_test = X[train_nodes], X[test_nodes]
        y_train, y_test = y[train_nodes], y[test_nodes]
        svm = LinearSVC(dual=False)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)
    result_macro_f1_list = (np.mean(macro_f1_list), np.std(macro_f1_list))
    result_micro_f1_list = (np.mean(micro_f1_list), np.std(micro_f1_list))
    return np.mean(macro_f1_list), np.mean(micro_f1_list)
