"""
utility functions
"""

from __future__ import division
import scipy
from scipy.stats import sem
import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.preprocessing import OneHotEncoder

class MaxNFEException(Exception): pass


def rms_norm(tensor):
  return tensor.pow(2).mean().sqrt()

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def make_norm(state):
  if isinstance(state, tuple):
    state = state[0]
  state_size = state.numel()

  def norm(aug_state):
    y = aug_state[1:1 + state_size]
    adj_y = aug_state[1 + state_size:1 + 2 * state_size]
    return max(rms_norm(y), rms_norm(adj_y))

  return norm


def print_model_params(model):
  total_num_params = 0
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      total_num_params += param.numel()
  print("Model has a total of {} params".format(total_num_params))


def adjust_learning_rate(optimizer, lr, epoch, burnin=50):
  if epoch <= burnin:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr * epoch / burnin


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not int(fill_value) == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-0.5)
  deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
  return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def coo2tensor(coo, device=None):
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  values = coo.data
  v = torch.FloatTensor(values)
  shape = coo.shape
  print('adjacency matrix generated with shape {}'.format(shape))
  # test
  return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_sym_adj(data, opt, improved=False):
  edge_index, edge_weight = gcn_norm(  # yapf: disable
    data.edge_index, data.edge_attr, data.num_nodes,
    improved, opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  return coo2tensor(coo)


def get_rw_adj_old(data, opt):
  if opt['self_loop_weight'] > 0:
    edge_index, edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                       fill_value=opt['self_loop_weight'])
  else:
    edge_index, edge_weight = data.edge_index, data.edge_attr
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  normed_csc = normalize(coo, norm='l1', axis=0)
  return coo2tensor(normed_csc.tocoo())


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)

#  if norm_dim != 2:
#    deg_inv_sqrt = deg.pow_(-1)
#    edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]

  deg_inv_sqrt = deg.pow_(-0.5)
  edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
  return edge_index, edge_weight


def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  return torch.sparse.FloatTensor(i, v * d, s.size())


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval


def get_full_adjacency(num_nodes):
  # what is the format of the edge index?
  edge_index = torch.zeros((2, num_nodes ** 2),dtype=torch.long)
  for idx in range(num_nodes):
    edge_index[0][idx * num_nodes: (idx + 1) * num_nodes] = idx
    edge_index[1][idx * num_nodes: (idx + 1) * num_nodes] = torch.arange(0, num_nodes,dtype=torch.long)
  return edge_index



from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr


# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
  r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
  out = src - src.max()
  # out = out.exp()
  out = (out + torch.sqrt(out ** 2 + 4)) / 2

  if ptr is not None:
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
  elif index is not None:
    N = maybe_num_nodes(index, num_nodes)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
  else:
    raise NotImplementedError

  return out / (out_sum + 1e-16)


# Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val


class DummyDataset(object):
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class DummyData(object):
  def __init__(self, edge_index=None, edge_Attr=None, num_nodes=None):
    self.edge_index = edge_index
    self.edge_attr = edge_Attr
    self.num_nodes = num_nodes


import torch
import numpy as np
import scipy.sparse as sp
import numpy.random as random
from torch_geometric.utils import negative_sampling
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score

def accuracy(pred, target):
    r"""Computes the accuracy of correct predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()



def true_positive(pred, target, num_classes):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)



def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)



def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)



def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)



def precision(pred, target, num_classes):
    r"""Computes the precision:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out



def recall(pred, target, num_classes):
    r"""Computes the recall:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out



def f1_score_calc(pred, target, num_classes):
    r"""Computes the :math:`F_1` score:
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score

def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


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

def svm_test(X, y, test_sizes=(0.4, 0.6, 0.8, 0.9), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list

def evaluate_results_nc(embeddings, labels, num_classes, mode='test'):
    svm_macro_f1_list, svm_micro_f1_list = [], []
    nmi_mean, nmi_std, ari_mean, ari_std = 0,0,0,0
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)

    if mode=='test':
        print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
        print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std

def make_negative_edge(src, dst, N, neg_sample):
    length = len(src)
    num_nodes = N

    ########################
    N,M=0,0
    node2index_s = {}
    index2node_s = {}
    node2index_d = {}
    index2node_d = {}

    index2node_s_list = []
    index2node_d_list = []

    for i in src.tolist():
        if i not in index2node_s:
            node2index_s[len(node2index_s)] = i
            index2node_s[i] = len(index2node_s)
            N+=1
        index2node_s_list.append(index2node_s[i])

    index2node_s_tensor = torch.zeros(torch.max(src)+1).cuda(0)
    for i in src.tolist():
        index2node_s_tensor[index2node_s[i]] = i

    for i in dst.tolist():
        if i not in index2node_d:
            node2index_d[len(node2index_d)] = i
            index2node_d[i] = len(index2node_d)
            M+=1
        index2node_d_list.append(index2node_d[i])

    index2node_d_tensor = torch.zeros(torch.max(dst)+1).cuda(0)
    for i in dst.tolist():
        index2node_d_tensor[index2node_d[i]] = i


    src_ = []
    dst_ = []
    src = index2node_s_list
    dst = index2node_d_list
    ########################

    data = np.ones(length)
    src, dst = np.array(src), np.array(dst)

    neg_src,neg_dst,neg_type = [],[],[]

    #   Sparse
#    src,dst = torch.LongTensor(src), torch.LongTensor(dst)
#    edge_index = torch.cat((src.unsqueeze(0), dst.unsqueeze(0)),0)
#    neg_edge = negative_sampling(edge_index, num_nodes=len(index2node_d), num_neg_samples=neg_sample*len(data))
#    if len(neg_edge[0]) == 0:
#        return torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])
#    neg_src = index2node_s_tensor[neg_edge[0]].long()
#    neg_dst = index2node_d_tensor[neg_edge[1]].long()
#    return neg_src, neg_dst

    #   Dense
    A = sp.csc_matrix((data, (src,dst)), shape=(N,M)).todense()
    tmp = (1-A).nonzero()
    tmp = torch.cat((torch.LongTensor(tmp[0]).unsqueeze(0),torch.LongTensor(tmp[1]).unsqueeze(0)),0)#.cuda()

    np.random.seed(np.random.randint(1,100))
    if len(tmp[0]) == 0:
        return torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])
    index = np.random.choice(len(tmp[0]), neg_sample * len(src))

    index = torch.LongTensor(index)
    node2index_s = index2node_s_tensor[tmp[0][index]].long()
    node2index_d = index2node_d_tensor[tmp[1][index]].long()

    del A
    del tmp
    del src_
    del dst_
    del index2node_s_list
    del index2node_d_list
    del index
    return node2index_s, node2index_d

def make_output_files(edges, dataset):
    with open('data/'+dataset+'/train_made.txt','w') as f:
        for i in range(edges['pos_train'].size()[1]):
            string = '%d %d %d\n' %((edges['type_train'][i]+2)//2, edges['pos_train'][0][i], edges['pos_train'][1][i])
            f.write(string)

    with open('data/'+dataset+'/valid_made.txt','w') as f:
        for i in range(edges['pos_val'].size()[1]):
            string = '%d %d %d 1\n' %((edges['type_val'][i]+2)//2, edges['pos_val'][0][i], edges['pos_val'][1][i])
            f.write(string)
        for i in range(edges['neg_val'].size()[1]):
            string = '%d %d %d 0\n' %((edges['type_val'][i]+2)//2, edges['neg_val'][0][i], edges['neg_val'][1][i])
            f.write(string)

    with open('data/'+dataset+'/test_made.txt','w') as f:
        for i in range(edges['pos_test'].size()[1]):
            string = '%d %d %d 1\n' %((edges['type_test'][i]+2)//2, edges['pos_test'][0][i], edges['pos_test'][1][i])
            f.write(string)
        for i in range(edges['neg_test'].size()[1]):
            string = '%d %d %d 0\n' %((edges['type_test'][i]+2)//2, edges['neg_test'][0][i], edges['neg_test'][1][i])
            f.write(string)


def make_data(edges):
    train_data, val_data, test_data = '','',''
    for i in range(len(edges['pos_type_train'])):
        train_data = train_data + '%d %d %d\n' %(int(edges['pos_type_train'][i]//2+1), int(edges['pos_train'][0][i]), int(edges['pos_train'][1][i]))
    with open('train.txt','w') as f:
        f.write(train_data)

    for i in range(len(edges['pos_type_val'])):
        val_data = val_data + '%d %d %d 1\n' %(int(edges['pos_type_val'][i]//2+1), int(edges['pos_val'][0][i]), int(edges['pos_val'][1][i]))
    for i in range(len(edges['neg_type_val'])):
        val_data = val_data + '%d %d %d 0\n' %(int(edges['neg_type_val'][i]//2+1), int(edges['neg_val'][0][i]), int(edges['neg_val'][1][i]))
    with open('valid.txt','w') as f:
        f.write(val_data)

    for i in range(len(edges['pos_type_test'])):
        test_data = test_data + '%d %d %d 1\n' %(int(edges['pos_type_test'][i]//2+1), int(edges['pos_test'][0][i]), int(edges['pos_test'][1][i]))
    for i in range(len(edges['neg_type_test'])):
        test_data = test_data + '%d %d %d 0\n' %(int(edges['neg_type_test'][i]//2+1), int(edges['neg_test'][0][i]), int(edges['neg_test'][1][i]))
    with open('test.txt','w') as f:
        f.write(test_data)


def make_data_divides(edges):
    train_data, val_data, test_data = {}, {}, {}
    for i in range(len(edges['pos_type_train'])):
        train_data = train_data + '%d %d %d\n' %(int(edges['pos_type_train'][i]//2+1), int(edges['pos_train'][0][i]), int(edges['pos_train'][1][i]))
    with open('train.txt','w') as f:
        f.write(train_data)

    for i in range(len(edges['pos_type_val'])):
        val_data = val_data + '%d %d %d 1\n' %(int(edges['pos_type_val'][i]//2+1), int(edges['pos_val'][0][i]), int(edges['pos_val'][1][i]))
    for i in range(len(edges['neg_type_val'])):
        val_data = val_data + '%d %d %d 0\n' %(int(edges['neg_type_val'][i]//2+1), int(edges['neg_val'][0][i]), int(edges['neg_val'][1][i]))
    with open('valid.txt','w') as f:
        f.write(val_data)

    for i in range(len(edges['pos_type_test'])):
        test_data = test_data + '%d %d %d 1\n' %(int(edges['pos_type_test'][i]//2+1), int(edges['pos_test'][0][i]), int(edges['pos_test'][1][i]))
    for i in range(len(edges['neg_type_test'])):
        test_data = test_data + '%d %d %d 0\n' %(int(edges['neg_type_test'][i]//2+1), int(edges['neg_test'][0][i]), int(edges['neg_test'][1][i]))
    with open('test.txt','w') as f:
        f.write(test_data)



import networkx as nx
import numpy as np
import scipy
import pickle


def load_IMDB_data2(prefix='data/preprocessed/IMDB_processed'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
                                                          
    edge_metapath_indices_list = [[idx00, idx01], [idx10, idx11], [idx20, idx21]]
    N = adjM.shape[0]
    adjM = np.zeros((N,N))

    for i in range(2):
        meta_length = edge_metapath_indices_list[i][0].shape[1]
        for j in range(4019):
            for k in range(edge_metapath_indices_list[i][j].shape[0]):
                for l in range(meta_length-1):
                    adjM[edge_metapath_indices_list[i][j][k][l]][edge_metapath_indices_list[i][j][k][l+1]] = 1                  
                             
                                 
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_LastFM_data(prefix='data/preprocessed/LastFM_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_user_artist.npz')
    train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_user_artist.npz')

    return [[adjlist00, adjlist01, adjlist02],[adjlist10, adjlist11, adjlist12]],\
           [[idx00, idx01, idx02], [idx10, idx11, idx12]],\
           adjM, type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist


# load skipgram-format embeddings, treat missing node embeddings as zero vectors
def load_skipgram_embedding(path, num_embeddings):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings = np.zeros((num_embeddings, dim))
        for line in infile.readlines():
            count += 1
            line = line.strip().split(' ')
            embeddings[int(line[0])] = np.array(list(map(float, line[1:])))
    print('{} out of {} nodes have non-zero embeddings'.format(count, num_embeddings))
    return embeddings


# load metapath2vec embeddings
def load_metapath2vec_embedding(path, type_list, num_embeddings_list, offset_list):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings_dict = {type: np.zeros((num_embeddings, dim)) for type, num_embeddings in zip(type_list, num_embeddings_list)}
        offset_dict = {type: offset for type, offset in zip(type_list, offset_list)}
        for line in infile.readlines():
            line = line.strip().split(' ')
            # drop </s> token
            if line[0] == '</s>':
                continue
            count += 1
            embeddings_dict[line[0][0]][int(line[0][1:]) - offset_dict[line[0][0]]] = np.array(list(map(float, line[1:])))
    print('{} node embeddings loaded'.format(count))
    return embeddings_dict


def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'data/wordvec/GloVe/glove.6B.50d.txt',
        100: 'data/wordvec/GloVe/glove.6B.100d.txt',
        200: 'data/wordvec/GloVe/glove.6B.200d.txt',
        300: 'data/wordvec/GloVe/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs


def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx



import networkx as nx
import numpy as np
import scipy
import pickle
import torch


def load_ACM_data(prefix='data/preprocessed/ACM_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()

    emb = np.load(prefix + '/metapath2vec_emb.npy')


    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    N = features_0.shape[0] + features_1.shape[0] + features_2.shape[0]
    adj = np.zeros([N, N])
    for i in range(len(idx00)):
        adj[idx00[i][:,0], idx00[i][:,1]] = 1
        adj[idx00[i][:,1], idx00[i][:,2]] = 1
    for i in range(len(idx01)):
        adj[idx01[i][:,0], idx01[i][:,1]] = 1
        adj[idx01[i][:,1], idx01[i][:,2]] = 1

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01], \
           [idx00, idx01], \
           [features_0, features_1, features_2], emb, \
           adj, \
           type_mask, \
           labels, \
           train_val_test_idx

def load_aminer(ratio, type_num):
    # The order of node types: 0 p 1 a 2 r
    path = "./data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    N = feat_p.shape[0] + feat_a.shape[0] + feat_r.shape[0]
    adjM = np.zeros([N,N])
    for edges in [nei_a, nei_r]:
        for i in range(len(nei_a)):
            adjM[i, edges[i]] = 1
            adjM[edges[i], i] = 1

    N = feat_p.shape[0] + feat_a.shape[0] + feat_r.shape[0]
    cnt_array = [feat_p.shape[0], feat_a.shape[0], feat_r.shape[0]]
    adjM = np.zeros([N,N])
    cnt = 0
    M=0
    for edges in [nei_a, nei_r]:
        M += cnt_array[cnt]
        cnt += 1
        for i in range(len(nei_a)):
            adjM[i, M+edges[i]] = 1
            adjM[M+edges[i], i] = 1


    label = torch.FloatTensor(label)
    nei_a = [torch.LongTensor(i) for i in nei_a]
    nei_r = [torch.LongTensor(i) for i in nei_r]
    feat_p = (preprocess_features(feat_p))
    feat_a = (preprocess_features(feat_a))
    feat_r = (preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    return [], \
           [], \
           [feat_p, feat_a, feat_r], [], \
           adjM, \
           [], \
           np.argmax(label,1), \
           {'train_idx':train[0], 'val_idx':val[0], 'test_idx':test[0]}


def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "./data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    N = feat_m.shape[0] + feat_d.shape[0] + feat_a.shape[0] + feat_w.shape[0]
    cnt_array = [feat_m.shape[0], feat_d.shape[0], feat_a.shape[0], feat_w.shape[0]]
    adjM = np.zeros([N,N])
    cnt = 0
    M=0
    for edges in [nei_d, nei_a, nei_w]:
        M += cnt_array[cnt]
        cnt += 1
        for i in range(len(nei_a)):
            adjM[i, M+edges[i]] = 1
            adjM[M+edges[i], i] = 1

    label = torch.FloatTensor(label)
    nei_d = [torch.LongTensor(i) for i in nei_d]
    nei_a = [torch.LongTensor(i) for i in nei_a]
    nei_w = [torch.LongTensor(i) for i in nei_w]
    feat_m = torch.FloatTensor(preprocess_features(feat_m))
    feat_d = torch.FloatTensor(preprocess_features(feat_d))
    feat_a = torch.FloatTensor(preprocess_features(feat_a))
    feat_w = torch.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    tmp_train = np.zeros(feat_m.shape[0])
    tmp_val = np.zeros(feat_m.shape[0])
    tmp_test = np.zeros(feat_m.shape[0])
    tmp_train[train[0]] = 1
    tmp_train[train[1]] = 1
    tmp_train[train[2]] = 1
    tmp_not_train = 1-tmp_train
    val_test = tmp_not_train.nonzero()[0]
    np.random.shuffle(val_test)

    train[0] = tmp_train.nonzero()[0]
    val[0] = val_test[:val_test.shape[0]//2]
    test[0] = val_test[:val_test.shape[0]//2]

#    train = torch.cat([torch.LongTensor(i) for i in t^rain])
#    val = torch.cat([torch.LongTensor(i) for i in val])
#    test = torch.cat([torch.LongTensor(i) for i in test])

    return [], \
           [], \
           [feat_m, feat_d, feat_a, feat_w], [], \
           adjM, \
           [], \
           np.argmax(label,1), \
           {'train_idx':train[0], 'val_idx':val[0], 'test_idx':test[0]}




import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def two_dim_embedding_plot2(z, transformed_z, label, num_classes, colors=['green','red', 'blue', 'purple']):
    model = PCA(n_components=2)
    model.fit(z.detach().cpu().numpy())
    z = model.transform(z.detach().cpu().numpy())
    transformed_z = model.transform(transformed_z[4057:].detach().cpu().numpy())

    for i in range(num_classes-1, -1, -1):
        x1 = z[label==i,0]    
        x2 = z[label==i,1] 
        plt.scatter(x1, x2, s=8, color=colors[i])
    plt.show()

    for i in range(num_classes-1, -1, -1):
        x1 = z[label==i,0]    
        x2 = z[label==i,1] 
        plt.scatter(x1, x2, s=8, color=colors[i])
    x1 = transformed_z[:,0]    
    x2 = transformed_z[:,1] 
    plt.scatter(x1, x2, s=8, color='yellow')
    plt.savefig('out.jpg')
    plt.show()


def make_edge(edges, edge_type, N, dataset):
    num_type = edge_type.max()+1
    adj = []
    for i in range(num_type):
        row = np.array(edges[0][edge_type==i].cpu())
        col = np.array(edges[1][edge_type==i].cpu())
        data = np.array([1 for x in row])
        index = torch.LongTensor((row,col))
        v = torch.FloatTensor(data)
        tmp = torch.sparse.FloatTensor(index,v, torch.Size([N,N]))
        adj.append(tmp)
    if dataset == 'DBLP':
        metapaths = [[0,1],[0,2,4,1],[0,3,5,1]]
        paths = [[0],[1],[2],[3],[4],[5]]
    if dataset == 'IMDB':
#         metapaths = [[0,2],[1,3],[2,0],[3,1],[2,1,3,0],[3,0,2,1]] 
         metapaths = [[0,2],[1,3]] 
         paths = [[0],[1],[2],[3]]  # 0:M-D,  1:M-A,  2:D-M,  3:A-M
        # 02 : MDM, 13 : MAM, 2130 : DMAMD, 3021 : AMDMA
    if dataset == 'ACM':
         metapaths = [[0,2],[1,3]] 
         paths = [[0],[1],[2],[3]]  # 0:M-D,  1:M-A,  2:D-M,  3:A-M
    if dataset == 'FREEBASE':
        paths = [[0],[1],[2],[3],[4],[5]]
        metapaths = [[0],[1]]
    if dataset == 'AMINER':
        paths = [[0],[1],[2],[3]]
        metapaths = [[0,2],[1,3]]

    for j,metapath in enumerate(metapaths):
        tmp = adj[metapath[-1]].to_dense()
        for i in range(len(metapath)-2,-1,-1):
            tmp = torch.spmm(adj[metapath[i]],tmp)
        
        tmp = tmp.nonzero().to('cuda')
        tmp_ = torch.ones(tmp.size()[0]).long().to('cuda') * j

        if j == 0:
            meta_edge = tmp
            meta_edge_type = tmp_ 

        else:
            meta_edge = torch.cat((meta_edge, tmp),0)
            meta_edge_type = torch.cat((meta_edge_type, tmp_))
        del tmp, tmp_

    for j,path in enumerate(paths):
        tmp = adj[path[-1]].to_dense()
        for i in range(len(path)-2,-1,-1):
            tmp = torch.spmm(adj[path[i]],tmp)
        
        tmp = tmp.nonzero().to('cuda')
        tmp_ = torch.ones(tmp.size()[0]).long().to('cuda') * j

        if j == 0:
            edge = tmp
            edge_type = tmp_

        else:
            edge = torch.cat((edge, tmp),0)
            edge_type = torch.cat((edge_type, tmp_))
        del tmp, tmp_

    return edge.t(), edge_type, meta_edge.t(), meta_edge_type

def make_negative_edge(src, dst, N, neg_sample):
    length = len(src)

    ########################
    N,M=0,0
    node2index_s = {}
    index2node_s = {}
    node2index_d = {}
    index2node_d = {}

    index2node_s_list = []
    index2node_d_list = []

    for i in src.tolist():
        if i not in index2node_s:
            node2index_s[len(node2index_s)] = i
            index2node_s[i] = len(index2node_s)
            N+=1
        index2node_s_list.append(index2node_s[i])

    index2node_s_tensor = torch.zeros(torch.max(src)+1).cuda(0)
    for i in src.tolist():
        index2node_s_tensor[index2node_s[i]] = i

    for i in dst.tolist():
        if i not in index2node_d:
            node2index_d[len(node2index_d)] = i
            index2node_d[i] = len(index2node_d)
            M+=1
        index2node_d_list.append(index2node_d[i])

    index2node_d_tensor = torch.zeros(torch.max(dst)+1).cuda(0)
    for i in dst.tolist():
        index2node_d_tensor[index2node_d[i]] = i


    src_ = []
    dst_ = []
    src = index2node_s_list
    dst = index2node_d_list
    ########################

    data = np.ones(length)
    src, dst = np.array(src), np.array(dst)
#    src, dst = src.cpu().detach().numpy(), dst.cpu().detach().numpy()
    A = sp.csc_matrix((data, (src,dst)), shape=(N,M)).todense()

    neg_src,neg_dst,neg_type = [],[],[]

    tmp = (1-A).nonzero()
    tmp = torch.cat((torch.LongTensor(tmp[0]).unsqueeze(0),torch.LongTensor(tmp[1]).unsqueeze(0)),0)#.cuda()
    index = np.random.choice(len(tmp[0]), neg_sample * len(src))

    index = torch.LongTensor(index)
    node2index_s = index2node_s_tensor[tmp[0][index]].long()
    node2index_d = index2node_d_tensor[tmp[1][index]].long()

#    for i in range(len(src)):
#        for j in range(neg_sample):
#            u = src[i]
#            v = np.random.choice((A[u]-1).nonzero()[1], 1)

#            neg_src.append(node2index_s[u])
#            neg_dst.append(node2index_d[v])
#            neg_type.append(types)

    del A,tmp
    del src_
    del dst_
    del index2node_s_list
    del index2node_d_list
    del index

    return node2index_s, node2index_d
#    return torch.LongTensor(neg_src), torch.LongTensor(neg_dst), torch.LongTensor(neg_type)


def load_CODA_data(prefix='data/CODA'):
    N = 288092
    edge_dict = {}
    for i in range(N):
        edge_dict[i] = []

    f = open(prefix+"/train.txt", "r")
    while True:
        line = f.readline()
        if not line: break
        src, dst, type = line.split('\t')
        src, dst = int(src), int(dst)
        edge_dict[src].append(dst)
        edge_dict[dst].append(src)
    f.close()

    f = open(prefix+"/valid.txt", "r")
    while True:
        line = f.readline()
        if not line: break
        src, dst, type = line.split('\t')
        src, dst = int(src), int(dst)
        edge_dict[src].append(dst)
        edge_dict[dst].append(src)
    f.close()

    f = open(prefix+"/test.txt", "r")
    while True:
        line = f.readline()
        if not line: break
        src, dst, type = line.split('\t')
        src, dst = int(src), int(dst)
        edge_dict[src].append(dst)
        edge_dict[dst].append(src)
    f.close()

def barlow_loss(z1, z2):
   N, dim = z1.size()
   z1 = (z1 - z1.mean(0)) / z1.std(0)
   z2 = (z2 - z2.mean(0)) / z2.std(0)
   corr = z1.T @ z2 / N
   I = torch.eye(dim).cuda()
   loss = torch.norm(corr-I, p=2)
   return loss * 0.01

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


def add_feature_noise(x, noisy_rate=0.01):
    seed = torch.randint(0, 1001, (1,))
    torch.manual_seed(seed[0])
    I = torch.eye(x.size()[0]).cuda()
    return x.detach() + torch.randn_like(x).detach() * noisy_rate
    
    NM = x.size()[0] * x.size()[1]
    feats =  x.nonzero().t()
    num_feats = feats.size()[1]
    idx = torch.randperm(num_feats)[:int(num_feats * noisy_rate)]
    x[feats[0, idx], feats[1, idx]] = 0

    idx = torch.randperm(NM)[:int(num_feats * noisy_rate)]
    row, col  = torch.div(idx, x.size()[1], rounding_mode='floor'), torch.remainder(idx, x.size()[1])
    x[row, col] = 1
    return x

import seaborn as sns
def two_dim_embedding_plot(z, label, num_classes, colors=['green','red', 'blue', 'purple', 'pink', 'yellow']):
    plt.figure(figsize=(3, 2))
    qualitative_colors = sns.color_palette("Set3", 10)
    sns.set_palette("deep")
    colors = [sns.color_palette()[0],sns.color_palette()[1],sns.color_palette()[2],sns.color_palette()[3],sns.color_palette()[4],sns.color_palette()[5],sns.color_palette()[6]] 
    model = TSNE(n_components=3, perplexity=3, n_iter=300)
    sns.set_style("white")
    z = model.fit_transform(z.detach().cpu().numpy())
    
    for i in range(num_classes):
        x1 = z[(label.cpu().numpy()==i).nonzero(),0]
        x2 = z[(label.cpu().numpy()==i).nonzero(),1]
        x3 = z[(label.cpu().numpy()==i).nonzero(),2]
        plt.scatter(x1, x2, x3, color=colors[i])

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0)  # Adjust the width as necessary

    plt.savefig('out.jpg')
    plt.show()