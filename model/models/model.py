import torch
import pickle
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GCN2Conv, GINConv, APPNP
from torch_geometric.utils import to_dense_adj
from models.layers import DenseGCNConv, MLP
import math
from torch_scatter import scatter_add
import scipy.sparse.linalg as sp

def SinusoidalPosEmb(x, num_steps, dim, rescale=4):
    x = x / num_steps * num_steps * rescale
    device = x.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


class Denoising_Model(torch.nn.Module):
    def __init__(self, model, nlabel, nfeat, num_layers, num_linears, data, nhid, nhead=4, skip=False):
        super(Denoising_Model, self).__init__()
        nlabel = nhid
        self.data = data
        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = int(nhid / nhead) if model == 'GATConv' else nhid
        self.nlabel = nlabel
        self.model = model
        self.pe_dim = 128 if data=='cora' else 64
        nhead = 8
        self.nhead = nhead
        self.skip = skip
        self.layers = torch.nn.ModuleList()

        for i in range(self.depth):
            if i == 0:
                self.layers.append(nn.Linear(self.nfeat+self.nlabel+self.pe_dim, self.nhid))                    
            else:
                self.layers.append(nn.Linear(self.nhid+self.nlabel+self.pe_dim, self.nhid))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        self.layers.append(nn.Linear(self.nhid+self.nlabel+self.pe_dim, self.nlabel))
        torch.nn.init.xavier_uniform_(self.layers[-1].weight)

    #-----------------------------------
        self.layers_student = torch.nn.ModuleList()
        for i in range(5):
            if i == 0:
                self.layers_student.append(nn.Linear(self.nfeat+self.pe_dim, self.nhid*2))                    
            else:
                self.layers_student.append(nn.Linear(self.nhid*2, self.nhid*2))
            torch.nn.init.xavier_uniform_(self.layers_student[-1].weight)
        self.layers_student.append(nn.Linear(self.nhid*2, self.nhid))
        torch.nn.init.xavier_uniform_(self.layers_student[-1].weight)
    #-----------------------------------

        self.activation = torch.nn.ELU()
        self.fdim = self.nhid + self.nlabel
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.nhid)
        )

    def student_forward(self, x, train=False):
        x = F.dropout(x, 0.3, training=train)
        for i in range(4):
            x = self.layers_student[i](x)
            x = F.dropout(x, 0.3, training=train)
            x = F.normalize(x, p=2, dim=1)
        pred_y = self.layers_student[-1](x)
        return pred_y

    def forward(self, x, q_Y_sample, adj, t, num_steps, train=True):
        t = SinusoidalPosEmb(t, num_steps, 128)
        t = self.time_mlp(t)
        x, pe = x[:,:-self.pe_dim], x[:,-self.pe_dim:]
    
        x_asd = torch.zeros_like(x).to(x.device)
        x = torch.cat([x, q_Y_sample, pe], dim = -1)   # X, noise_label
        for i in range(self.depth):
            x_before_act = self.layers[i](x) + t
            x = torch.cat([x_before_act, q_Y_sample, pe], dim = -1)
            del x_before_act
        pred_y = self.layers[self.depth](x)

        x = torch.cat([x_asd, q_Y_sample, pe*0], dim = -1)   # X, noise_label
        for i in range(self.depth):
            x_before_act = self.layers[i](x) + t
            x = torch.cat([x_before_act, q_Y_sample, pe*0], dim = -1)
        pred_y_uncon = self.layers[self.depth](x)
        del x, x_before_act
        pred_y = 2 * (pred_y - pred_y_uncon) + pred_y_uncon
        return pred_y