import pickle
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch_geometric.nn import APPNP
from torch_geometric.utils import negative_sampling

########################
def feature_kernel(X, feature_map, trans_X):
    """ 변환 T를 적용한 후의 특징 커널을 계산합니다. """
    transformed_features = trans_X
    return torch.mm(transformed_features, transformed_features.t())

def label_kernel(Y):
    """ 라벨 커널을 계산합니다. """
    return (Y.unsqueeze(1) == Y.unsqueeze(0)).float()

def kernel_alignment(X, Y, feature_map, trans_X):
    """ 커널 정렬 메트릭을 계산합니다. """
    Kx = feature_kernel(X, feature_map, trans_X)
    Ky = label_kernel(Y).cuda()
    alignment = torch.sum(Kx * Ky) / torch.sqrt(torch.sum(Kx * Kx) * torch.sum(Ky * Ky))
    return alignment

# 예시 특징 맵 함수 (예: 랜덤 푸리에 특징)
def example_feature_map(x):
    """ 랜덤 푸리에 특징 맵 함수입니다. """
    return torch.exp(-0.5 * (x - x.mean())**2)

# 예시 변환 함수
def example_transformation(x):
    """ 간단한 예시 변환 함수입니다. """
    return x + torch.randn_like(x) * 0.1  # 예: 작은 노이즈 추가
########################


def sum_except_batch(x, num_dims=1):
    return torch.sum(x, dim = -1)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s = 0.015):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    timesteps = (
        torch.arange(timesteps + 1, dtype=torch.float64) / timesteps + s
    )
    alphas = timesteps / (1 + s) * math.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = betas.clamp(max=0.999)
    betas = torch.cat(
            (torch.tensor([0], dtype=torch.float64), betas), 0
        )
    betas = betas.clamp(min=0.001)
    return betas


class diffusion_model(torch.nn.Module):
    def __init__(self, device, timesteps):
        super(diffusion_model, self).__init__()

        betas = cosine_beta_schedule(timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        alphas_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas[:-1]), 0
        )
        posterior_variance = betas
        self.register("betas", betas.to(device[0]))
        self.register("alphas", alphas.to(device[0]))
        self.register("alphas_prev", alphas_prev.to(device[0]))
        self.register("alphas_cumprod", alphas_cumprod.to(device[0]))
        self.register("sqrt_alphas", torch.sqrt(alphas).to(device[0]))
        self.register("sqrt_betas", torch.sqrt(betas).to(device[0]))
        self.register("alphas_cumprod_prev", alphas_cumprod_prev.to(device[0]))
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device[0]))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).to(device[0]))
        self.register("thresh", (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod)
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod).to(device[0]))
        self.register("posterior_variance", posterior_variance.to(device[0]))
        self.num_timesteps = timesteps
        self.device = device

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x, t):
        noise = torch.randn_like(x) # epsilon : noise
        return (
            self.sqrt_alphas_cumprod[t] * x
            + self.sqrt_one_minus_alphas_cumprod[t] * noise, noise
        )


class gaussian_ddpm_losses:
    def __init__(self, num_timesteps, device):
        self.diff_Y = diffusion_model(device=device, timesteps = num_timesteps)
        self.num_timesteps = num_timesteps
        self.device = device
        self.kl_loss = torch.nn.KLDivLoss('mean')
        self.nc_loss = torch.nn.CrossEntropyLoss()
        self.prop = APPNP(K=1, alpha=0.5)

    def loss_fn(self, model, x, adj, y, label, train_mask, batch = 1):
        losses = 0
        for i in range(batch):
            t = self.sample_time(self.device)
            q_Y_sample, noise = self.diff_Y.q_sample(y, t)
            eps = model(x, q_Y_sample, adj, t, self.num_timesteps, train=True)
            losses = losses + torch.mean(torch.sum((eps - noise)**2, dim = -1))    # DDIM
            del q_Y_sample
            del eps
        del x, adj
        return losses / batch
        
    def lsp(self, z1, z2, adj):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        S_z1 = torch.mm(z1, z1.T)
        S_z2 = torch.mm(z2, z2.T)
        return (S_z1 - S_z2)[adj[0], adj[1]].mean()

    def sds(self, model, x, adj, y, teacher_h, label, train_mask, batch = 1):
        losses = 0
        for i in range(batch):
            t = self.sample_time(self.device)
            q_Y_sample, noise = self.diff_Y.q_sample(y, t)
            eps = model(x, q_Y_sample, adj, t, self.num_timesteps, train=False)

            q_Y_sample = self.diff_Y.sqrt_alphas_cumprod[t] * teacher_h + self.diff_Y.sqrt_one_minus_alphas_cumprod[t] * noise
            eps2 = model(x, q_Y_sample, adj, t, self.num_timesteps, train=False)

            w = (1 - self.diff_Y.sqrt_alphas_cumprod[t])
            grad = w * (eps - eps2) / 2
            target = (y - grad).detach()
            losses = losses + torch.mean(torch.sum((y - target)**2, dim = -1))
            del q_Y_sample
            del eps
        del x, adj
        return losses / batch

    def estimate(self, model, x, adj, y, label, temp=0.01):
        updated_h = torch.randn_like(y) * temp
        for i in range(1, self.num_timesteps):
            eps = model(x, updated_h, adj, torch.tensor([self.num_timesteps-i]).to(x.device), self.num_timesteps, False)
            updated_h = (1/self.diff_Y.sqrt_alphas[self.num_timesteps-i])*(updated_h-self.diff_Y.thresh[self.num_timesteps-i]*eps)
#            if i == self.num_timesteps:
#                break
#            updated_h = self.diff_Y.sqrt_alphas[self.num_timesteps-i-1]*updated_h + self.diff_Y.sqrt_betas[self.num_timesteps-i-1]*eps
        return updated_h

    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device[0]).long()
        return t
    
    
class simple_losses:
    def __init__(self, device):
        self.device = device

    def loss_fn(self, model, x, adj, y, train_mask, batch = 1):
        h, pred_y = model(x, adj)
        losses = F.nll_loss(pred_y[train_mask], torch.argmax(y[train_mask], dim = -1))
        return h, losses
    
    def contra_loss(self, z, adj, temperature=2):
        z = F.normalize(z, p=2, dim=1)
        neg_adj = negative_sampling(adj, num_neg_samples=adj.size()[1]*3, num_nodes=z.size()[0])
        pred_pos = torch.sum(z[adj[0]] * z[adj[1]], 1)
        pred_neg = torch.sum(z[neg_adj[0]] * z[neg_adj[1]], 1)
        pred_pos = torch.exp(pred_pos / temperature).sum()
        pred_neg = torch.exp(pred_neg / temperature).sum()
        loss = -torch.log(pred_pos / (pred_pos + pred_neg))
        return loss

    def estimate(self, model, x, adj, y, train_mask):
        return model(x, adj)
