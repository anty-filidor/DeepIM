import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import time
import networkx as nx
import random
import pickle
from scipy.special import softmax
from scipy.sparse import csr_matrix
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import pandas as pd
import argparse
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

from main.utils import load_dataset, InverseProblemDataset, adj_process, diffusion_evaluation
from main.model.gat import GAT, SpGAT
from main.model.model import GNNModel, VAEModel, DiffusionPropagate, Encoder, Decoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable


os.environ["CUDA_VISIBLE_DEVICES"]="3"
print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sampling(inverse_pairs):
    diffusion_count = []
    for i, pair in enumerate(inverse_pairs):
        diffusion_count.append(pair[:, 1].sum())
    diffusion_count = torch.Tensor(diffusion_count)
    top_k = diffusion_count.topk(int(0.1*inverse_pairs.shape[0])).indices
    return top_k  # return top 10% of lagrest indices


# #############################################################################
# CLI parser
# #############################################################################
parser = argparse.ArgumentParser(description="GenIM")
datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience', 'random5']
parser.add_argument("-d", "--dataset", default="cora_ml", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="LT", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=1, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
mode = ['Normal', 'Budget Constraint']
parser.add_argument("-m", "--mode", default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))
args = parser.parse_args(args=[])
# #############################################################################


# #############################################################################
# Train VAE + Diffusion model
# #############################################################################

# read data
# 2810x2810 -> adjacency matrix [nodes_num x nodes_num]
# 2x100x2810 -> inverse_pairs [seed_set_combinations x nodes_num x start_stop_states]
with open('data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)
adj, inverse_pairs = graph['adj'], graph['inverse_pairs']
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.Tensor(adj.toarray()).to_sparse()

# data loader
if args.dataset == 'random5':
    batch_size = 2
    hidden_dim = 4096
    latent_dim = 1024
else:
    batch_size = 16
    hidden_dim = 1024
    latent_dim = 512
train_set, test_set = torch.utils.data.random_split(
    inverse_pairs, [len(inverse_pairs)-batch_size, batch_size]
)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader  = DataLoader(dataset=test_set,  batch_size=1, shuffle=False)

# decoder - encoder
encoder = Encoder(
    input_dim= inverse_pairs.shape[1], 
    hidden_dim=hidden_dim, 
    latent_dim=latent_dim
)
decoder = Decoder(
    input_dim=latent_dim, 
    latent_dim=latent_dim, 
    hidden_dim=hidden_dim, 
    output_dim=inverse_pairs.shape[1]
)
vae_model = VAEModel(Encoder=encoder, Decoder=decoder).to(device)

# diffusion model
forward_model = SpGAT(
    nfeat=1, 
    nhid=64, 
    nclass=1, 
    dropout=0.2, 
    nheads=4, 
    alpha=0.2
)
forward_model = forward_model.to(device)
forward_model.train()

# bulk optimizer for all modela
optimizer = Adam(
    [{'params': vae_model.parameters()}, {'params': forward_model.parameters()}], 
    lr=1e-4
)

adj = adj.to(device)

def loss_all(x, x_hat, y, y_hat):
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    forward_loss = F.mse_loss(y_hat, y, reduction='sum')
    # forward_loss = F.binary_cross_entropy(y_hat, y, reduction='sum')
    return reproduction_loss+forward_loss, reproduction_loss, forward_loss

# train the representation of the model and VAE
for epoch in range(600):
    total_overall = 0
    forward_loss = 0
    reproduction_loss = 0
    precision_for = 0
    recall_for = 0
    precision_re = 0
    recall_re = 0

    begin = time.time()
    for batch_idx, data_pair in enumerate(train_loader):
        # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)

        # read state of nodes before the propagation and at the steady state
        x = data_pair[:, :, 0].float().to(device)
        y = data_pair[:, :, 1].float().to(device)
        x_true = x.cpu().detach().numpy()
        y_true = y.cpu().detach().numpy()
        
        # inference from the algorithms wrt particular training samle
        optimizer.zero_grad()
        loss = 0
        for i, x_i in enumerate(x):
            y_i = y[i]
            
            # obtain estimation of the x with variational auto encoder
            x_hat = vae_model(x_i.unsqueeze(0))

            # obtain prediction of the x
            x_pred = x_hat.cpu().detach().numpy()
            x_pred[x_pred>0.01] = 1
            x_pred[x_pred!=1] = 0   

            # obtain prediciton of y with diffusion model
            y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj) 

            # compute total loss wrt representation of the seed set and diffusion prediction
            total, re, forw = loss_all(x_i.unsqueeze(0), x_hat, y_i, y_hat.squeeze(-1))        
            loss += total           
            precision_re += precision_score(x_true[i], x_pred[0], zero_division=0)
            recall_re += recall_score(x_true[i], x_pred[0], zero_division=0)
        
        # compute, propagate loss and update feature space
        total_overall += loss.item()
        loss = loss/x.size(0)
        loss.backward()
        optimizer.step()

        # clamp diffusion model params to avoid gradinent troubles
        for p in forward_model.parameters():
            p.data.clamp_(min=0)  
    end = time.time()

    print("Epoch: {}".format(epoch+1), 
          "\tTotal: {:.4f}".format(total_overall / len(train_set)),
          "\tReconstruction Precision: {:.4f}".format(precision_re / len(train_set)),
          "\tReconstruction Recall: {:.4f}".format(recall_re / len(train_set)),
          "\tTime: {:.4f}".format(end - begin)
         )

# when the VAE is trained - freeze it
for param in vae_model.parameters():
    param.requires_grad = False
# do it also for the diffusion model
for param in forward_model.parameters():
    param.requires_grad = False
# #############################################################################


# #############################################################################
# Train seed set inference (model that samples seed set)
# #############################################################################
encoder = vae_model.Encoder
decoder = vae_model.Decoder

def loss_inverse(y_true, y_hat, x_hat):
    forward_loss = F.mse_loss(y_hat, y_true)
    L0_loss = torch.sum(torch.abs(x_hat))/x_hat.shape[1]
    return forward_loss+L0_loss, L0_loss

# read data
with open('data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)
adj, inverse_pairs = graph['adj'], graph['inverse_pairs'] # what are inverse pairs
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.Tensor(adj.toarray()).to_sparse().to(device)

# obtain seeding budget
seed_num = int(inverse_pairs[0, :, 0].sum().item())  # change order!
y_true = torch.ones(inverse_pairs[0, :, 0].shape).unsqueeze(0).to(device)

# take top k (10%) of all cases where influence was highest 
topk_seed = sampling(inverse_pairs)

# predict embeddings of these sets (note 0th slice) and concatenate them
z_hat = 0
for i in topk_seed:
    z_hat += encoder(inverse_pairs[i, :, 0].unsqueeze(0).to(device))
z_hat = z_hat/len(topk_seed)
z_hat = z_hat.detach()

# initialise optimisator of the latent feature space p(z)
z_hat.requires_grad = True
z_optimizer = Adam([z_hat], lr=1e-4)

# iteratively optimise the feature space
for i in range(300):
    
    x_hat = decoder(z_hat) # sample seed set from latent space

    # regularise - check robustness of this set according to the diffusion model
    y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj) 
    y = torch.where(y_hat > 0.05, 1, 0) 
    
    # compute, propagate loss and update feature space
    loss, L0 = loss_inverse(y_true, y_hat, x_hat)
    loss.backward()
    z_optimizer.step()

    print('Iteration: {}\t Total Loss:{:.5f}'.format(i+1, loss.item()))

# obtain final, suboptimal seed set
top_k = x_hat.topk(seed_num)
seed = top_k.indices[0].cpu().detach().numpy()
# #############################################################################


# #############################################################################
# Evaluate solution - how chosen seed set performs on the real spreading model
# #############################################################################
with open('data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)
adj, inverse_pairs = graph['adj'], graph['inverse_pairs']
influence = diffusion_evaluation(adj, seed, diffusion = args.diffusion_model)
# #############################################################################
