import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from sklearn.neighbors import kneighbors_graph

from models.concare import ConCareLayer
from models.rnn import RNNLayer
from models.utils import generate_mask, get_last_visit


def random_init(dataset, num_centers, device):
    num_points = dataset.size(0) # batch_size
    dimension = dataset.size(1) # context_dim
    indices = torch.tensor(np.array(random.sample(range(num_points), k=num_centers)), dtype=torch.long)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension).to(device=device))
    return centers


# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)

    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes


# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float).to(device=device)
    cnt = torch.zeros(num_centers, dtype=torch.float)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension).to(device=device), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float))
    centers /= cnt.view(-1, 1).to(device=device)
    return centers


def cluster(dataset, num_centers, device):
    centers = random_init(dataset, num_centers, device)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers, device)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            break
        if num_iterations > 1000:
            break
        codes = new_codes
    return centers, codes


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter("bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x, device):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float().to(device=device)
        else:
            return output


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GRASPLayer(nn.Module):
    """GRASPLayer layer.

    Paper: Liantao Ma et al. GRASP: generic framework for health status representation learning based on incorporating knowledge from similar patients. AAAI 2021.

    This layer is used in the GRASP model. But it can also be used as a
    standalone layer.

    Args:
        lab_dim: dynamic feature size.
        demo_dim: static feature size, if 0, then no static feature is used.
        hidden_dim: hidden dimension of the GRASP layer, default 128.
        cluster_num: number of clusters, default 12. The cluster_num should be no more than the number of samples.
        dropout: dropout rate, default 0.5.
        block: the backbone model used in the GRASP layer ('ConCare', 'LSTM' or 'GRU'), default 'ConCare'.

    Examples:
        >>> from pyhealth.models import GRASPLayer
        >>> x = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = GRASPLayer(64, cluster_num=2)
        >>> c = layer(x)
        >>> c.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        hidden_dim: int = 128,
        cluster_num: int = 2,
        dropout: int = 0.5,
        block: str = "GRU",
    ):
        super(GRASPLayer, self).__init__()

        self.lab_dim = lab_dim
        self.demo_dim = demo_dim
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num
        self.dropout = dropout
        self.block = block

        if self.block == "ConCare":
            self.backbone = ConCareLayer(lab_dim, demo_dim, hidden_dim, hidden_dim, dropout=0)
        elif self.block == "GRU":
            self.backbone = RNNLayer(lab_dim, hidden_dim, rnn_type="GRU", dropout=0)
        elif self.block == "LSTM":
            self.backbone = RNNLayer(lab_dim, hidden_dim, rnn_type="LSTM", dropout=0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.weight1 = nn.Linear(self.hidden_dim, 1)
        self.weight2 = nn.Linear(self.hidden_dim, 1)
        self.GCN = GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True)
        self.GCN.initialize_parameters()
        self.GCN_2 = GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True)
        self.GCN_2.initialize_parameters()
        self.A_mat = None

        self.bn = nn.BatchNorm1d(self.hidden_dim)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)

        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device):
        y = logits + self.sample_gumbel(logits.size()).to(device=device)
        return torch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, device, hard=False):
        """
        ST-gumple-softmax
        x: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, device)

        if not hard:
            return y.view(-1, self.cluster_num)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
        """
        # rnn will only apply dropout between layers
        if self.block == "ConCare":
            hidden_t, _ = self.backbone(x, mask=mask, static=static)
        else:
            _, hidden_t = self.backbone(x, mask)
        hidden_t = torch.squeeze(hidden_t, 0)

        centers, codes = cluster(hidden_t, self.cluster_num, x.device)

        if self.A_mat == None:
            A_mat = np.eye(self.cluster_num)
        else:
            A_mat = kneighbors_graph(np.array(centers.detach().cpu().numpy()),20,mode="connectivity",include_self=False,).toarray()

        adj_mat = torch.tensor(A_mat).to(device=x.device)

        e = self.relu(torch.matmul(hidden_t, centers.transpose(0, 1)))  # b clu_num

        scores = self.gumbel_softmax(e, temperature=1, device=x.device, hard=True)
        digits = torch.argmax(scores, dim=-1)  #  b

        h_prime = self.relu(self.GCN(adj_mat, centers, x.device))
        h_prime = self.relu(self.GCN_2(adj_mat, h_prime, x.device))

        clu_appendix = torch.matmul(scores, h_prime)

        weight1 = torch.sigmoid(self.weight1(clu_appendix))
        weight2 = torch.sigmoid(self.weight2(hidden_t))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        final_h = weight1 * clu_appendix + weight2 * hidden_t
        out = final_h
        return out


class GRASP(nn.Module):
    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        hidden_dim: int = 64,
        cluster_num: int = 2,
        dropout: int = 0.5,
        block: str = "GRU",
        **kwargs
    ):
        super(GRASP, self).__init__()
        self.lab_dim = lab_dim
        self.demo_dim = demo_dim
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num
        self.dropout = dropout
        self.block = block

        self.grasp_encoder = GRASPLayer(lab_dim, demo_dim, hidden_dim, cluster_num, dropout, block)
    
    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        batch_size, time_steps, _ = x.size()
        # out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        # for cur_time in range(time_steps):
        #     cur_x = x[:, :cur_time+1, :]
        #     cur_mask = mask[:, :cur_time+1]
        #     cur_out = self.grasp_encoder(cur_x, static, cur_mask)
        #     out[:, cur_time, :] = cur_out
        out = self.grasp_encoder(x, static, mask)
        return out