import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, BatchNorm2d, ModuleList, Parameter
import torch.nn.functional as F
import math
from tqdm import trange
import time
from torch.autograd import Variable
import numpy as np


class AdaptiveWaveletLayer(nn.Module):
    """
    SEA-GWNN lifting layer supporting input of shape (B,N,T,C) (or any order via `_format`).
    Adjacency is provided dynamically per forward call (learnable/adaptive).
    """
    def __init__(self, nnode, in_features, out_features, hop, alpha,
                 residual=False, variant=False, leaky_alpha=0.2, alp=0.9):
        super().__init__()
        self.variant = variant
        self.nnode = nnode
        self.alpha_ = alpha
        self.hop = hop
        self.alp = alp
        self.leaky_alpha = leaky_alpha

        self.in_features = 2*in_features if variant else in_features
        self.out_features = out_features
        self.residual = residual

        # Attention vector (used in adaptive attention)
        self.a = nn.Parameter(torch.empty(size=(2*self.in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.leaky_alpha)

        # Lifting parameters
        self.temp = Parameter(torch.Tensor(self.hop+1))
        Temp = self.alp * np.arange(self.hop+1)
        Temp = Temp / np.sum(np.abs(Temp)) if np.sum(np.abs(Temp)) > 0 else Temp
        self.cheb = Parameter(torch.tensor(Temp, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(0.0)
        for k in range(self.hop+1):
            self.cheb.data[k] = self.alp*(1-self.alp)**k
        self.cheb.data[-1] = (1-self.alp)**2

    # ---------- Attention ----------
    def attention(self, x_BTNC, adj):
        """
        x_BTNC: (B,T,N,C)
        adj: (N,N) or (B,N,N)
        returns U, P: (B,T,N,N)
        """
        B, T, N, C = x_BTNC.shape
        device = x_BTNC.device
        a1 = self.a[:self.in_features, :]   # (C,1)
        a2 = self.a[self.in_features:, :]   # (C,1)

        feat_1 = torch.matmul(x_BTNC, a1)                # (B,T,N,1)
        feat_2 = torch.matmul(x_BTNC, a2)                # (B,T,N,1)
        e = feat_1 + feat_2.transpose(-2, -1)            # (B,T,N,N)
        e = self.leakyrelu(e)

        # Build mask from adjacency
        if adj.dim() == 2:       # shared adjacency
            mask = (adj > 0).view(1,1,N,N).expand(B,T,-1,-1)
        elif adj.dim() == 3:     # batch-specific adjacency
            mask = (adj > 0).unsqueeze(1).expand(-1,T,-1,-1)  # (B,T,N,N)
        else:
            raise ValueError("adj must be (N,N) or (B,N,N)")
        
        # # Make sure mask is on the same device
        # mask = mask.to(device)

        neg_inf = torch.finfo(e.dtype).min
        e_masked = torch.where(mask, e, e.new_full((), neg_inf))

        U = torch.softmax(e_masked, dim=-1)  # (B,T,N,N)
        P = 0.5 * U
        return U, P

    # ---------- Lifting ----------
    def forward_lifting_bases(self, x_BTNC, h0_BTNC, P_BTNN, U_BTNN, adj):
        """
        x_BTNC: (B,T,N,C)
        h0_BTNC: (B,T,N,C)
        P_BTNN: (B,T,N,N)
        U_BTNN: (B,T,N,N)
        adj: (N,N) or (B,N,N)
        """
        B, T, N, C = x_BTNC.shape
        coe = torch.sigmoid(self.temp)   # (hop+1,)
        cheb_coe = torch.sigmoid(self.cheb)

        # Mix adjacency with P
        if adj.dim() == 2:
            AdjP = adj.view(1,1,N,N) * P_BTNN
        elif adj.dim() == 3:
            AdjP = adj.unsqueeze(1) * P_BTNN
        else:
            raise ValueError("adj must be (N,N) or (B,N,N)")

        rowsum = AdjP.sum(-1)   # (B,T,N)

        update = x_BTNC
        feat_prime = None

        for step in range(self.hop):
            # update = U @ update
            update = torch.einsum("btij,btjc->btic", U_BTNN, update)

            if self.alpha_ is None:
                feat_even_bar = coe[0]*x_BTNC + update
            else:
                feat_even_bar = update

            if step >= 1:
                rowsum = cheb_coe[step-1] * rowsum

            feat_odd_bar = update - feat_even_bar * rowsum.unsqueeze(-1)

            if step == 0:
                if self.alpha_ is None:
                    feat_fuse = coe[1]*feat_even_bar + (1-coe[1])*feat_odd_bar
                    feat_prime = coe[2]*x_BTNC + (1-coe[2])*feat_fuse
                else:
                    feat_fuse = self.alpha_*feat_even_bar + (1-self.alpha_)*feat_odd_bar
                    feat_prime = self.alpha_*x_BTNC + (1-self.alpha_)*feat_fuse
            else:
                if self.alpha_ is None:
                    feat_fuse = coe[1]*feat_even_bar + (1-coe[1])*feat_odd_bar
                    feat_prime = coe[2]*feat_prime + (1-coe[2])*feat_fuse
                else:
                    feat_fuse = self.alpha_*feat_even_bar + (1-self.alpha_)*feat_odd_bar
                    feat_prime = self.alpha_*feat_prime + (1-self.alpha_)*feat_fuse

        return feat_prime   # (B,T,N,C)

    # ---------- Forward ----------
    def forward(self, input, h0, adj, _format="BNTC"):
        """
        input: tensor in shape given by `_format` (any permutation of 'B','N','T','C')
        h0:    same shape as input
        adj:   adjacency (N,N) or (B,N,N)
        _format: string describing input/output format, e.g. 'BNTC', 'TBNC', etc.

        returns: tensor in the same format as input
        """
        fmt_to_idx = {ch: i for i, ch in enumerate(_format)}

        # Permute input to (B,T,N,C)
        perm_to_BTNC = [fmt_to_idx['B'], fmt_to_idx['T'],
                        fmt_to_idx['N'], fmt_to_idx['C']]
        x_BTNC = input.permute(*perm_to_BTNC).contiguous()
        h0_BTNC = h0.permute(*perm_to_BTNC).contiguous()

        # Apply attention + lifting
        U, P = self.attention(x_BTNC, adj)                   # (B,T,N,N)
        out_BTNC = self.forward_lifting_bases(x_BTNC, h0_BTNC, P, U, adj)  # (B,T,N,C)

        # Permute back to original format
        back_perm = ["BTNC".index(ch) for ch in _format]
        out = out_BTNC.permute(*back_perm).contiguous()
        return out
    
class TemporalAttentionAdj(nn.Module):
    """
    Temporal attention-based adjacency for T time steps.
    Computes A_time dynamically from input features.

    Input:
        X: (B, N, T, C)  features
    Output:
        A_time: (B, T, T) temporal adjacency per batch
    """
    def __init__(self, in_dim, hidden_dim=32, causal=False, self_loop=True, device='cuda'):
        super().__init__()
        self.causal = causal
        self.self_loop = self_loop
        self.device = device
        # Project features at each time step into query/key spaces
        self.W_q = nn.Linear(in_dim, hidden_dim)
        self.W_k = nn.Linear(in_dim, hidden_dim)

    def forward(self, X):
        B, N, T, C = X.shape

        # Pool across nodes (average) to get time-step embeddings
        X_t = X.mean(dim=1)               # (B, T, C)

        # Compute Q and K
        Q = self.W_q(X_t)                 # (B, T, H)
        K = self.W_k(X_t)                 # (B, T, H)

        # Compute similarity (scaled dot product)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # (B, T, T)
        # Apply causality mask if needed
        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=X.device), diagonal=0)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Row-wise softmax → adjacency
        A_time = F.softmax(attn_scores, dim=-1)  # (B, T, T)

        # Optionally remove self-loops
        if not self.self_loop:
            eye = torch.eye(T, device=X.device).unsqueeze(0)  # (1,T,T)
            A_time = A_time * (1 - eye)
        return A_time


class STSEAGWNN(torch.nn.Module):
    """docstring for TCNLayer"""
    def __init__(self, in_channels=2, out_channels=12, residual_channels=32, 
        dilation_channels=32, skip_channels=256, end_channels=512,
        kernel_size=2, blocks=4, layers=2, num_nodes=0, adj=None, device='cuda'):
        super(STSEAGWNN, self).__init__()
        self.device = device
        self.blocks = blocks
        self.layers = layers
        self.adj = torch.tensor(adj, device=device)
        self.start_conv = Conv2d(in_channels = 1,
                                    out_channels = residual_channels,
                                    kernel_size = (1,1))
    
        #w_in = 323
        #self.diag_weight = torch.nn.Parameter(torch.Tensor(w_in).to(device))
        #stdv = 1. / math.sqrt(w_in)
        #self.diag_weight.data.uniform_(-stdv,stdv)

        receptive_field = 1
        depth = list(range(blocks*layers))

        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1,1)) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.spatial_convs = ModuleList([AdaptiveWaveletLayer(num_nodes, dilation_channels, residual_channels, 4, None) for _ in depth])
        self.temporal_convs = ModuleList([AdaptiveWaveletLayer(out_channels, dilation_channels, residual_channels, 4, None) for _ in depth])
        self.temporal_graph_generator = TemporalAttentionAdj(dilation_channels, residual_channels, self_loop=False)
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()

        # for b in range(blocks):
        #     additional_scope = kernel_size - 1
        #     D = 1 #dilation
        #     for i in range(layers):
        #         #dilated convolutions
        #         self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
        #         self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
        #         D *= 2
        #         receptive_field += additional_scope
        #         additional_scope *= 2
        #     self.receptive_field = receptive_field

        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1,1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_channels, (1,1), bias=True)

    def forward(self, x):
        #Update: Input shape is (batch_size, n_timesteps, n_nodes, features)
        #Input shape is (batch_size, features, n_nodes, n_timesteps)
        # x = x.unsqueeze(3)
        # x = x.permute(0,3,2,1)
        # we want input shape (batch_size, n_timestamps, n_nodes, features)
        # print (f'-- Debug input shape: {x.shape} --')
        x = x.permute(0,3,2,1)
        x = self.start_conv(x)  #B,C,N,T
        tempAdj = self.temporal_graph_generator(x.permute(0,2,3,1))
        skip = 0
        #TCN Layers
        for i in range(self.blocks*self.layers):
            #Each block
            residual = x
            #parameterized skip connection
            s = self.skip_convs[i](x)
            try:
                skip = skip[:,:,:, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks*self.layers - 1): #last X getting ignored anyway
                break
            #x = x.permute(0,3,2,1).squeeze(-1)
            # print (f"----debug x device {x.device}----")
            x = self.spatial_convs[i](x, x, self.adj, _format='BCNT')
            x = self.temporal_convs[i](x, x, tempAdj, _format='BCTN')
            #x = self.residual_convs[i](x)
            #x = x.unsqueeze(-1)
            x = x + residual[:,:,:, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) #downsample to (bs, seq_lenth, n_nodes, n_features)
        # 
        output = x
        print (f"--debug output shape: {x.shape}--")
        return output




