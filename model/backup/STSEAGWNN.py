import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, BatchNorm2d, ModuleList, Parameter
import torch.nn.functional as F
import math
from tqdm import trange
import time
from torch.autograd import Variable
import numpy as np

def to_edge_index(adj_dense: torch.Tensor, device=None):
    """
    adj_dense: (N,N) dense 0/weight matrix
    returns:
        edge_index: (2,E) long  (src, dst)
        edge_weight: (E,) float
    """
    if device is None:
        device = adj_dense.device
    coo = adj_dense.to_sparse().coalesce()
    edge_index = coo.indices().to(device)   # (2,E)
    edge_weight = coo.values().to(device)   # (E,)
    return edge_index, edge_weight

def temporal_mix_safe(x_bcnt, A_time, blend=0.5, clip_val=5.0):
    """
    x_bcnt: (B,C,N,T), A_time: (B,T,T) row-stochastic.
    Returns: (B,C,N,T) with residual smoothing + numeric sanitization.
    """
    x_new = torch.einsum("bij,bcnj->bcni", A_time, x_bcnt)  # temporal mixing
    x_out = blend * x_bcnt + (1.0 - blend) * x_new          # residual smoothing
    x_out = torch.nan_to_num(x_out, nan=0.0, posinf=clip_val, neginf=-clip_val)
    return x_out.clamp_(-clip_val, clip_val)

def safe_row_softmax_strict(
    logits, mask=None, tau=1.5, eps=1e-12,
    row_floor=0.05, row_ceiling=0.35
):
    """
    Numerically safe softmax:
      - subtracts row max for stability
      - masks with -1e9 instead of -inf
      - temperature (tau>1 smooths)
      - adds uniform floor to prevent delta-rows
      - optional per-entry ceiling + renorm
    """
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -1e9)

    # stabilize
    logits = logits - logits.max(dim=-1, keepdim=True)[0]

    # temperature softmax
    A = torch.softmax(logits / tau, dim=-1)
    A = A.clamp_min(eps)

    # add floor
    T = A.size(-1)
    A = (1.0 - row_floor) * A + row_floor * (1.0 / T)

    # ceiling (optional)
    if row_ceiling is not None:
        A = torch.minimum(A, A.new_full((), row_ceiling))
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(eps)

    # final renorm
    A = A / A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A


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
        B, T, N, C = x_BTNC.shape
        a1, a2 = self.a[:self.in_features, :], self.a[self.in_features:, :]

        feat_1 = torch.matmul(x_BTNC, a1)              # (B,T,N,1)
        feat_2 = torch.matmul(x_BTNC, a2)              # (B,T,N,1)
        e = self.leakyrelu(feat_1 + feat_2.transpose(-2, -1))   # (B,T,N,N)

        # Treat adj as a soft prior if it's float; fall back to hard mask if it's bool/int
        if adj.dtype.is_floating_point:
            # adj can be (N,N) or (B,N,N)
            if adj.dim() == 2:
                log_prior = (adj.clamp_min(1e-12)).log().view(1,1,N,N)     # (1,1,N,N)
            elif adj.dim() == 3:
                log_prior = (adj.clamp_min(1e-12)).log().unsqueeze(1)      # (B,1,N,N)
            else:
                raise ValueError("adj must be (N,N) or (B,N,N)")
            logits = e + log_prior                                         # differentiable w.r.t. adj
            U = torch.softmax(logits, dim=-1)                              # (B,T,N,N)
            P = 0.5 * U
        else:
            # fallback: non-diff mask
            if adj.dim() == 2:
                mask = (adj > 0).view(1,1,N,N).expand(B,T,-1,-1)
            elif adj.dim() == 3:
                mask = (adj > 0).unsqueeze(1).expand(-1,T,-1,-1)
            neg_inf = torch.finfo(e.dtype).min
            e_masked = torch.where(mask, e, e.new_full((), neg_inf))
            U = torch.softmax(e_masked, dim=-1)
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
    
class SparseNodeAttentionMP(nn.Module):
    def __init__(self, in_features, negative_slope=0.2):
        super().__init__()
        self.a = nn.Parameter(torch.empty(2*in_features, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.neg_slope = negative_slope

    def forward(self, x_BTNC, edge_index):
        B,T,N,C = x_BTNC.shape
        device  = x_BTNC.device
        dst, src = edge_index[0].to(device), edge_index[1].to(device)
        E = src.numel()

        a1, a2 = self.a[:C,:], self.a[C:,:]
        feat1 = torch.matmul(x_BTNC, a1).squeeze(-1)  # (B,T,N)
        feat2 = torch.matmul(x_BTNC, a2).squeeze(-1)  # (B,T,N)
        e_ij  = F.leaky_relu(feat1[..., dst] + feat2[..., src], negative_slope=self.neg_slope)

        exp_e = torch.exp(e_ij)                       # (B,T,E)

        denom = torch.zeros(B, T, N, device=device)
        idx_bte = dst.view(1,1,E).repeat(B,T,1)       # (B,T,E) -> MATCH
        denom.scatter_add_(2, idx_bte, exp_e)
        denom = denom.clamp_min_(1e-8)

        denom_edges = denom.gather(2, idx_bte)        # (B,T,E) -> MATCH
        alpha_ij = exp_e / denom_edges

        x_src = x_BTNC[..., src, :]                   # (B,T,E,C)
        msgs  = alpha_ij.unsqueeze(-1) * x_src        # (B,T,E,C)

        out = torch.zeros(B, T, N, C, device=device)
        idx4d = dst.view(1,1,E,1).repeat(B,T,1,C)     # (B,T,E,C)
        out.scatter_add_(2, idx4d, msgs)

        return out

    
class AdaptiveWaveletLayerSparse(nn.Module):
    def __init__(self, nnode, in_features, out_features, hop, alpha, edge_index):
        super().__init__()
        self.hop = hop
        self.alpha_ = alpha
        self.out_proj = nn.Linear(in_features, out_features)

        self.temp = nn.Parameter(torch.zeros(hop+1))
        self.cheb = nn.Parameter(torch.ones(hop+1)*0.5)

        self.sparse_attn_mp = SparseNodeAttentionMP(in_features)
        self.edge_index = edge_index

    def forward_lifting_bases(self, x_BTNC, h0_BTNC):
        update = self.sparse_attn_mp(x_BTNC, self.edge_index)  # (B,T,N,C)
        coe = torch.sigmoid(self.temp)
        cheb_coe = torch.sigmoid(self.cheb)
        feat_prime = None

        for step in range(self.hop):
            if step > 0:
                update = self.sparse_attn_mp(update, self.edge_index)

            feat_even_bar = (coe[0]*x_BTNC + update) if self.alpha_ is None else update
            if step >= 1:
                update = cheb_coe[step-1] * update
            feat_odd_bar = update - feat_even_bar

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
        return self.out_proj(feat_prime)

    def forward(self, x, h0, _format="BNTC"):
        idx = {ch:i for i,ch in enumerate(_format)}
        x_BTNC  = x.permute(idx['B'], idx['T'], idx['N'], idx['C']).contiguous()
        h0_BTNC = h0.permute(idx['B'], idx['T'], idx['N'], idx['C']).contiguous()
        out_BTNC = self.forward_lifting_bases(x_BTNC, h0_BTNC)
        back = ["BTNC".index(ch) for ch in _format]
        return out_BTNC.permute(*back).contiguous()


class TemporalAttentionAdj(nn.Module):
    def __init__(self, in_dim, hidden_dim, self_loop=False):
        super().__init__()
        self.self_loop = self_loop

        # Query & Key projections
        self.W_q = nn.Linear(in_dim, hidden_dim)
        self.W_k = nn.Linear(in_dim, hidden_dim)

        self.tau = 1.0

        # Init with asymmetry to avoid uniform softmax
        nn.init.xavier_uniform_(self.W_q.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_k.weight, gain=1.414)
        nn.init.uniform_(self.W_q.bias, -0.1, 0.1)
        nn.init.uniform_(self.W_k.bias, -0.1, 0.1)

    def forward(self, x, band=2):
        """
        x: (B, N, T, C)
        returns:
            A_time: (B, T, T) temporal adjacency
            logits: (B, T, T) raw scores
        """
        B, N, T, C = x.shape
        device = x.device

        # --- temporal pooling ---
        x_mean = x.mean(dim=1)  # (B, T, C)

        # --- Q, K projections ---
        Q = self.W_q(x_mean)    # (B, T, d)
        K = self.W_k(x_mean)    # (B, T, d)

        # normalize for stability
        Q = torch.nn.functional.layer_norm(Q, Q.shape[-1:])
        K = torch.nn.functional.layer_norm(K, K.shape[-1:])
        Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
        K = torch.nn.functional.normalize(K, p=2, dim=-1)

        # bounded logits
        logits = torch.matmul(Q, K.transpose(-2, -1))  # (B, T, T)
        logits = logits.clamp_(-6.0, 6.0)

        # --- finite mask ---
        idx = torch.arange(T, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        mask = (dist <= band).float().unsqueeze(0)  # (1, T, T)
        if not self.self_loop:
            mask[:, idx, idx] = 0.0

        # --- safe softmax ---
        tau = getattr(self, "tau", 1.5)
        row_floor = getattr(self, "row_floor", 0.05)
        row_ceiling = getattr(self, "row_ceiling", 0.35)

        A_time = safe_row_softmax_strict(
            logits, mask=mask, tau=tau,
            row_floor=row_floor, row_ceiling=row_ceiling
        )

        return A_time, logits



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
        
        self.edge_index, self.edge_weight = to_edge_index(
            torch.tensor(adj, dtype=torch.float32), device=device
        )
    
        depth = list(range(blocks*layers))

        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1,1)) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        # self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.norms = ModuleList([nn.GroupNorm(residual_channels, residual_channels) for _ in depth])
        self.spatial_convs = ModuleList([AdaptiveWaveletLayerSparse(num_nodes, dilation_channels, residual_channels, 4, None, edge_index=self.edge_index
                ) for _ in depth])
        self.temporal_convs = ModuleList([AdaptiveWaveletLayer(out_channels, dilation_channels, residual_channels, 4, None) for _ in depth])
        self.temporal_graph_generator = TemporalAttentionAdj(dilation_channels, residual_channels, self_loop=False)
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()

        self.end_fc1 = nn.Linear(residual_channels, residual_channels*4)
        self.end_fc2 = nn.Linear(residual_channels*4, 1)

        
    def forward(self, x):
        #Update: Input shape is (batch_size, n_timesteps, n_nodes, features)
        #Input shape is (batch_size, features, n_nodes, n_timesteps)
        # we want input shape (batch_size, n_timestamps, n_nodes, features)
        # print (f'-- Debug input shape: {x.shape} --')
        x = x.permute(0,3,2,1)
        # initialize reg term
        reg = {}
        # print (f"--debug x shape before start_conv: {x.shape}--")
        x = self.start_conv(x)  #B,C,N,T
        tempAdj, logits = self.temporal_graph_generator(x.permute(0,2,3,1))
        #TCN Layers
        # print (f"--debug x shape: {x.shape}--")
        for i in range(self.blocks*self.layers):
            #x = x.permute(0,3,2,1).squeeze(-1)
            # print (f"----debug x device {x.device}----")
            residual = x
            x = F.relu(temporal_mix_safe(x, tempAdj, blend=0.5, clip_val=5.0))
            x = self.spatial_convs[i](x, x, _format='BCNT')
            # x = self.temporal_convs[i](x, x, tempAdj, _format='BCTN')
            # x = torch.nan_to_num(x).clamp_(-5.0, 5.0)
            x = self.norms[i](x)
            # x = torch.nan_to_num(x).clamp_(-5.0, 5.0)
            x = F.relu(x + residual) 
        x = x.permute(0,3,2,1)   # B,C,N,T --> B,T,N,C
        # print (f"--debug x shape after S-T layers: {x.shape}--")
        x = F.relu(self.end_fc1(x))
        # print (f"--debug output shape after end_1: {x.shape}--")
        x = self.end_fc2(x) #downsample to (bs, seq_lenth, n_nodes, n_features)
        # 
        # print (f"--debug output shape after end_2: {x.shape}--")
        output = x

        reg['time_adj'] = tempAdj

        # with torch.no_grad():
        #     print("A_time finite:", torch.isfinite(tempAdj).all().item(),
        #         "min/max:", float(tempAdj.min()), float(tempAdj.max()))
        #     print("logits min/max:", float(logits.min()), float(logits.max()))
        #     print("x min/max:", float(x.min()), float(x.max()))
        
        return output[:,0:1,:,:], reg




