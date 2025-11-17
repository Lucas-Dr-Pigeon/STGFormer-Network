import torch.nn as nn
import torch
import numpy as np
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
from typing import Optional, Literal, Tuple, List

def edges_to_sparse_coo(logits_edges, edge_index, N):
    B,L,E = logits_edges.shape
    device = logits_edges.device
    I,J = edge_index
    b = torch.arange(B, device=device).view(B,1,1).expand(B,L,E).reshape(-1)
    l = torch.arange(L, device=device).view(1,L,1).expand(B,L,E).reshape(-1)
    i = I.view(1,1,E).expand(B,L,E).reshape(-1)
    j = J.view(1,1,E).expand(B,L,E).reshape(-1)
    idx = torch.stack([b,l,i,j], dim=0)
    vals = logits_edges.reshape(-1)
    return torch.sparse_coo_tensor(idx, vals, size=(B,L,N,N))

class DirectedLowRankEdgeScorer(nn.Module):
    """
    (B,T,N,C) -> (B, L, E) logits on the provided edges.
      - Directed: separate source/destination node scorers
      - Low-rank: rank r
      - Per-lag mixing: either diagonal (fast) or full bilinear (r x r)

    Args:
      in_dim:    node feature dim C (after your temporal projection)
      hidden:    hidden size in node MLPs
      rank_r:    low rank (>=1)
      L:         number of lags to predict
      pooling:   'last' or 'mean' over time dimension T
      mix:       'diag' (default, efficient) or 'full' (r x r per lag)
      nonneg:    if True, clamp embeddings >= 0 (optional)
    """
    def __init__(self, in_dim: int, hidden: int, rank_r: int, L: int,
                 pooling: str = "last", mix: str = "diag", nonneg: bool = False):
        super().__init__()
        assert pooling in ("last", "mean")
        assert mix in ("diag", "full")
        self.pooling, self.mix, self.L, self.r = pooling, mix, L, rank_r
        self.nonneg = nonneg

        # Directed node scorers → a (source), b (dest)
        self.src_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, rank_r)  # (B,N,r)
        )
        self.dst_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, rank_r)  # (B,N,r)
        )

        if mix == "diag":
            # per-lag diagonal mixing γ_{ℓ,k}
            self.gamma = nn.Parameter(torch.zeros(L, rank_r)) 
        else:
            # per-lag full bilinear W_{ℓ} ∈ R^{r×r}
            self.W = nn.Parameter(torch.zeros(L, rank_r, rank_r))
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming on last layers; small bias to break symmetry
        for lin in (self.src_mlp[-1], self.dst_mlp[-1]):
            nn.init.kaiming_uniform_(lin.weight, a=0.1)
            nn.init.uniform_(lin.bias, -0.01, 0.01)
        if self.mix == "diag":
            nn.init.uniform_(self.gamma, 0.01, 0.05)  # diag case
        else:
            nn.init.uniform_(self.W, -0.05, 0.05)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        X:          (B, T, N, C)
        edge_index: (2, E) long (same device as X)
        Returns:
          logits_edges: (B, L, E)  — BCEWithLogits-ready
        """
        device = X.device
        B, T, N, C = X.shape
        I, J = edge_index.to(device)  # (E,)

        # Pool in time
        H = X[:, -1] if self.pooling == "last" else X.mean(dim=1)  # (B,N,C)

        # Node embeddings
        a = self.src_mlp(H)   # (B,N,r)
        b = self.dst_mlp(H)   # (B,N,r)

        if self.nonneg:
            a, b = F.relu(a), F.relu(b)

        # Gather only needed edges
        # a_i: (B,E,r), b_j: (B,E,r)
        a_i = a.index_select(dim=1, index=I)  # source
        b_j = b.index_select(dim=1, index=J)  # dest

        if self.mix == "diag":
            # (B,E,r) * (L,r) -> (B,L,E) via einsum: (be r)·(l r) over r
            logits = torch.einsum("ber,lr,ber->ble", a_i, self.gamma, b_j)
        else:
            # full bilinear: (a_i W_l)·b_j for each lag l
            # a_i: (B,E,r), W: (L,r,r) → tmp: (B,E,L,r)
            tmp = torch.einsum("ber,lrs->bels", a_i, self.W)
            logits = torch.einsum("bels,ber->ble", tmp, b_j)

        return logits  # (B, L, E) logits

class FastAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, qkv_bias=False, kernel=1):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)

        self.out_proj = nn.Linear(
            2 * model_dim if kernel != 12 else model_dim, model_dim
        )
        # self.proj_in = nn.Conv2d(model_dim, model_dim, (1, kernel), 1, 0) if kernel > 1 else nn.Identity()
        self.fast = 1

    def forward(self, x, edge_index=None, dim=0):
        # x = self.proj_in(x.transpose(1, 3)).transpose(1, 3)
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        if self.fast:
            out_s = self.fast_attention(x, qs, ks, vs, dim=dim)
        else:
            out_s = self.normal_attention(x, qs, ks, vs, dim=dim)
        if x.size(1) > 1:
            qs = torch.stack(
                torch.split(query.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            ks = torch.stack(
                torch.split(key.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            vs = torch.stack(
                torch.split(value.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            if self.fast:
                out_t = self.fast_attention(
                    x.transpose(1, 2), qs, ks, vs, dim=dim
                ).transpose(1, 2)
            else:
                out_t = self.normal_attention(
                    x.transpose(1, 2), qs, ks, vs, dim=dim
                ).transpose(1, 2)
            out = torch.concat([out_s, out_t], -1)
            out = self.out_proj(out)
        else:
            out = self.out_proj(out_s)

        return out

    def fast_attention(self, x, qs, ks, vs, dim=0):
        qs = nn.functional.normalize(qs, dim=-1)
        ks = nn.functional.normalize(ks, dim=-1)
        N = qs.shape[1]
        b, l = x.shape[dim : dim + 2]

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[1]]).to(ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)
        )  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        out = attention_num / attention_normalizer.clamp_min(1e-5)  # [N, H, D]
        out = torch.unflatten(out, dim, (b, l)).flatten(start_dim=3)
        return out

    def normal_attention(self, x, qs, ks, vs, dim=0):
        b, l = x.shape[dim : dim + 2]
        qs, ks, vs = qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2)
        x = (
            torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        x = torch.unflatten(x, dim, (b, l)).flatten(start_dim=3)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, qkv_bias=False, kernel=1):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x, edge_index=None):
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-3)
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-3)
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-3)
        x = (
            torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        x = self.out_proj(x)
        return x


class GraphPropagate(nn.Module):
    def __init__(self, Ks, gso, dropout = 0.2):
        super(GraphPropagate, self).__init__()
        self.Ks = Ks
        self.gso = gso
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, graph):
        # print (f"--debug graph shape: {graph.shape} x shape: {x.shape} --")
        if self.Ks < 1:
            raise ValueError(
                f"ERROR: Ks must be a positive integer, received {self.Ks}."
            )
        x_k = x; x_list = [x]
        # print (graph.shape)
        for k in range(1, self.Ks):
            x_k = torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))

        return x_list
    
class MultiScaleGraphPropagate(nn.Module):
    """
    Sparse message passing over K-1 hops using L lagged graphs encoded by edge weights.

    Inputs:
      x:        (B, T, N, F)     node features over time
      edge_w:   (B, L, E)        per-batch, per-lag edge weights for a shared edge_index
      weights:  (B, L) | (L,)    optional per-lag combination weights
               - If provided and combine='weighted', softmax over L is applied before combining.

    Edge structure:
      edge_index: (2, E) LongTensor  [target i, source j], shared across batch and lags

    Returns:
      x_list:    list [x^(0), x^(1), ..., x^(K-1)], each (B, T, N, F)

    Notes:
      - combine: 'mean'|'sum'|'weighted' controls how to reduce L lags into a single (B,E) weight.
      - normalize/self_loops are optional (off by default to mirror your dense reference).
    """
    def __init__(self,
                 Ks: int,
                 edge_index: torch.Tensor,        # (2, E) long
                 dropout: float = 0.2,
                 combine: str = "mean",
                 normalize: bool = False,
                 self_loops: float = 0.0):
        super().__init__()
        if Ks < 1:
            raise ValueError(f"ERROR: Ks must be >= 1, received {Ks}.")
        if combine not in ("mean", "sum", "weighted"):
            raise ValueError("combine must be one of {'mean','sum','weighted'}")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be (2,E); got {tuple(edge_index.shape)}")

        self.Ks = Ks
        self.combine = combine
        self.dropout = nn.Dropout(dropout)
        self.normalize = bool(normalize)
        self.self_loops = float(self_loops)

        # Keep indices as buffers so .to(device) works on the module
        self.register_buffer("tgt", edge_index[0].long())
        self.register_buffer("src", edge_index[1].long())

    # ---------- Lag combination: (B,L,E) -> (B,E) ----------
    def _combine_lag_weights(self,
                             edge_w_BLE: torch.Tensor,
                             weights: Optional[torch.Tensor]) -> torch.Tensor:
        """
        edge_w_BLE: (B, L, E)
        weights:    None | (L,) | (B, L)
        returns:    (B, E)   (combined over L according to self.combine)
        """
        B, L, E = edge_w_BLE.shape

        if self.combine == "weighted":
            if weights is None:
                raise ValueError("combine='weighted' requires 'weights' in forward().")
            if weights.ndim == 1:
                if weights.numel() != L:
                    raise ValueError(f"weights (L,) expected L={L}, got {tuple(weights.shape)}")
                w = torch.softmax(weights, dim=0).view(1, L, 1)         # (1,L,1)
                return (edge_w_BLE * w).sum(dim=1)                      # (B,E)
            elif weights.ndim == 2:
                if weights.shape != (B, L):
                    raise ValueError(f"weights must be (B,L)={B,L}, got {tuple(weights.shape)}")
                w = torch.softmax(weights, dim=1).unsqueeze(-1)         # (B,L,1)
                return (edge_w_BLE * w).sum(dim=1)                      # (B,E)
            else:
                raise ValueError("weights must be (L,) or (B,L)")

        if self.combine == "sum":
            return edge_w_BLE.sum(dim=1)                                # (B,E)

        # default: mean
        return edge_w_BLE.mean(dim=1)                                   # (B,E)

    # ---------- Optional row-normalization over targets ----------
    @staticmethod
    def _rownorm_BE(w_BE: torch.Tensor, tgt: torch.Tensor, N: int):
        """
        Row-normalize per-batch edge weights by target-degree.
        Returns (w_norm_BE, deg_BN).
        """
        B, E = w_BE.shape
        deg = torch.zeros(B, N, device=w_BE.device, dtype=w_BE.dtype)   # (B,N)
        deg.index_add_(1, tgt, w_BE)                                    # sum into target rows
        scale = deg[:, tgt].clamp_min_(1e-9)                             # (B,E)
        return w_BE / scale, deg                                        # (B,E), (B,N)

    # ---------- One-hop sparse propagation with (B,E) weights ----------
    @staticmethod
    def _prop_once_BE(x_BTNC: torch.Tensor,
                      tgt: torch.Tensor, src: torch.Tensor,
                      w_BE: torch.Tensor, N: int,
                      self_loops: float, deg_BN: Optional[torch.Tensor],
                      normalize: bool):
        """
        x: (B,T,N,F) -> y: (B,T,N,F)
        """
        B, T, Nchk, F = x_BTNC.shape
        assert Nchk == N
        device, dtype = x_BTNC.device, x_BTNC.dtype

        # Gather neighbor features collapsed over (T*F)
        X_BTCN = x_BTNC.permute(0, 2, 3, 1).reshape(B, T * F, N)        # (B,TF,N)
        msgs = X_BTCN[..., src] * w_BE.unsqueeze(1)                     # (B,TF,E)

        # Scatter to targets in one fused op over rows=B*TF
        Yflat = torch.zeros(B * T * F, N, device=device, dtype=dtype)
        Yflat.index_add_(1, tgt, msgs.reshape(B * T * F, -1))           # (B*TF,N)
        Y = Yflat.view(B, T, F, N).permute(0, 1, 3, 2).contiguous()      # (B,T,N,F)

        # Optional self-loop
        if self_loops != 0.0:
            if normalize:
                denom = (deg_BN + self_loops).unsqueeze(1).unsqueeze(-1)  # (B,1,N,1)
                Y = Y + (self_loops * x_BTNC) / denom
            else:
                Y = Y + self_loops * x_BTNC
        return Y

    # ---------- Forward ----------
    def forward(self,
                x: torch.Tensor,            # (B, T, N, F)
                edge_w_BLE: torch.Tensor,   # (B, L, E)
                weights: Optional[torch.Tensor] = None):
        if x.ndim != 4:
            raise ValueError(f"x must be (B,T,N,F); got {tuple(x.shape)}")
        if edge_w_BLE.ndim != 3:
            raise ValueError(f"edge_w must be (B,L,E); got {tuple(edge_w_BLE.shape)}")

        B, T, N, F = x.shape
        _, L, E = edge_w_BLE.shape
        if self.tgt.numel() != E:
            raise ValueError(f"edge_index E={self.tgt.numel()} does not match edge_w E={E}")

        # 1) Combine lags -> effective per-batch edge weights (B,E)
        w_BE = self._combine_lag_weights(edge_w_BLE, weights)           # (B,E)

        # 2) Optional row-normalization once per forward
        if self.normalize:
            w_BE, deg_BN = self._rownorm_BE(w_BE, self.tgt, N)
        else:
            deg_BN = None

        # 3) K-1 hops of sparse propagation
        x_k = x
        x_list = [x_k]
        for _ in range(1, self.Ks):
            x_k = self._prop_once_BE(
                x_k, self.tgt, self.src, w_BE, N, self.self_loops, deg_BN, self.normalize
            )
            x_list.append(self.dropout(x_k))

        return x_list


    
# class MultiScaleGraphPropagate(nn.Module):
#     """
#     Message passing over K-1 hops using a (possibly multi-lag) directed graph.

#     Inputs:
#       x:      (B, T, N, F)   node features over time
#       graph:  (B, N, N)  or  (B, L, N, N)
#               if 4D, lags are combined before use (mean by default)
#       weights (optional): per-lag weights to combine (B, L) or (L,)
#               If provided, a softmax over L is applied and a weighted sum of A_l is used.

#     Returns:
#       x_list: list of tensors [x^(0), x^(1), ..., x^(K-1)]
#               each x^(k) has shape (B, T, N, F)
#     """
#     def __init__(self, Ks: int, dropout: float = 0.2, combine: str = "mean"):
#         super().__init__()
#         if Ks < 1:
#             raise ValueError(f"ERROR: Ks must be >= 1, received {Ks}.")
#         if combine not in ("mean", "sum", "weighted"):
#             raise ValueError("combine must be one of {'mean','sum','weighted'}")
#         self.Ks = Ks
#         self.combine = combine
#         self.dropout = nn.Dropout(dropout)

#     def _combine_lags(self, graph: torch.Tensor, weights: torch.Tensor | None):
#         """
#         graph:   (B,N,N) or (B,L,N,N)
#         weights: (B,L) or (L,) or None
#         returns: (B,N,N)
#         """
#         if graph.ndim == 3:
#             # already (B,N,N)
#             return graph

#         # (B,L,N,N)
#         B, L, N, _ = graph.shape

#         if self.combine == "weighted":
#             if weights is None:
#                 raise ValueError("combine='weighted' requires 'weights' in forward().")
#             # normalize weights over L
#             if weights.ndim == 1:
#                 w = torch.softmax(weights, dim=0).view(1, L, 1, 1)              # (1,L,1,1)
#                 A = (graph * w).sum(dim=1)                                      # (B,N,N)
#             elif weights.ndim == 2:
#                 if weights.shape != (B, L):
#                     raise ValueError(f"weights must be (B,L) for batch-specific weights; got {tuple(weights.shape)}")
#                 w = torch.softmax(weights, dim=1).view(B, L, 1, 1)              # (B,L,1,1)
#                 A = (graph * w).sum(dim=1)                                      # (B,N,N)
#             else:
#                 raise ValueError("weights must be (L,) or (B,L)")
#             return A

#         if self.combine == "sum":
#             return graph.sum(dim=1)                                             # (B,N,N)

#         # default: mean
#         return graph.mean(dim=1)                                                # (B,N,N)

#     def forward(self, x: torch.Tensor, graph: torch.Tensor, weights: torch.Tensor | None = None):
#         """
#         x:     (B, T, N, F)
#         graph: (B, N, N) or (B, L, N, N)
#         weights (optional): see _combine_lags
#         """
#         if self.Ks < 1:
#             raise ValueError(f"ERROR: Ks must be >= 1, received {self.Ks}.")

#         if x.ndim != 4:
#             raise ValueError(f"x must be (B,T,N,F); got {tuple(x.shape)}")
#         if graph.ndim not in (3, 4):
#             raise ValueError(f"graph must be (B,N,N) or (B,L,N,N); got {tuple(graph.shape)}")

#         B, T, N, F = x.shape
#         # Combine lags to a single adjacency per batch
#         A = self._combine_lags(graph, weights)                                  # (B,N,N)

#         # Hop-0 is the input itself
#         x_k = x
#         x_list = [x]

#         # Repeated propagation for hops 1..Ks-1:
#         # y[b,t,i,f] = sum_j A[b,i,j] * x_k[b,t,j,f]
#         # einsum: (B,N,N) x (B,T,N,F) -> (B,T,N,F)
#         for _ in range(1, self.Ks):
#             x_k = torch.einsum("bij,btjf->btif", A, x_k)
#             x_list.append(self.dropout(x_k))

#         return x_list


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        mlp_ratio=2,
        num_heads=8,
        dropout=0,
        mask=False,
        kernel=3,
        supports=None,
        order=2,
        edge_index=None,
    ):
        super().__init__()
        self.locals = GraphPropagate(Ks=order, gso=supports[0]) if edge_index is None \
            else MultiScaleGraphPropagate(Ks=order, edge_index=edge_index)
        self.attn = nn.ModuleList(
            [
                FastAttentionLayer(model_dim, num_heads, mask, kernel=kernel)
                for _ in range(order)
            ]
        )
        self.pws = nn.ModuleList(
            [nn.Linear(model_dim, model_dim) for _ in range(order)]
        )
        for i in range(0, order):
            nn.init.constant_(self.pws[i].weight, 0)
            nn.init.constant_(self.pws[i].bias, 0)

        self.kernel = kernel
        self.fc = Mlp(
            in_features=model_dim,
            hidden_features=int(model_dim * mlp_ratio),
            act_layer=nn.ReLU,
            drop=dropout,
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = [1, 0.01, 0.001]

    def forward(self, x, graph):
        x_loc = self.locals(x, graph)
        c = x_glo = x
        for i, z in enumerate(x_loc):
            att_outputs = self.attn[i](z)
            x_glo += att_outputs * self.pws[i](c) * self.scale[i]
            c = att_outputs
        x = self.ln1(x + self.dropout(x_glo))
        x = self.ln2(x + self.dropout(self.fc(x)))
        return x

class PropSTGformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=12,
        num_heads=4,
        num_lags=3,
        supports=None,
        num_layers=3,
        dropout=0.1,
        mlp_ratio=2,
        use_mixed_proj=True,
        dropout_a=0.3,
        kernel_size=[1],
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        # ---- channel accounting ----
        self.base_model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
        )  # strictly NO adaptive channels
        self.model_dim = self.base_model_dim + adaptive_embedding_dim  # full feature dim

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        self.dropout = nn.Dropout(dropout_a)
        self.pooling = nn.AvgPool2d(kernel_size=(1, kernel_size[0]), stride=1)

        # define graphs
        self.road_graph, self.plausible_connectivity = supports[0], supports[1]

        # fusion weights
        self.mix_logits = nn.Parameter(torch.zeros(3))

        # edge index for endpoint proximity graph
        I, J = self.plausible_connectivity.nonzero(as_tuple=True)
        self.edge_index = torch.stack([I, J], dim=0)

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim, mlp_ratio, num_heads, dropout,
                    kernel=size, supports=supports, 
                )
                for size in kernel_size
            ]
        )
        self.attn_layers_d1 = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim, mlp_ratio, num_heads, dropout,
                    kernel=size, supports=supports, edge_index=self.edge_index,
                )
                for size in kernel_size
            ]
        )
        self.attn_layers_d2 = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim, mlp_ratio, num_heads, dropout,
                    kernel=size, supports=supports, edge_index=self.edge_index,
                )
                for size in kernel_size
            ]
        )

        # ------------ projections ------------
        # full stream uses (base + adaptive) channels
        self.temporal_proj_full = nn.Conv2d(self.model_dim, self.model_dim, (1, kernel_size[0]), 1, 0)
        # dynamic-graph stream uses ONLY base channels (no adaptive)
        self.temporal_proj_dyn  = nn.Conv2d(self.base_model_dim, self.base_model_dim, (1, kernel_size[0]), 1, 0)

        self.encoder_proj = nn.Linear(
            (in_steps - sum(k - 1 for k in kernel_size)) * self.model_dim,
            self.model_dim,
        )
        self.kernel_size = kernel_size[0]

        self.encoder = nn.ModuleList(
            [
                Mlp(
                    in_features=self.model_dim,
                    hidden_features=int(self.model_dim * mlp_ratio),
                    act_layer=nn.ReLU,
                    drop=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.model_dim, out_steps * output_dim)

        # --------- dynamic graph scorers (STRICTLY base dim) ---------
        in_dim_dyn = self.base_model_dim  # ensure θ, φ see no adaptive channels
        self.pos_prop_graph = DirectedLowRankEdgeScorer(
            in_dim_dyn, hidden=64, rank_r=1, L=num_lags, pooling="last", mix="diag"
        )
        self.neg_prop_graph = DirectedLowRankEdgeScorer(
            in_dim_dyn, hidden=64, rank_r=1, L=num_lags, pooling="last", mix="diag"
        )
        self.num_lags = num_lags

        

    def forward(self, x):
        # x: (B, in_steps, N, input_dim+tod+dow=3)
        B = x.shape[0]

        # ----- split metadata -----
        tod = x[..., 1] if self.tod_embedding_dim > 0 else None
        dow = x[..., 2] if self.dow_embedding_dim > 0 else None
        x_sig = x[..., : self.input_dim]  # (B, T, N, input_dim)

        # ----- base embeddings (NO adaptive) -----
        x_base = self.input_proj(x_sig)  # (B, T, N, input_embedding_dim)

        feats = []
        if self.tod_embedding_dim > 0:
            # expect tod \in [0,1) scaled by steps_per_day upstream (your original had this commented)
            feats.append(self.tod_embedding((tod * self.steps_per_day).long()))
        if self.dow_embedding_dim > 0:
            feats.append(self.dow_embedding(dow.long()))
        if self.spatial_embedding_dim > 0:
            # if you have spatial embeddings, append here
            raise NotImplementedError("Provide spatial embeddings before enabling.")

        base_cat = x_base if len(feats) == 0 else torch.cat([x_base] + feats, dim=-1)
        assert base_cat.shape[-1] == self.base_model_dim, \
            f"base_model_dim mismatch: {base_cat.shape[-1]} vs {self.base_model_dim}"

        # ----- full features = base + adaptive (used by backbone) -----
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(B, *self.adaptive_embedding.shape)  # (B,T,N,D_adp)
            full_x = torch.cat([base_cat, self.dropout(adp_emb)], dim=-1)
        else:
            full_x = base_cat

        # ----- temporal projections -----
        # full stream for backbone/attention blocks
        x_full = self.temporal_proj_full(full_x.transpose(1, 3)).transpose(1, 3)  # (B,T,N,C_full)

        # dyn stream for graph construction (strictly base channels)
        x_dyn = self.temporal_proj_dyn(base_cat.transpose(1, 3)).transpose(1, 3)   # (B,T,N,C_base)

        # ================== Static graph from adaptive only ==================
        if self.adaptive_embedding_dim > 0:
            G_adp = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))   # (T,N,N)
            G_adp = self.pooling(G_adp.transpose(0, 2)).transpose(0, 2)                              # pool over T-window
            G_adp = F.softmax(F.relu(G_adp), dim=-1)                                                 # (N,N) row-stochastic
        else:
            # fallback: identity if no adaptive channels
            G_adp = torch.eye(self.num_nodes, device=x.device)

        x_0 = x_full.clone()
        for attn in self.attn_layers_s:
            x_0 = attn(x_0, G_adp)

        # ================== Sample-wise dynamic graphs (θ(x)φ(x)^T) ==================
        # IMPORTANT: these scorers were built with in_dim = base_model_dim,
        # and we feed x_dyn (no adaptive channels).
        logits_p = self.pos_prop_graph(x_dyn, self.edge_index)  # (B,L,E)
        logits_n = self.neg_prop_graph(x_dyn, self.edge_index)  # (B,L,E)

        # A_pos = edges_to_sparse_coo(logits_p, self.edge_index, self.num_nodes)  # (B, N, N) sparse
        # A_neg = edges_to_sparse_coo(logits_n, self.edge_index, self.num_nodes)  # (B, N, N) sparse

        x_1, x_2 = x_full.clone(), x_full.clone()
        for attn in self.attn_layers_d1:
            x_1 = attn(x_1, logits_p)   # keep API consistent with your attn
        for attn in self.attn_layers_d2:
            x_2 = attn(x_2, logits_n)

        # =============== fusion ===============
        w = torch.softmax(self.mix_logits, dim=0)  # (3,)
        x = w[0] * x_0 + w[1] * x_1 + w[2] * x_2

        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x)

        out = self.output_proj(x).view(B, self.num_nodes, self.out_steps, self.output_dim)
        out = out.transpose(1, 2)  # (B, out_steps, N, output_dim)

        return out, logits_p, logits_n


