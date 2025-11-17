import torch.nn as nn
import torch
import numpy as np
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F

def dense_to_edges(G: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Autograd-safe gather: (B,L,N,N) -> (B,L,E).
    Gradients flow back to the selected entries of G.
    """
    assert G.ndim == 4, f"expected (B,L,N,N), got {tuple(G.shape)}"
    B, L, N, _ = G.shape
    I, J = edge_index
    assert I.device == G.device and J.device == G.device
    assert I.dtype == torch.long and J.dtype == torch.long

    # 1) select destination columns (dim=3) -> (B, L, N, E)
    cols = G.index_select(dim=3, index=J)

    # 2) gather source rows (dim=2) using a 4D index of shape (B,L,1,E)
    I_4d = I.view(1, 1, 1, -1).expand(B, L, 1, I.numel())  # (B,L,1,E)
    out = torch.gather(cols, dim=2, index=I_4d).squeeze(2) # -> (B,L,E)
    return out

def masked_row_softmax(G_logits: torch.Tensor,
                       mask: torch.Tensor | None,
                       temperature: float = 1.0,
                       fallback_self_loop: bool = True) -> torch.Tensor:
    """
    G_logits: (B,N,N)
    mask:     (1 or B,N,N) bool (1 = allowed; control self-loops via diag)
    Returns row-stochastic probs over allowed neighbors.
    """
    B, N, _ = G_logits.shape
    if mask is None:
        return F.softmax(G_logits / temperature, dim=-1)

    m = mask.bool()
    if m.dim() == 3 and m.size(0) == 1:
        m = m.expand(B, -1, -1)  # (B,N,N)

    if fallback_self_loop:
        row_has = m.any(dim=-1)
        if (~row_has).any():
            eye = torch.eye(N, dtype=torch.bool, device=G_logits.device).unsqueeze(0)
            m = torch.where((~row_has).unsqueeze(-1) & eye, True, m)

    neg_inf = torch.finfo(G_logits.dtype).min / 4
    x = torch.where(m, G_logits / temperature, torch.full_like(G_logits, neg_inf))
    x = x - x.amax(dim=-1, keepdim=True)
    exps = torch.exp(x) * m
    denom = exps.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return exps / denom  # (B,N,N)


class DirectedLowRankGraphHead(nn.Module):
    """
    (B,T,N,C) -> (B,N,N) via directed low-rank outer products.
      - rank r=1 gives a rank-1 directed graph: G = a b^T
      - larger r increases capacity: G = sum_k a_k b_k^T

    Args:
      in_dim:     node feature dim C (after temporal projection)
      hidden:     hidden size in node scorers
      rank_r:     low rank (>=1)
      pooling:    'last' or 'mean' over time for node features
      normalize:  'row_softmax' (row-stochastic) or 'sigmoid' (independent probs)
      temperature: softmax/sigmoid temperature
      nonneg:     if True, clamp a,b >= 0 (keeps non-negative weights)
    """
    def __init__(self, in_dim: int, hidden: int = 64, rank_r: int = 1,
                 pooling: str = "last", normalize: str = "sigmoid",
                 temperature: float = 1.0, nonneg: bool = True):
        super().__init__()
        assert pooling in ("last", "mean")
        assert normalize in ("row_softmax", "sigmoid")
        assert rank_r >= 1
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.nonneg = nonneg
        self.rank_r = rank_r

        # Separate source and destination scorers (directed)
        self.src_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, rank_r)   # (B,N,r)
        )
        self.dst_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, rank_r)   # (B,N,r)
        )

    def forward(self, X: torch.Tensor, mask_M: torch.Tensor | None = None):
        """
        X: (B,T,N,C)
        mask_M: (1 or B,N,N) bool; 1=allowed (diag controls self-loops)
        Returns:
          G: (B,N,N)
        """
        H = X[:, -1] if self.pooling == "last" else X.mean(dim=1)  # (B,N,C)

        a = self.src_mlp(H)   # (B,N,r)
        b = self.dst_mlp(H)   # (B,N,r)
        if self.nonneg:
            a = nn.functional.relu(a)
            b = nn.functional.relu(b)

        # Sum of rank-1 outer products over r:
        # G_logits[b,i,j] = sum_k a[b,i,k] * b[b,j,k]
        G_logits = torch.einsum("bik,bjk->bij", a, b)  # (B,N,N)

        if self.normalize == "row_softmax":
            G = masked_row_softmax(G_logits, mask_M, self.temperature, fallback_self_loop=True)
        else:
            G = torch.sigmoid(G_logits / self.temperature)
            if mask_M is not None:
                G = G * mask_M
        return G  # (B,N,N)

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

    def forward(self, x, graph, batch_first=False):
        # print (f"--debug graph shape: {graph.shape} x shape: {x.shape} --")
        if self.Ks < 1:
            raise ValueError(
                f"ERROR: Ks must be a positive integer, received {self.Ks}."
            )
        x_k = x; x_list = [x]
        # print (graph.shape)
        for k in range(1, self.Ks):
            x_k = torch.einsum("bhi,btij->bthj", graph, x_k.clone()) if batch_first \
                else torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))
        return x_list


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
    ):
        super().__init__()
        self.locals = GraphPropagate(Ks=order, gso=supports[0])
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

    def forward(self, x, graph, batch_first=False):
        x_loc = self.locals(x, graph, batch_first)
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
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
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
        # self.dropout = DropPath(dropout_a)
        self.pooling = nn.AvgPool2d(kernel_size=(1, kernel_size[0]), stride=1)
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    mlp_ratio,
                    num_heads,
                    dropout,
                    kernel=size,
                    supports=supports,
                )
                for size in kernel_size
            ]
        )
        self.attn_layers_d1 = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    mlp_ratio,
                    num_heads,
                    dropout,
                    kernel=size,
                    supports=supports,
                )
                for size in kernel_size
            ]
        )
        self.attn_layers_d2 = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    mlp_ratio,
                    num_heads,
                    dropout,
                    kernel=size,
                    supports=supports,
                )
                for size in kernel_size
            ]
        )

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
        # self.temporal_proj = TCNLayer(self.model_dim, self.model_dim, max_seq_length=in_steps)
        self.temporal_proj = nn.Conv2d(
            self.model_dim, self.model_dim, (1, kernel_size[0]), 1, 0
        )
        # get congestion graph
        self.pos_prop_graph = nn.ModuleList([ DirectedLowRankGraphHead( self.model_dim, 
                                                        hidden=self.model_dim*2, 
                                                        rank_r=1,
                                                        pooling="last", 
                                                        normalize="sigmoid",
                                                        temperature=1.0, 
                                                        nonneg=True
                                                    ) for _ in range(num_lags) ])
        self.neg_prop_graph = nn.ModuleList([ DirectedLowRankGraphHead( self.model_dim, 
                                                        hidden=self.model_dim*2, 
                                                        rank_r=1,
                                                        pooling="last", 
                                                        normalize="sigmoid",
                                                        temperature=1.0, 
                                                        nonneg=True
                                                    ) for _ in range(num_lags) ])
        self.num_lags = num_lags
        # define graphs
        self.road_graph, self.plausible_connectivity = supports[0], supports[1]
        # print (f"--debug device  {self.road_graph.device}{self.plausible_connectivity.device}--")
        # define weights for graph aggregation fusion
        self.mix_logits = nn.Parameter(torch.zeros(3)) 
        # define edge index for endpoint proximity graph
        I, J = self.plausible_connectivity.nonzero(as_tuple=True)          # (E,)
        self.edge_index = torch.stack([I, J], dim=0)       # (2, E)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = torch.tensor([]).to(x)
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features = torch.concat([features, tod_emb], -1)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features = torch.concat([features, dow_emb], -1)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features = torch.concat([features, self.dropout(adp_emb)], -1)
        x = torch.cat(
            [x] + [features], dim=-1
        )  # (batch_size, in_steps, num_nodes, model_dim)
        x = self.temporal_proj(x.transpose(1, 3)).transpose(1, 3)

        # ---- Static graphs ---- 
        graph_0 = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))
        graph_0 = self.pooling(graph_0.transpose(0, 2)).transpose(0, 2)
        graph_0 = F.softmax(F.relu(graph_0), dim=-1)
        x_0 = x.clone()
        for attn in self.attn_layers_s:
            x_0 = attn(x_0, graph_0, batch_first=False)     # consistent API

        # ---- Dynamic graphs (per lag), then average over lags ----
        dyn_pos, dyn_neg = [], []

        Pred_pos = []
        Pred_neg = []

        for l in range(self.num_lags):
            A_pos = self.pos_prop_graph[l](x, self.plausible_connectivity)  # (B,N,N)
            A_neg = self.neg_prop_graph[l](x, self.plausible_connectivity)
            Pred_pos.append(dense_to_edges(A_pos.unsqueeze(1), self.edge_index))
            Pred_neg.append(dense_to_edges(A_neg.unsqueeze(1), self.edge_index))
            z1, z2 = x.clone(), x.clone()
            for attn in self.attn_layers_d1:
                z1 = attn(z1, A_pos, batch_first=True)
            for attn in self.attn_layers_d2:
                z2 = attn(z2, A_neg, batch_first=True)

            dyn_pos.append(z1)
            dyn_neg.append(z2)

        x_1 = torch.stack(dyn_pos, dim=0).mean(0)   # (B,T,N,C)
        x_2 = torch.stack(dyn_neg, dim=0).mean(0)

        # graph output fusion
        w = torch.softmax(self.mix_logits, dim=0)        # (3,)
        x = w[0]*x_0 + w[1]*x_1 + w[2]*x_2
                
        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x)
        # (batch_size, in_steps, num_nodes, model_dim)

        out = self.output_proj(x).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        return out, torch.cat(Pred_pos,dim=1), torch.cat(Pred_neg,dim=1)


