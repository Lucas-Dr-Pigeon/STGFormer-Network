import torch.nn as nn
import torch
import numpy as np
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
from typing import Tuple

class MotifTopoEncoder(nn.Module):
    """
    Encode motif-based per-node features into topo embeddings z_topo.

    Input:
        motif_list: list of K LongTensors (M_k, N_k)
        num_nodes : N

    Output:
        z_topo: (N, d_t)
    """
    def __init__(self, d_t: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = None
        self.d_t = d_t
        self.hidden_dim = hidden_dim

    def build_mlp(self, in_dim: int):
        # simple 2-layer MLP, build lazily once we know in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.d_t),
        )

    def forward(self, z):
        # 1) build raw motif features

        # 2) simple log1p transform to tame large counts
        feat = torch.log1p(z)

        # 3) lazy-create MLP if not built yet
        if self.mlp is None:
            in_dim = feat.shape[-1]
            self.build_mlp(in_dim)
            self.mlp.to(feat.device)

        # 4) project to topo embeddings
        z_topo = self.mlp(feat)  # (N, d_t)

        return z_topo

class TopoAugmentedQK(nn.Module):
    """
    GTAT-style topology augmentation for Q/K.

    x      : (B, T, N, C)   feature tensor
    z_topo : (N, d_t)       per-node topology embeddings

    Returns:
    Q, K, V: (B, T, N, C)   same last-dim as x (model_dim), so head_dim unchanged
    """
    def __init__(
        self,
        model_dim: int,
        topo_dim: int,
        num_heads: int,
        topo_proj_dim: int = None,   # must equal model_dim for "add"
        init_beta: float = 0.05,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.topo_proj_dim = topo_proj_dim or model_dim
        assert self.topo_proj_dim == model_dim, \
            "For additive fusion, topo_proj_dim must equal model_dim"

        # Feature branch
        self.qkv_feat = nn.Linear(model_dim, 3 * model_dim)

        # Topology branch
        self.q_topo = nn.Linear(topo_dim, self.topo_proj_dim)
        self.k_topo = nn.Linear(topo_dim, self.topo_proj_dim)

        # Learnable scale for topo contribution (≈ sqrt(beta))
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,        # (B, T, N, C)
        z_topo: torch.Tensor,   # (N, d_t)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, N, C = x.shape
        assert C == self.model_dim
        assert z_topo.shape[0] == N, "z_topo must have one embedding per node"

        # ---- feature Q/K/V ----
        qkv_feat = self.qkv_feat(x)  # (B, T, N, 3*C)
        q_feat, k_feat, v_feat = qkv_feat.chunk(3, dim=-1)  # each (B, T, N, C)

        # ---- topo Q/K ----
        # z_topo: (N, d_t) -> (1,1,N,d_t) -> (B,T,N,d_t)
        z_bt = z_topo.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        q_topo = self.q_topo(z_bt)  # (B, T, N, C)
        k_topo = self.k_topo(z_bt)  # (B, T, N, C)

        # add topo contribution
        scale = self.beta
        Q = q_feat + scale * q_topo  # (B, T, N, C)
        K = k_feat + scale * k_topo  # (B, T, N, C)
        V = v_feat                   # (B, T, N, C)
        # print (f"--q_feat mean {torch.mean(q_feat)}, q_topo mean {scale.item()*torch.mean(q_topo)}  k_feat mean {torch.mean(k_feat)}, k_topo mean {scale.item()*torch.mean(k_topo)} beta {scale.item()}---")
        return Q, K, V

class FastAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, topo_dim=8, kernel=1):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.qkv = TopoAugmentedQK(model_dim, topo_dim, num_heads, model_dim)

        self.out_proj = nn.Linear(
            2 * model_dim if kernel != 12 else model_dim, model_dim
        )
        # self.proj_in = nn.Conv2d(model_dim, model_dim, (1, kernel), 1, 0) if kernel > 1 else nn.Identity()
        self.fast = 1

    def forward(self, x, z, dim=0):
        # z: topology embedding
        query, key, value = self.qkv(x, z)
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


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        mlp_ratio=2,
        num_heads=8,
        dropout=0,
        topo_dim=7,
        kernel=3,
        supports=None,
        order=2,
    ):
        super().__init__()
        self.locals = GraphPropagate(Ks=order, gso=supports[0])
        self.attn = nn.ModuleList(
            [
                FastAttentionLayer(model_dim, num_heads, topo_dim, kernel=1)
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

    def forward(self, x, graph, z_topo):
        x_loc = self.locals(x, graph)
        x_glo = x                       # start from x, but DON'T modify in-place
        c = x
        for i, x_l in enumerate(x_loc):
            att_outputs = self.attn[i](x_l, z_topo)
            # x_glo += att_outputs * self.pws[i](c) * self.scale[i]
            gate = self.pws[i](c) * self.scale[i]
            x_glo = x_glo + att_outputs * gate  # out-of-place
            c = att_outputs
        x = self.ln1(x + self.dropout(x_glo))
        x = self.ln2(x + self.dropout(self.fc(x)))
        return x


class TopoSTGformer(nn.Module):
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
        supports=None,
        topo_feature=None,
        num_layers=3,
        dropout=0.1,
        mlp_ratio=2,
        use_mixed_proj=True,
        dropout_a=0.3,
        kernel_size=[1],
        order=2,
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

        self.z = topo_feature / (topo_feature.norm(dim=1, keepdim=True) + 1e-6) # (N, K)
        self.topo_embedding_dim = topo_feature.shape[1]

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
                    self.topo_embedding_dim,
                    kernel=size,
                    supports=supports,
                    order=order,
                )
                for size in kernel_size
            ]
        )

        self.encoder_proj = nn.Linear(
            (in_steps - sum(k - 1 for k in kernel_size)) * self.model_dim,
            self.model_dim,
        )
        self.kernel_size = kernel_size[0]

        self.topology_encoder = MotifTopoEncoder(self.topo_embedding_dim)
        self.topo_ln = nn.LayerNorm(self.topo_embedding_dim)

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

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        # z: topology (num_nodes, topology_embedding_dim) -> fixed
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        z_topo = self.topo_ln(self.topology_encoder(self.z)) # z_topo is fixed
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
        graph = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))
        graph = self.pooling(graph.transpose(0, 2)).transpose(0, 2)
        graph = F.softmax(F.relu(graph), dim=-1)
        # print (f"--debug graph shape: {graph.shape} x shape: {x.shape} --")
        for attn in self.attn_layers_s:
            x = attn(x, graph, z_topo)
        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x)
        # (batch_size, in_steps, num_nodes, model_dim)

        out = self.output_proj(x).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        return out


