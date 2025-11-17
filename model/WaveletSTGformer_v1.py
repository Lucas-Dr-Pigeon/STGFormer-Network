import torch.nn as nn
import torch
import numpy as np
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
import math

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
        # print (f"-- Debug transformer shape error: {x.shape, qs.shape, ks.shape, vs.shape} --")
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

class GraphWaveletLayer(torch.nn.Module):
    """docstring for GraphWaveletLayer"""
    def __init__(self, in_channels, out_channels, n_nodes, scale_dim, dropout=0.3):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = n_nodes
        self.out_channels = n_nodes
        self.scale_dim = scale_dim
        self.dropout = dropout
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        self.diag_weights = torch.nn.Parameter(torch.Tensor(self.scale_dim, self.in_channels)) 
        # print (self.diag_weights.shape, ' -- weight shape')
        self.bias = torch.nn.Parameter(torch.FloatTensor(self.in_channels))

    def init_parameters(self): 
        stdv = 1. / math.sqrt(self.in_channels)
        with torch.no_grad():
            self.diag_weights.uniform_(-stdv, stdv)
            self.bias.uniform_(-stdv, stdv)
    
    def forward(self, input, phi_matrices, phi_inverse_matrices):
        # input: (B, T, N, C)
        # phi_matrices, phi_inverse_matrices: (S, N, N)
        # self.diag_weights: (S, N)

        B, T, N, C = input.shape
        x = input.permute(0, 3, 2, 1) # reshape to (B, C, N, T)


        S = phi_matrices.shape[0]

        # Step 1: Diagonal scale weights to matrix
        diag = torch.diag_embed(self.diag_weights)  # (S, N, N)

        # print (f"-- Debug: {phi_matrices.device}, {diag.device} --")
        # Step 2: Compute φ @ diag @ φ⁻¹
        phi_product = phi_matrices @ diag @ phi_inverse_matrices  # (S, N, N)

        if torch.isnan(phi_product).any():
            print("NaN in phi_product")

        # Step 3: Reshape input: (B, C, N, T) → (B*C*T, N)
        x = x.permute(0, 1, 3, 2).reshape(-1, N)  # (B*C*T, N)

        # Step 4: Apply wavelet transform
        x_trans = torch.einsum('sij,bj->sbi', phi_product, x)  # (S, BCT, N)
        x_trans = x_trans.permute(1, 0, 2)  # (BCT, S, N)

        # Step 5: Aggregate across scales (e.g., sum)
        h = x_trans.sum(dim=1)  # (BCT, N)

        # Step 6: Reshape back to (B, C, N, T)
        h = h.reshape(B, C, T, N).permute(0, 1, 3, 2)  # (B, C, N, T)

        h = h.permute(0, 3, 2, 1) # (B, T, N, C)

        # Removed final projection from current version
        h = F.dropout(h, self.dropout, training=self.training)
        return h, phi_product


class GraphPropagate(nn.Module):
    def __init__(self, Ks, gso, dropout = 0.2):
        super(GraphPropagate, self).__init__()
        self.Ks = Ks
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, graph):
        if self.Ks < 1:
            raise ValueError(
                f"ERROR: Ks must be a positive integer, received {self.Ks}."
            )
        x_k = x; x_list = [x]
        for k in range(1, self.Ks):
            x_k = torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))

        return x_list
    
class GraphWaveletPropagate(nn.Module):
    def __init__(self, model_dim, num_nodes, num_wavelets, order=2, dropout = 0.2):
        super(GraphWaveletPropagate, self).__init__()
        self.wavelet_layer = GraphWaveletLayer(model_dim, model_dim, num_nodes, num_wavelets, dropout)
        self.order = order
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, phi_matrices, inverse_phi_matrices):
        if self.order < 1:
            raise ValueError(
                f"ERROR: Ks must be a positive integer, received {self.order}."
            )
        x_k = x; x_list = [x]
        for k in range(1, self.order):
            x_k, _ = self.wavelet_layer(x_k.clone(), phi_matrices, inverse_phi_matrices)
            x_list.append(x_k)

        return x_list

class SelfAttentionLayerWavelet(nn.Module):
    def __init__(
        self,
        model_dim,
        num_nodes,
        num_wavelets,
        mlp_ratio=2,
        num_heads=8,
        dropout=0,
        mask=False,
        kernel=3,
        order=2,
    ):
        super().__init__()
        self.locals = GraphWaveletPropagate(model_dim, num_nodes, num_wavelets, order, dropout)
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

    def forward(self, x, phi_matrices, invesre_phi_matrices):
        # print (f"-- Debug: check graph shape {graph.shape} --")
        x_loc = self.locals(x, phi_matrices, invesre_phi_matrices)
        c = x_glo = x
        for i, z in enumerate(x_loc):
            att_outputs = self.attn[i](z)
            x_glo += att_outputs * self.pws[i](c) * self.scale[i]
            c = att_outputs
        x = self.ln1(x + self.dropout(x_glo))
        x = self.ln2(x + self.dropout(self.fc(x)))
        return x

# class selfAttentionLayerWavelet(nn.Module):
#     def __init__(
#         self,
#         model_dim,
#         mlp_ratio=2,
#         num_heads=8,
#         dropout=0,
#         mask=False,
#         kernel=3,
#         supports=None,
#         num_wavelets=3,
#         order=2
#     ):
#         super().__init__()
#         num_nodes = supports[0].shape[0]
#         self.locals = GraphWaveletLayer(model_dim, model_dim, num_nodes, num_wavelets, dropout)
#         self.attn = nn.ModuleList(
#             [
#                 FastAttentionLayer(model_dim, num_heads, mask, kernel=kernel)
#                 for _ in range(order)
#             ]
#         )
#         self.pws = nn.ModuleList(
#             [nn.Linear(model_dim, model_dim) for _ in range(order)]
#         )
#         for i in range(0, order):
#             nn.init.constant_(self.pws[i].weight, 0)
#             nn.init.constant_(self.pws[i].bias, 0)

#         self.kernel = kernel
#         self.fc = Mlp(
#             in_features=model_dim,
#             hidden_features=int(model_dim * mlp_ratio),
#             act_layer=nn.ReLU,
#             drop=dropout,
#         )
#         self.ln1 = nn.LayerNorm(model_dim)
#         self.ln2 = nn.LayerNorm(model_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.scale = [1, 0.01, 0.001]

#     def forward(self, x, phi, inv_phi):
#         x_ = x.permute(0,3,2,1)
#         x_, _ = self.locals(x_, phi, inv_phi)
#         x_loc = x_.permute(0,3,2,1)
#         # print (f'-- Debug hidden layer shape:{x_loc.shape} --')
#         c = x_glo = x
#         for i, z in enumerate(x_loc):
#             att_outputs = self.attn[i](z)
#             x_glo += att_outputs * self.pws[i](c) * self.scale[i]
#             c = att_outputs
#         x = self.ln1(x + self.dropout(x_glo))
#         x = self.ln2(x + self.dropout(self.fc(x)))
#         return x

class WaveletSTGformer(nn.Module):
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
        num_layers=3,
        dropout=0.1,
        mlp_ratio=2,
        use_mixed_proj=True,
        dropout_a=0.3,
        kernel_size=[1],
        wavelet_phi_matrices=None,
        wavelet_inverse_phi_matrices=None,

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
        self.model_dim = input_embedding_dim
 
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.graph = torch.stack(supports, dim=0).to(torch.float32)

        # GWNN setup
        self.phi_matrices = wavelet_phi_matrices
        self.inv_phi_matrices = wavelet_inverse_phi_matrices
        num_wavelets = self.phi_matrices.shape[0]

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
                SelfAttentionLayerWavelet(
                    self.model_dim,
                    num_nodes,
                    num_wavelets,
                    mlp_ratio,
                    num_heads,
                    dropout,
                    kernel=size,
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

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        x = self.temporal_proj(x.transpose(1, 3)).transpose(1, 3)
        # print (f'-- Debug hidden layer shape:{x.shape} --')
        for attn in self.attn_layers_s:
            x = attn(x, self.phi_matrices, self.inv_phi_matrices)
        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x)
        # (batch_size, in_steps, num_nodes, model_dim)

        out = self.output_proj(x).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = F.sigmoid(out)
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        return out


