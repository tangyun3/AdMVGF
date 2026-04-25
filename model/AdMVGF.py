import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp


class DynamicParameterGenerator(nn.Module):
    def __init__(self, model_dim, reduction_ratio=4):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(model_dim, model_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(model_dim // reduction_ratio, model_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return torch.ones_like(x)


class FastAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, qkv_bias=False, kernel=1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dyn_gen = DynamicParameterGenerator(model_dim)
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.fast = 0

    def forward(self, x, edge_index=None, dim=0):
        dynamic_weights = self.dyn_gen(x)
        x = x * dynamic_weights
        return self.out_proj(x)

    def fast_attention(self, x, qs, ks, vs, dim=0):
        return x

    def normal_attention(self, x, qs, ks, vs, dim=0):
        return x


class SpatioTemporalCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, dropout=0.1, decay_scales=[1.0, 0.01, 0.001],
                 temporal_kernel=3, spatial_kernel=3):
        super().__init__()
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.decay_scales = decay_scales
        self.dyn_gen = DynamicParameterGenerator(model_dim)
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.attn_layers = nn.ModuleList([nn.Identity() for _ in decay_scales])
        self.temporal_path = nn.Identity()
        self.spatial_path = nn.Identity()
        self.fusion_gate = nn.Linear(2 * model_dim, 2)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)

    def forward(self, x):
        return self.norm1(x + self.out_proj(x))


class GraphPropagate(nn.Module):
    def __init__(self, Ks, gso, dropout=0.05):
        super().__init__()
        self.Ks = Ks
        self.gso = gso
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, graph):
        return [x for _ in range(self.Ks)]


class LSTMThenMultiScaleTC(nn.Module):
    def __init__(
            self, model_dim, kernel_sizes=[3,5,7], dilations=[1,2,4],
            lstm_layers=2, use_residual=True, dropout=0.05
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_residual = use_residual
        self.lstm = nn.LSTM(model_dim, model_dim, batch_first=True)
        self.branches = nn.ModuleList([nn.Identity() for _ in kernel_sizes])
        self.branch_weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_residual:
            return self.norm(x)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, mlp_ratio=2, num_heads=16, dropout=0,
            mask=False, kernel=3, supports=None, order=2
    ):
        super().__init__()
        self.locals = GraphPropagate(Ks=order, gso=supports[0])
        self.attn = nn.ModuleList([FastAttentionLayer(model_dim, num_heads, mask, kernel) for _ in range(order)])
        self.pws = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(order)])
        self.fc = Mlp(model_dim, int(model_dim*mlp_ratio), act_layer=nn.ReLU, drop=dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = [1, 0.01, 0.001]

    def forward(self, x, graph):
        x_loc = self.locals(x, graph)
        x = self.ln1(x)
        x = self.ln2(x + self.fc(x))
        return x


class AdMVGF(nn.Module):
    def __init__(
            self, num_nodes, in_steps=12, out_steps=12, steps_per_day=288,
            input_dim=3, output_dim=1, input_embedding_dim=24,
            tod_embedding_dim=24, dow_embedding_dim=24, holiday_embedding_dim=0,
            spatial_embedding_dim=0, adaptive_embedding_dim=24, num_heads=16,
            supports=None, num_layers=3, dropout=0.1, mlp_ratio=2,
            tconv_kernel=3, use_tconv=True, num_tim_layers=4,
            use_mixed_proj=True, dropout_a=0.3, kernel_size=[1],
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
        self.holiday_embedding_dim = holiday_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + holiday_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.use_tconv = use_tconv

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if holiday_embedding_dim > 0:
            self.holiday_embedding = nn.Embedding(3, holiday_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(torch.randn(in_steps, num_nodes, adaptive_embedding_dim))

        self.dropout = nn.Dropout(dropout_a)
        self.pooling = nn.AvgPool2d(kernel_size=(1, kernel_size[0]), stride=1)
        self.temporal_proj = nn.Conv2d(self.model_dim, self.model_dim, (1, kernel_size[0]), 1, 0)

        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, mlp_ratio, num_heads, dropout, kernel=kernel_size[0], supports=supports)
            for _ in range(num_layers)
        ])

        if self.use_tconv:
            self.tconvs = LSTMThenMultiScaleTC(model_dim=self.model_dim)

        self.encoder_proj = nn.Linear(in_steps * self.model_dim, self.model_dim)
        self.encoder = nn.ModuleList([Mlp(self.model_dim) for _ in range(num_layers)])
        self.output_proj = nn.Linear(self.model_dim, out_steps * output_dim)

        self.cross_attns = nn.ModuleList([
            SpatioTemporalCrossAttention(model_dim=self.model_dim, num_heads=16)
            for _ in range(num_layers)
        ])

        self.gate_weights = nn.Parameter(torch.ones(num_layers, 2))

    def forward(self, x):
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        if self.holiday_embedding_dim > 0:
            holiday = x[..., 3]

        x = x[..., :self.input_dim]
        x = self.input_proj(x)
        features = torch.tensor([]).to(x)

        if self.tod_embedding_dim > 0:
            features = torch.concat([features, self.tod_embedding((tod * self.steps_per_day).long())], -1)
        if self.dow_embedding_dim > 0:
            features = torch.concat([features, self.dow_embedding(dow.long())], -1)
        if self.holiday_embedding_dim > 0:
            features = torch.concat([features, self.holiday_embedding(holiday.long())], -1)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(batch_size, *self.adaptive_embedding.shape)
            features = torch.concat([features, self.dropout(adp_emb)], -1)

        x = torch.cat([x, features], dim=-1)
        x = self.temporal_proj(x.transpose(1, 3)).transpose(1, 3)

        graph = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))
        graph = self.pooling(graph.transpose(0, 2)).transpose(0, 2)
        graph = F.softmax(F.relu(graph), dim=-1)

        for layer_idx in range(self.num_layers):
            x_spatial = self.attn_layers_s[layer_idx](x, graph)

        for layer_idx in range(self.num_layers):
            if self.use_mixed_proj:
                x_cross = self.cross_attns[layer_idx](x_spatial)
                gate = F.softmax(self.gate_weights[layer_idx], dim=0)
                x = gate[0] * x_spatial + gate[1] * x_cross
            if self.use_tconv:
                x = self.tconvs(x)

        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x)

        out = self.output_proj(x).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        return out.transpose(1, 2)