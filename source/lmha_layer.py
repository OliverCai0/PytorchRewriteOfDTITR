import torch
import torch.nn as nn
import torch.nn.functional as F

# def linear_proj_matrix(dim_k):
#     return nn.Linear(dim_k, dim_k)

class LinearAttentionHead(nn.Module):
    def __init__(self, dropout, E_proj, F_proj):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        Q, K, V = inputs

        if mask is not None:
            mask = self.mask[:, 0, 0, :]
            K = torch.where(self.mask, torch.tensor(0.0), K)
            V = torch.where(self.mask, torch.tensor(0.0), V)

        dim_k = K.size(-1)
        scale = 1 / torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))

        K = K.permute(0, 2, 1)
        K = self.E(K)

        V = V.permute(0, 2, 1)
        V = self.F(V)
        V = V.permute(0, 2, 1)
        Q = torch.matmul(Q, K)

        scaled_attention_scores = Q * scale

        attention_weights = F.softmax(scaled_attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights


# It seems like the E_proj and F_proj are just linear here, this probably simplifies the argument passing
class LMHAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate, parameter_sharing, dim_k):
        super(LMHAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.parameter_sharing = parameter_sharing
        self.dim_k = dim_k
        # self.E_proj = nn.Linear(dim_k, dim_k)
        # self.F_proj = nn.Linear(dim_k, dim_k)

    # def build(self, input_shape):
        self.heads = nn.ModuleList()
        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        if self.parameter_sharing != "layerwise":
            self.E_proj = nn.Linear(self.dim_k, self.dim_k)
            self.F_proj = nn.Linear(
                self.dim_k, self.dim_k) if self.parameter_sharing == "none" or self.parameter_sharing == "headwise" else self.E_proj

        for _ in range(self.num_heads):
            if self.parameter_sharing == "none":
                self.E_proj = nn.Linear(self.dim_k, self.dim_k)
                self.F_proj = nn.Linear(self.dim_k, self.dim_k)
            attn = LinearAttentionHead(self.dropout_rate, self.E_proj, self.F_proj)
            self.heads.append(attn)
            self.to_q.append(nn.Linear(self.d_model // self.num_heads, self.d_model // self.num_heads))
            self.to_k.append(nn.Linear(self.d_model // self.num_heads, self.d_model // self.num_heads))
            self.to_v.append(nn.Linear(self.d_model // self.num_heads, self.d_model // self.num_heads))

        self.out = nn.Linear(self.d_model, self.d_model)

        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, inputs, mask):
        query, key, value = inputs

        head_outputs = []
        head_weights = [] # what is this doing here?

        for index, head in enumerate(self.heads):
            Q = self.to_q[index](query)
            K = self.to_k[index](key)
            V = self.to_v[index](value)
            head_outputs.append(head([Q, K, V], mask))

        mha_out = torch.cat([i[0] for i in head_outputs], dim=-1)
        mha_weights = torch.cat([torch.unsqueeze(i[1], dim=1) for i in head_outputs], dim=1)
        mha_out = self.dropout_layer(self.out(mha_out))

        return mha_out, mha_weights
