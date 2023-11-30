import torch
import torch.nn as nn

from layer_utils import PosWiseFF
from mha_layer import MultiHeadAttention
from lmha_layer import LMHAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, atv_fun, dropout_rate, dim_k, parameter_sharing, full_attention):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate
        self.dim_k = dim_k
        self.parameter_sharing = parameter_sharing
        self.full_attention = full_attention

        if full_attention:
            self.mha_layer = MultiHeadAttention(d_model, num_heads, dropout_rate)
        else:
            # self.E_proj = linear_proj_matrix(dim_k)
            self.mha_layer = LMHAttention(d_model, num_heads, dropout_rate, parameter_sharing, dim_k)

        self.poswiseff_layer = PosWiseFF(d_model, d_ff, atv_fun, dropout_rate)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-5)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-5)

    def forward(self, inputs, mask=None):
        x = inputs

        attn_out, attn_w = self.mha_layer([x, x, x], mask=mask)

        sublayer1_out = self.layernorm1(x + attn_out)

        poswiseff_out = self.poswiseff_layer(sublayer1_out)

        sublayer2_out = self.layernorm2(sublayer1_out + poswiseff_out)

        return sublayer2_out, attn_w

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, atv_fun, dropout_rate,
                 dim_k, parameter_sharing, full_attention, return_intermediate=False):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate
        self.dim_k = dim_k
        self.parameter_sharing = parameter_sharing
        self.full_attention = full_attention
        self.return_intermediate = return_intermediate

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, atv_fun, dropout_rate, dim_k, parameter_sharing, full_attention)
            for _ in range(num_layers)
        ])

    def forward(self, inputs, mask):
        x = inputs
        intermediate = []
        attention_weights = {}

        for i, layer in enumerate(self.enc_layers):
            x, attn_enc_w = layer(x, mask)

            if self.return_intermediate:
                intermediate.append(x)

            attention_weights[f'encoder_layer{i + 1}'] = attn_enc_w

        if self.return_intermediate:
            return torch.stack(intermediate, dim=0), attention_weights

        return x, attention_weights
