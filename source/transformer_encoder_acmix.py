import torch
import torch.nn as nn

from layer_utils import PosWiseFF
from mha_layer import MultiHeadAttention
from lmha_layer import LMHAttention
from pytorch_admin import as_module

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, atv_fun, dropout_rate, dim_k, parameter_sharing, full_attention, num_layers):
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
        # self.residual = as_module(num_res_layers=2 * num_layers, as_parameter=True, embed_dim=d_model)

        # ACmix Integration
        self.conv1x1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.fc = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, inputs, mask=None):
        x = inputs

        x = self.layernorm1(x)

        attn_out, attn_w = self.mha_layer([x, x, x], mask=mask)

        # sublayer1_out = self.layernorm1(self.residual(x, attn_out))
         # Pre ln
        sublayer1_out = x + attn_out
        
        # Stage I: 1x1 Convolution
        conv_out = self.conv1x1(inputs.transpose(1, 2)).transpose(1, 2)

        # Stage II: Self-attention and Convolution paths
        attn_out, attn_w = self.mha_layer([conv_out, conv_out, conv_out], mask=mask)
        conv_path_out = self.fc(conv_out)

        # Combine outputs from both paths
        combined_out = self.alpha * attn_out + self.beta * conv_path_out

        # Continue with Position-Wise Feed Forward Network
        poswiseff_out = self.poswiseff_layer(combined_out)
        sublayer2_out = self.layernorm2(combined_out + poswiseff_out)

        return sublayer2_out, attn_w

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, atv_fun, dropout_rate, dim_k, parameter_sharing, full_attention, return_intermediate=False):
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
            EncoderLayer(d_model, num_heads, d_ff, atv_fun, dropout_rate, dim_k, parameter_sharing, full_attention, num_layers)
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
