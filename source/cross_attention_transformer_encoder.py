import torch
import torch.nn as nn
import torch.nn.functional as F

from mha_layer import MultiHeadAttention
from lmha_layer import LMHAttention
from layer_utils import PosWiseFF

class CrossAttnLayer(nn.Module):
    def __init__(self, d_model, cross_num_heads, x1_num_heads, x2_num_heads,
                 x1_d_ff, x2_d_ff, atv_fun, dropout_rate, x1_dim_k,
                 x1_parameter_sharing, x1_full_attention,
                 x2_dim_k, x2_parameter_sharing, x2_full_attention, **kwargs):
        super(CrossAttnLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.cross_num_heads = cross_num_heads
        self.x1_num_heads = x1_num_heads
        self.x2_num_heads = x2_num_heads
        self.x1_d_ff = x1_d_ff
        self.x2_d_ff = x2_d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate
        self.x1_dim_k = x1_dim_k
        self.x1_parameter_sharing = x1_parameter_sharing
        self.x1_full_attention = x1_full_attention
        self.x2_dim_k = x2_dim_k
        self.x2_parameter_sharing = x2_parameter_sharing
        self.x2_full_attention = x2_full_attention

        self.mha_layer_1 = MultiHeadAttention(self.d_model, self.cross_num_heads, self.dropout_rate) #x12
        self.mha_layer_2 = MultiHeadAttention(self.d_model, self.cross_num_heads, self.dropout_rate) #x21
        self.ln_1 = nn.LayerNorm(self.d_model, eps=1e-5)
        self.ln_2 = nn.LayerNorm(self.d_model, eps=1e-5)
        self.ln_3 = nn.LayerNorm(self.d_model, eps=1e-5)
        self.ln_4 = nn.LayerNorm(self.d_model, eps=1e-5)
        self.ln_5 = nn.LayerNorm(self.d_model, eps=1e-5)
        self.ln_6 = nn.LayerNorm(self.d_model, eps=1e-5)

        if self.x1_full_attention:
            self.mha_layer_3 = MultiHeadAttention(self.d_model, self.x1_num_heads, self.dropout_rate) #x21
        else:
            self.mha_layer_3 = LMHAttention(self.d_model, self.x1_num_heads, self.dropout_rate,
                                            self.x1_parameter_sharing,
                                            self.x1_dim_k)

        if self.x2_full_attention:
            self.mha_layer_4 = MultiHeadAttention(self.d_model, self.x2_num_heads, self.dropout_rate) #x12
        else:
            self.mha_layer_4 = LMHAttention(self.d_model, self.x2_num_heads, self.dropout_rate,
                                            self.x2_parameter_sharing,
                                            self.x2_dim_k)

        self.poswiseff_layer_1 = PosWiseFF(self.d_model, self.x1_d_ff, self.atv_fun, self.dropout_rate)
        self.poswiseff_layer_2 = PosWiseFF(self.d_model, self.x2_d_ff, self.atv_fun, self.dropout_rate)

    def rearrange_qkv(self, input1, input2):
        input1_pred_token = input1[:, 0, :].unsqueeze(1)
        input1_tokens = input1[:, 1:, :]
        input2_pred_token = input2[:, 0, :].unsqueeze(1)
        input2_tokens = input2[:, 1:, :]

        return input1_pred_token, input1_tokens, input2_pred_token, input2_tokens

    def forward(self, inputs,  mask_x12, mask_x21):
        x1_p_t, x1_t, x2_p_t, x2_t = self.rearrange_qkv(inputs[0], inputs[1])

        x12_qkv = torch.cat([x1_p_t, x2_t], dim=1)
        x21_qkv = torch.cat([x2_p_t, x1_t], dim=1)


        # print(f'DEBUG: x12_qkv {x12_qkv.size()}, expanded: {x12_qkv[:,0,:].size()}')
        # exit()
        attn_x12_out, attn_x12_w = self.mha_layer_1([ x12_qkv[:,0,:].unsqueeze(1), x12_qkv, x12_qkv], mask_x12)
        attn_x21_out, attn_x21_w = self.mha_layer_2([x12_qkv[:,0,:].unsqueeze(1), x21_qkv, x21_qkv], mask_x21)

        x1_p_t_cross = self.ln_1(x1_p_t + attn_x12_out)
        #print(f'Debug x1_p_t_cross: {x1_p_t_cross.size()}, x1_p_t: {x1_p_t.size()}, attn_x12_out: {attn_x12_out.size()}')
        x2_p_t_cross = self.ln_2(x2_p_t + attn_x21_out)

        x1_cross = torch.cat([x1_p_t_cross, x1_t], dim=1)
        x2_cross = torch.cat([x2_p_t_cross, x2_t], dim=1)

        if self.x1_full_attention:
            #print(f'Debug mask_21: {mask_x21.size()}, mask_12: {mask_x12.size()}')
            attn_x1_out, attn_x1_w = self.mha_layer_3([x1_cross, x1_cross, x1_cross], mask_x21)
        else:
            attn_x1_out, attn_x1_w = self.mha_layer_3([x1_cross, x1_cross, x1_cross], mask_x21)

        if self.x2_full_attention:
            attn_x2_out, attn_x2_w = self.mha_layer_4([x2_cross, x2_cross, x2_cross], mask_x12)
        else:
            attn_x2_out, attn_x2_w = self.mha_layer_4([x2_cross, x2_cross, x2_cross], mask_x12)

        x1_cross = self.ln_3(x1_cross + attn_x1_out)
        x2_cross = self.ln_4(x2_cross + attn_x2_out)

        x1_cross_posff_out = self.poswiseff_layer_1(x1_cross)
        x2_cross_posff_out = self.poswiseff_layer_2(x2_cross)

        x1_cross = self.ln_5(x1_cross + x1_cross_posff_out)
        x2_cross = self.ln_6(x2_cross + x2_cross_posff_out)

        return [x1_cross, x2_cross], attn_x12_w, attn_x21_w, attn_x1_w, attn_x2_w

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, num_layers, cross_num_heads, x1_num_heads, x2_num_heads,
                 x1_d_ff, x2_d_ff, atv_fun, dropout_rate, x1_dim_k,
                 x1_parameter_sharing, x1_full_attention,
                 x2_dim_k, x2_parameter_sharing, x2_full_attention,
                 return_intermediate=False, **kwargs):

        super(CrossAttnBlock, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.cross_num_heads = cross_num_heads
        self.x1_num_heads = x1_num_heads
        self.x2_num_heads = x2_num_heads
        self.x1_d_ff = x1_d_ff
        self.x2_d_ff = x2_d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate
        self.x1_dim_k = x1_dim_k
        self.x1_parameter_sharing = x1_parameter_sharing
        self.x1_full_attention = x1_full_attention
        self.x2_dim_k = x2_dim_k
        self.x2_parameter_sharing = x2_parameter_sharing
        self.x2_full_attention = x2_full_attention
        self.return_intermediate = return_intermediate

        self.cross_attn_layers = [CrossAttnLayer(self.d_model, self.cross_num_heads, self.x1_num_heads,
                                                 self.x2_num_heads,
                                                 self.x1_d_ff, self.x2_d_ff, self.atv_fun,
                                                 self.dropout_rate, self.x1_dim_k,
                                                 self.x1_parameter_sharing,
                                                 self.x1_full_attention, self.x2_dim_k,
                                                 self.x2_parameter_sharing,
                                                 self.x2_full_attention)
                                  for i in range(self.num_layers)]
        
    def forward(self, inputs, x12_mask, x21_mask):
        x = inputs
        intermediate = []
        attention_weights = {}

        for layer in self.cross_attn_layers:
            x, x12_attn_w, x21_attn_w, x1_cross_attn_w, x2_cross_attn_w = layer(x, x12_mask, x21_mask)

            if self.return_intermediate:
                intermediate.append(x)

            attention_weights['attn_weights_layer{}'.format(self.cross_attn_layers.index(layer) + 1)] = [x12_attn_w,
                                                                                                        x21_attn_w,
                                                                                                        x1_cross_attn_w,
                                                                                                        x2_cross_attn_w]

        if self.return_intermediate:
            return torch.stack(intermediate, dim=0), attention_weights

        return x, attention_weights