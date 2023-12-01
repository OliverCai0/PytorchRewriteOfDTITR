import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.cuda_available = torch.cuda.is_available()

    def forward(self, inputs, mask):
        query, key, value = inputs
        # print('query shape', query.size())
        # print('key shape', key.size())
        # print('value shape', value.size())

        dim_k = key.size(-1)
        scale = 1 / torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))

        matmul_q_transp_k = torch.matmul(query, key.transpose(-2, -1))
        scaled_attention_scores = matmul_q_transp_k * scale

        if mask is not None:
            # print(mask.size())
            # print(scaled_attention_scores.size())
            scaled_attention_scores += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        assert d_model % num_heads == 0

        self.attention = ScaledDotProductAttention(dropout_rate)

        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)

        self.reshape = lambda x: x.view(x.size()[0], -1, self.num_heads, self.d_model // self.num_heads)
        # self.transpose = nn.Permute((2, 1, 3))

        # self.transpose_attn_output = nn.Permute((2, 1, 3))
        # self.transpose_attn_output = lambda x : torch.permute(x,(2, 1, 3))
        # self.reshape_attn_output = nn.Reshape((-1, d_model))
        self.reshape_attn_output = lambda x: x.view(x.size()[0], -1, self.d_model)

        self.out = nn.Linear(d_model, d_model)

        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, inputs, mask):
        query, key, value = inputs

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # print('query shape before reshape', query.size())
        # print('key shape before reshape', key.size())
        # print('value shape before reshape', value.size())

        query = self.reshape(query)
        key = self.reshape(key)
        value = self.reshape(value)

        # query = self.transpose(self.reshape(query))
        # key = self.transpose(self.reshape(key))
        # value = self.transpose(self.reshape(value))

        permute_tuple = (0, 2,1,3)
        

        query = torch.permute(query,permute_tuple)
        key = torch.permute(key,permute_tuple)
        value = torch.permute(value,permute_tuple)

        attention_output, attention_weights = self.attention([query, key, value], mask)

        # attention_output = self.transpose_attn_output(attention_output)
        torch.permute(attention_output,permute_tuple)
        attention_output = self.reshape_attn_output(attention_output)

        mh_attention_output = self.dropout_layer(self.out(attention_output))

        return mh_attention_output, attention_weights
