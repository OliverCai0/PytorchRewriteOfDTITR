import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosWiseFF(nn.Module):
    def __init__(self, d_model, d_ff, atv_fun, dropout_rate):
        super(PosWiseFF, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate

        self.dense_1 = nn.Linear(d_model, d_ff)
        self.dense_2 = nn.Linear(d_ff, d_model)
        self.dropout_layer_1 = nn.Dropout(dropout_rate)
        self.dropout_layer_2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dense_1(x)
        x = F.dropout(F.relu(x), p=self.dropout_rate, training=self.training)
        x = self.dense_2(x)
        x = self.dropout_layer_2(x)

        return x

class AttnPadMask(nn.Module):
    def __init__(self):
        super(AttnPadMask, self).__init__()

    def forward(self, x):
        mask = (x == 0).float().unsqueeze(1).unsqueeze(2)
        return mask

def add_reg_token(x, voc_size):
    """
    Rp and Rs Tokens Function: adds the Rp or the Rs token to the input sequences

    Args:
    - x: inputs sequences
    - voc_size [int]: number of unique tokens

    Shape:
    - Inputs:
    - x: (B,L) where B is the batch size, L is the sequence length
    - Outputs:
    - x: (B,1+L) where B is the batch size, L is the input sequence length

    """

    reg_token = tf.convert_to_tensor(voc_size + 1, dtype=tf.int32)
    broadcast_shape = tf.where([True, False], tf.shape(x), [0, 1])
    reg_token = tf.broadcast_to(reg_token, broadcast_shape)

    return tf.concat([reg_token, x], axis=1)