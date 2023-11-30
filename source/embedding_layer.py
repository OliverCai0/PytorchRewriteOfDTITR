import torch
import torch.nn as nn
import tensorflow as tf

class EmbeddingLayer(nn.Module):
    def __init__(self, voc_size, d_model, dropout_rate, positional_enc=True):
        super(EmbeddingLayer, self).__init__()

        self.voc_size = voc_size
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.positional_enc = positional_enc

        self.emb_layer = nn.Embedding(voc_size, d_model)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def position_embedding(self, max_len):
        # """
        # Positional Embedding: Adds info about the position of each token using sin and cosine functions

        # Args:
        # - max_len [int]: number of input tokens (length)

        # Shape:
        # - Outputs:
        # - pos_enc: (L,E) where L is the input sequence length, E is the embedding dimension

        # """
        # position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        # pos_enc = torch.sin(position * div_term) + torch.cos(position * div_term)

        # return pos_enc
        """
        Positional Embedding: Adds info about the position of each token using sin and cosine functions

        Args:
        - max_len [int]: number of input tokens (length)

        Shape:
        - Outputs:
        - pos_enc: (L,E) where L is the input sequence length, E is the embedding dimension

        """

        angle = tf.range(self.d_model, dtype=tf.float32)
        angle = 10000 ** (2 * (angle / self.d_model))

        angle = tf.expand_dims(tf.range(max_len, dtype=tf.float32), 1) / angle

        # for i in range(angle.shape[0]):
        #     for j in range(angle.shape[1]):
        #         if j % 2 == 0 :
        #             angle = tf.tensor_scatter_nd_update(angle,[[i,j]],[tf.math.sin(angle[i,j])])
        #         else :
        #             angle = tf.tensor_scatter_nd_update(angle,[[i,j]],[tf.math.cos(angle[i,j])])

        # return tf.cast(angle,dtype=tf.float32)

        values = tf.stack([tf.math.sin(angle[:, 0::2]), tf.math.cos(angle[:, 1::2])], axis=2)

        pos_enc = tf.reshape(values, shape=[tf.shape(values)[0], -1])

        return tf.cast(pos_enc, dtype=tf.float32)

    def forward(self, sequences):
        """

        Args:
        - sequences: input sequences

        Shape:
        - Inputs:
        - sequences: (B,L) where B is the batch size, L is the sequence length
        - Outputs:
        - output: (B,L,E) where B is the batch size, L is the input sequence length,
                        E is the embedding dimension

        """

        max_len = sequences.size(1)

        output = self.emb_layer(sequences) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        if self.positional_enc:  # Add Positional Info
            output = output + torch.tensor(self.position_embedding(max_len).numpy())
            output = self.dropout_layer(output)

        return output
