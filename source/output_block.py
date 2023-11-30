import torch
import torch.nn as nn

class OutputMLP(nn.Module):
    """
    Fully-Connected Feed-Forward Network (FCNN)

    Args:
    - mlp_depth [int]: number of dense layers for the FCNN
    - mlp_units [list of ints]: number of hidden neurons for each one of the dense layers
    - atv_fun: dense layers activation function
    - out_atv_fun: final dense layer activation function
    - dropout_rate [float]: % of dropout


    """
    def __init__(self, mlp_depth, mlp_units, atv_fun, out_atv_fun, dropout_rate, **kwargs):
        super(OutputMLP, self).__init__(**kwargs)

        self.mlp_depth = mlp_depth
        self.mlp_units = mlp_units
        self.atv_fun = atv_fun
        self.out_atv_fun = out_atv_fun
        self.dropout_rate = dropout_rate

        self.concatenate_layer = nn.Linear(mlp_units[0] * 2, mlp_units[0])
        self.output_dense = nn.Linear(mlp_units[-1], 1)

        layers = []
        for i in range(mlp_depth - 1):
            layers.append(nn.Linear(mlp_units[i], mlp_units[i+1]))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, inputs):
        """

        Args:
        - inputs: [input1, input2]: input 1 sequences and input 2 sequences

        Shape:
        - Inputs:
        - inputs1: (B, L_1, E_1) where B is the batch size, L_1 is the sequence length for input 1,
                                E_1 is the embedding dimension for input 1
        - inputs2: (B, L_2, E_2) where B is the batch size, L_2 is the sequence length for input 2,
                                E_2 is the embedding dimension for input 2

        - Outputs:
        - out: (B, 1): where B is the batch size

        """

        prot_input = inputs[0][:, 0, :]
        smiles_input = inputs[1][:, 0, :]

        concat_input = torch.cat([prot_input, smiles_input], dim=1)

        concat_input = self.concatenate_layer(concat_input)
        concat_input = self.mlp_head(concat_input)

        out = self.output_dense(concat_input)

        return out
