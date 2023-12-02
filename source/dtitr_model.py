# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import os
import glob
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from transformer_encoder import Encoder
from cross_attention_transformer_encoder import CrossAttnBlock
from embedding_layer import EmbeddingLayer
from layer_utils import AttnPadMask, add_reg_token
from output_block import OutputMLP
from dataset_builder_util import dataset_builder
from argument_parser import argparser, logging
import time
import tensorflow as tf
import wandb

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def c_index(y_true, y_pred):
    """
    Concordance Index Function

    Args:
    - y_true: true values
    - y_pred: predicted values

    """

    y_pred = torch.tensor(y_pred, requires_grad=True)
    y_true = torch.tensor(y_true)

    matrix_pred = y_pred.view(-1, 1) - y_pred
    matrix_pred = (matrix_pred == 0.0).float() * 0.5 + (matrix_pred > 0.0).float()

    matrix_true = (y_true.view(-1, 1) - y_true) > 0.0
    matrix_true = matrix_true.float()

    matrix_true_position = torch.nonzero(matrix_true)

    matrix_pred_values = matrix_pred[matrix_true_position[:, 0], matrix_true_position[:, 1]]

    # If equal to zero then it returns zero, else return the result of the division
    result = torch.where(matrix_pred_values.sum() == 0, torch.tensor(0.0),
                         matrix_pred_values.sum() / matrix_true.sum())

    return result.item()

class DTITR(nn.Module):
    def __init__(self, FLAGS, prot_trans_depth, smiles_trans_depth, cross_attn_depth,
                      prot_trans_heads, smiles_trans_heads, cross_attn_heads,
                      prot_parameter_sharing, prot_dim_k,
                      prot_d_ff, smiles_d_ff, d_model, dropout_rate, dense_atv_fun,
                      out_mlp_depth, out_mlp_units):
        super(DTITR, self).__init__()
        self.prot_mask = AttnPadMask()
        self.smiles_mask = AttnPadMask()

        if FLAGS.bpe_option[0]:
            self.encode_prot = EmbeddingLayer(FLAGS.protein_dict_bpe_len + 2, d_model,  # FLAGS.protein_bpe_len+1,
                                     dropout_rate, FLAGS.pos_enc_option)
            
        else:
            self.encode_prot = EmbeddingLayer(FLAGS.protein_dict_len + 2, d_model,  # FLAGS.protein_len+1,
                                        dropout_rate, FLAGS.pos_enc_option)
            
        self.encoder_prot_module = Encoder(d_model, prot_trans_depth, prot_trans_heads, prot_d_ff, dense_atv_fun,
                                dropout_rate, prot_dim_k, prot_parameter_sharing,
                                FLAGS.prot_full_attn,
                                FLAGS.return_intermediate)
        
        if FLAGS.bpe_option[1]:
            self.encode_smiles = EmbeddingLayer(FLAGS.smiles_dict_bpe_len + 2, d_model,  # FLAGS.smiles_bpe_len+1,
                                        dropout_rate, FLAGS.pos_enc_option)
        else:
            self.encode_smiles = EmbeddingLayer(FLAGS.smiles_dict_len + 2, d_model,  # FLAGS.smiles_len+1,
                                        dropout_rate, FLAGS.pos_enc_option)
            
        self.encoder_smiles_module = Encoder(d_model, smiles_trans_depth, smiles_trans_heads, smiles_d_ff, dense_atv_fun,
                               dropout_rate, FLAGS.smiles_dim_k, FLAGS.smiles_parameter_sharing,
                               FLAGS.smiles_full_attn, FLAGS.return_intermediate)
        
        self.cross_prot_smiles = CrossAttnBlock(d_model, cross_attn_depth, cross_attn_heads, prot_trans_heads,
                                          smiles_trans_heads, prot_d_ff, smiles_d_ff, dense_atv_fun,
                                          dropout_rate, prot_dim_k, prot_parameter_sharing,
                                          FLAGS.prot_full_attn, FLAGS.smiles_dim_k,
                                          FLAGS.smiles_parameter_sharing, FLAGS.smiles_full_attn,
                                          FLAGS.return_intermediate
        )

        self.out = OutputMLP(d_model, out_mlp_depth, out_mlp_units, dense_atv_fun,
                    FLAGS.output_atv_fun, dropout_rate)
        self.cuda_available = torch.cuda.is_available()

    def forward(self, prot, smiles):
        prot_mask = self.prot_mask(prot)#x1
        smiles_mask = self.smiles_mask(smiles)#x2

        prot_encoding = self.encode_prot(prot)
        smiles_encoding = self.encode_smiles(smiles)

        encoded_prot_for_cross, _ = self.encoder_prot_module(prot_encoding, prot_mask)
        encoded_smiles_for_cross, _ = self.encoder_smiles_module(smiles_encoding, smiles_mask)
        cross_prot_smiles_out, _ = self.cross_prot_smiles([encoded_prot_for_cross, encoded_smiles_for_cross], smiles_mask, prot_mask)
        return self.out(cross_prot_smiles_out)


def convert_tf_tensor_to_pytorch(tf_tensor):
    return torch.tensor(tf_tensor.numpy())



def run_train_model(FLAGS):
    """
    Run Train function

    Args:
    - FLAGS: arguments object

    """
    wandb.login()

    flag_config = vars(FLAGS)
    wandb.init(
        project='PytorchDTITR',
        config=flag_config
    )  

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    protein_data, smiles_data, kd_values = dataset_builder(FLAGS.data_path).transform_dataset(FLAGS.bpe_option[0],
                                                                                              FLAGS.bpe_option[1],
                                                                                              'Sequence',
                                                                                              'SMILES',
                                                                                              'Kd',
                                                                                              FLAGS.protein_bpe_len,
                                                                                              FLAGS.protein_len,
                                                                                              FLAGS.smiles_bpe_len,
                                                                                              FLAGS.smiles_len)
    # print("Converting protein data to pytorch")
    # protein_data = convert_tf_tensor_to_pytorch(protein_data)
    # print("Converting smiles data to pytorch")
    # smiles_data = convert_tf_tensor_to_pytorch(smiles_data)
    # print("Converting kd values to torch")
    # kd_values = torch.tensor(kd_values)
    # print("Done")

    if FLAGS.bpe_option[0] == True:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_bpe_len)
    else:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_len)

    if FLAGS.bpe_option[1] == True:
        smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_bpe_len)
    else:
        smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_len)

        # kd_values = tf.expand_dims(kd_values,axis=1)

    _, _, _, clusters, _, _, _, _ = dataset_builder(FLAGS.data_path).get_data()
    
    train_idx = pd.concat([i.iloc[:, 0] for t, i in clusters if t == 'train'])
    test_idx = [i for t, i in clusters if t == 'test'][0].iloc[:, 0]

    prot_train = tf.gather(protein_data, train_idx)
    prot_test = tf.gather(protein_data, test_idx)

    smiles_train = tf.gather(smiles_data, train_idx)
    smiles_test = tf.gather(smiles_data, test_idx)

    kd_train = tf.gather(kd_values, train_idx)
    kd_test = tf.gather(kd_values, test_idx)

    FLAGS.optimizer_fn = FLAGS.optimizer_fn[0]

    dtitr_model = DTITR(FLAGS, FLAGS.prot_transformer_depth[0], FLAGS.smiles_transformer_depth[0],
                                    FLAGS.cross_block_depth[0],
                                    FLAGS.prot_transformer_heads[0], FLAGS.smiles_transformer_heads[0],
                                    FLAGS.cross_block_heads[0],
                                    FLAGS.prot_parameter_sharing[0], FLAGS.prot_dim_k[0],
                                    FLAGS.prot_ff_dim[0], FLAGS.smiles_ff_dim[0], FLAGS.d_model[0],
                                    FLAGS.dropout_rate[0], FLAGS.dense_atv_fun[0],
                                    FLAGS.out_mlp_depth[0], FLAGS.out_mlp_hdim[0])

    if FLAGS.optimizer_fn[0] == 'radam':
        optimizer_fun = torch.optim.RAdam(dtitr_model.parameters(),lr=float(FLAGS.optimizer_fn[1]),
                                                     betas=(float(FLAGS.optimizer_fn[2]),float(FLAGS.optimizer_fn[3])),
                                                     eps=float(FLAGS.optimizer_fn[4]),
                                                     weight_decay=float(FLAGS.optimizer_fn[5]))
    elif FLAGS.optimizer_fn[0] == 'adam':
        optimizer_fun = torch.optim.RAdam(dtitr_model.parameters(),lr=float(FLAGS.optimizer_fn[1]),
                                                     betas=(float(FLAGS.optimizer_fn[2]),float(FLAGS.optimizer_fn[3])),
                                                     eps=float(FLAGS.optimizer_fn[4]),
                                                     weight_decay=float(FLAGS.optimizer_fn[5]))

    elif FLAGS.optimizer_fn[0] == 'adamw':
        optimizer_fun = torch.optim.AdamW(dtitr_model.parameters(),lr=float(FLAGS.optimizer_fn[1]),
                                                     betas=(float(FLAGS.optimizer_fn[2]),float(FLAGS.optimizer_fn[3])),
                                                     eps=float(FLAGS.optimizer_fn[4]),
                                                     weight_decay=float(FLAGS.optimizer_fn[5]))
    

    criterion = nn.MSELoss()
    prot_train = convert_tf_tensor_to_pytorch(prot_train)
    smiles_train = convert_tf_tensor_to_pytorch(smiles_train)
    kd_train = convert_tf_tensor_to_pytorch(kd_train)
    prot_test = convert_tf_tensor_to_pytorch(prot_test)
    smiles_test = convert_tf_tensor_to_pytorch(smiles_test)
    kd_test = convert_tf_tensor_to_pytorch(kd_test)

    data_loader = DataLoader(list(zip(prot_train, smiles_train, kd_train)), batch_size=FLAGS.batch_dim[0])
    test_loader = DataLoader(list(zip(prot_test, smiles_test, kd_test)), batch_size=FLAGS.batch_dim[0])

    wandb.watch(
        models = dtitr_model,
        criterion=torch.nn.functional.mse_loss,
        log='all',
        log_freq=1,
    )

    dtitr_model.train()
    for epoch in range(FLAGS.num_epochs[0]):
        for _, (prot_batch, smiles_batch, kd_batch) in enumerate(data_loader):
            model_outputs = dtitr_model(prot_batch, smiles_batch)
            loss = criterion(model_outputs.squeeze(dim=1), kd_batch)
            optimizer_fun.zero_grad()
            loss.backward()
            optimizer_fun.step()
        
        dtitr_model.eval()
        with torch.no_grad():
            total_loss = 0
            total_evals = 0
            for _, (prot_test_batch, smiles_test_batch, kd_test_batch) in enumerate(test_loader):
                test_output = dtitr_model(prot_test_batch, smiles_test_batch)
                test_loss = criterion(test_output.squeeze(dim=1), kd_test_batch) 
                total_loss += test_loss
                total_evals += 1
            wandb.log(
                {"mse_loss_per_epoch" : total_loss / total_evals}
            )
            print(f'Epoch {epoch + 1}/{FLAGS.num_epochs[0]}, MSE_LOSS = {total_loss / total_evals}')
        dtitr_model.train()
        if test_loss <= 0.001:
            break

    # mse, rmse, ci = dtitr_model.evaluate([prot_test, smiles_test], kd_test)

    if FLAGS.hugging_save:
        MODELPATH = os.path.join(os.getcwd(), f'../pytorchmodel/{FLAGS.hugging_save}.pth')
        torch.save(dtitr_model.state_dict(), MODELPATH)
        api = HfApi()
        api.upload_file(
            path_or_fileobj= MODELPATH,  
            path_in_repo=f'DTITR-{FLAGS.hugging_save}.pth',
            repo_id="DLSAutumn2023/DTITR_Recreation"
        )
        # dtitr_model.push_to_hub(f'DLSAutumn2023/DTITR_Recreation/DTITR-{FLAGS.hugging_save}')

    # Save the model to physical drive


    # logging("Test Fold - " + (" MSE = %0.3f, RMSE = %0.3f, CI = %0.3f" % (mse, rmse, ci)), FLAGS)

if __name__ == "__main__":
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if physical_devices:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.data_path = {'data': '../data/davis/dataset/davis_dataset_processed.csv',
                       'prot_dic': '../dictionary/davis_prot_dictionary.txt',
                       'smiles_dic': '../dictionary/davis_smiles_dictionary.txt',
                       'clusters': glob.glob('../data/davis/clusters/*'),
                       'prot_bpe': ['../dictionary/protein_codes_uniprot.txt',
                                    '../dictionary/subword_units_map_uniprot.csv'],
                       'smiles_bpe': ['../dictionary/drug_codes_chembl.txt',
                                      '../dictionary/subword_units_map_chembl.csv']}

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    logging(str(FLAGS), FLAGS)

    print(FLAGS.option)

    if FLAGS.option == 'Train':
        run_train_model(FLAGS)