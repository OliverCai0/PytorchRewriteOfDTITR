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
import numpy as np


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

        self.out = OutputMLP(out_mlp_depth, out_mlp_units, dense_atv_fun,
                    FLAGS.output_atv_fun, dropout_rate)
        
    def forward(self, prot, smiles):
        x1_mask = self.prot_mask(prot) #x1
        x2_mask = self.smiles_mask(smiles) #x2

        prot_encoding = self.encode_prot(prot)
        smiles_encoding = self.encode_smiles(smiles)

        encoded_prot_for_cross, _ = self.encoder_prot_module(prot_encoding, x1_mask)
        encoded_smiles_for_cross, _ = self.encoder_smiles_module(smiles_encoding, x2_mask)
        cross_prot_smiles_out, _ = self.cross_prot_smiles(encoded_prot_for_cross, encoded_smiles_for_cross, x2_mask, x1_mask)
        return self.out(cross_prot_smiles_out)


def convert_tf_tensor_to_pytorch(tf_tensor):
    return torch.tensor(tf_tensor.numpy())



def run_train_model(FLAGS):
    """
    Run Train function

    Args:
    - FLAGS: arguments object

    """

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

    print("dataset builder get_data call")
    _, _, _, clusters, _, _, _, _ = dataset_builder(FLAGS.data_path).get_data()
    print("finished")
    

    print("train_idx")
    train_idx = pd.concat([i.iloc[:, 0] for t, i in clusters if t == 'train'])
    test_idx = [i for t, i in clusters if t == 'test'][0].iloc[:, 0]
    print("finished")

    prot_train = tf.gather(protein_data, train_idx)
    prot_test = tf.gather(protein_data, test_idx)

    smiles_train = tf.gather(smiles_data, train_idx)
    smiles_test = tf.gather(smiles_data, test_idx)

    kd_train = tf.gather(kd_values, train_idx)
    kd_test = tf.gather(kd_values, test_idx)

    FLAGS.optimizer_fn = FLAGS.optimizer_fn[0]

    print("trying to instantiate model")
    dtitr_model = DTITR(FLAGS, FLAGS.prot_transformer_depth[0], FLAGS.smiles_transformer_depth[0],
                                    FLAGS.cross_block_depth[0],
                                    FLAGS.prot_transformer_heads[0], FLAGS.smiles_transformer_heads[0],
                                    FLAGS.cross_block_heads[0],
                                    FLAGS.prot_parameter_sharing[0], FLAGS.prot_dim_k[0],
                                    FLAGS.prot_ff_dim[0], FLAGS.smiles_ff_dim[0], FLAGS.d_model[0],
                                    FLAGS.dropout_rate[0], FLAGS.dense_atv_fun[0],
                                    FLAGS.out_mlp_depth[0], FLAGS.out_mlp_hdim[0])
    print('successful')

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
    print("data loader")
    print("Converting protein data to pytorch")
    prot_train = convert_tf_tensor_to_pytorch(prot_train)
    print("Converting smiles data to pytorch")
    smiles_train = convert_tf_tensor_to_pytorch(smiles_train)
    print("Converting kd values to torch")
    kd_train = convert_tf_tensor_to_pytorch(kd_train)
    print("Done")

    data_loader = DataLoader(list(zip(prot_train, smiles_train, kd_train)), shuffle=True, batch_size=FLAGS.batch_dim[0])
    print("Finished")
    dtitr_model.train()
    for epoch in range(FLAGS.num_epochs[0]):
        print("Epoch started")
        for _, (prot_batch, smiles_batch, kd_batch) in enumerate(data_loader):
            model_outputs = dtitr_model(prot_batch, smiles_batch)
            loss = criterion(model_outputs, kd_batch)
            optimizer_fun.zero_grad()
            loss.backward()
            optimizer_fun.step()
        
        dtitr_model.eval()
        with torch.no_grad():
            test_output = dtitr_model(prot_test, smiles_test)
            test_loss = criterion(test_output, kd_test)
            print(f'Epoch {epoch + 1}/{FLAGS.num_epochs[0]}, MSE_LOSS = {test_loss}')
        dtitr_model.train()
        if test_loss <= 0.001:
            break

    # mse, rmse, ci = dtitr_model.evaluate([prot_test, smiles_test], kd_test)

    if FLAGS.hugging_save:
        dtitr_model.save('dtitr_model.h5')
        api = HfApi()
        api.upload_file(
            path_or_fileobj= os.path.join(os.getcwd(), 'dtitr_model.h5'),  
            path_in_repo=f'DTITR-{FLAGS.hugging_save}',
            repo_id="DLSAutumn2023/DTITR_Recreation"
        )

    # Save the model to physical drive
    torch.save(dtitr_model, os.path.join(os.getcwd(), "../pytorchmodel/model.pth"))


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