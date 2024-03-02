"""
Deformable EEG Conformer.
Convolutional Transformer with deformable transformer decoder for EEG decoding.
Modified from https://github.com/eeyhsong/EEG-Conformer.
"""

import os
import numpy as np
import math
import random

import time
import datetime

import torch
from torch import nn

import json

# from torch.backends import cudnn
# cudnn.benchmark = False
# cudnn.deterministic = True

from ExP import ExP

config = {
    'res_path': './results/sub_result.txt',
    'sub_res_path': "./results/sub%d/log.txt",
    
    # train mode
    'mode': 'BCIC', # 'BCIC' / 'LYH' 
    'train_prop': 0.8,
    'pretrained': False,
    'pretrained_pth': './results_lyh/results_lyh_4cls_in',
    'use_center_loss': False,
    
    # model config
    'deformable': False,
    'emb_size': 40,
    'proj_size': 20,
    'encoder_depth': 6,
    'decoder_depth': 2,
    'n_classes': 4,
    'channel': 22,
    'seq_len': 1000,
    'num_queries': 6,
    
    # EEGMamba
    'mamba': True,
    'mamba_depth': 3,
    'block_depth': 3,
    'conv_channel': 20,
    
    'encoder_config': {
            'num_heads': 8,
            'drop_p': 0.5,
            'forward_expansion': 4,
            'forward_drop_p': 0.5,
            'num_of_points': 16
        },
    
    'decoder_config': {
            'num_heads': 4,
            'drop_p': 0.5,
            'forward_expansion': 4,
            'forward_drop_p': 0.5,
            'num_of_points': 24
        },
    
    'hidden_size_1': 256,
    'hidden_size_2': 64,
    'drop_p_1': 0.5,
    'drop_p_2': 0.3,
    
    # training config (adam)
    'batch_size': 72,
    'n_epochs': 500,
    'lr': 0.002,
    'b1': 0.9,
    'b2': 0.999,
    'Lambda': 0.0005
}


def main_lyh():
    res_path = config["res_path"]
    dir_name = os.path.dirname(res_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    result_write = open(res_path, "w")
    result_write.write('config: \n')
    result_write.write(json.dumps(config, indent=4))
    result_write.write("\n\n")
    print(config)
    
    starttime = datetime.datetime.now()
    seed_n = np.random.randint(2021)
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    print('LYH data train & test')
    exp = ExP(0, config)

    bestAcc, averAcc, cm, best_acc_ep = exp.train()
    print('THE BEST ACCURACY IS ' + str(bestAcc))
    result_write.write('Seed is: ' + str(seed_n) + "\n")
    result_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
    result_write.write('The average accuracy is: ' + str(averAcc) + "\n")
    result_write.write('Best accuracy appears in: ' + str(best_acc_ep) + " epoch.\n")

    endtime = datetime.datetime.now()
    result_write.write('Duration: ' + str(endtime - starttime) + "\n")
    print('Duration: ' + str(endtime - starttime))

    result_write.write('\nconfusion_matirx:\n')
    result_write.write(str(cm))
    result_write.close()


def main():
    best = 0
    aver = 0
    
    res_path = config["res_path"]
    dir_name = os.path.dirname(res_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    result_write = open(res_path, "w")
    result_write.write('config: \n')
    result_write.write(json.dumps(config, indent=4))
    result_write.write("\n\n")
    print(config)
    
    for i in range(9):
        starttime = datetime.datetime.now()


        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1, config)

        bestAcc, averAcc, cm, best_acc_ep = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Best accuracy appears in: ' + str(best_acc_ep) + " epoch.\n")

        endtime = datetime.datetime.now()
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Duration: ' + str(endtime - starttime) + "\n")
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        
        result_write.write('\nconfusion_matirx:\n')
        result_write.write(str(cm))
        result_write.write("\n\n*********************************************\n\n")
        


    best = best / 9
    aver = aver / 9

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    if config["mode"] == 'BCIC':
        main()
    elif config["mode"] == 'LYH':
        main_lyh()
          
    print(time.asctime(time.localtime(time.time())))
