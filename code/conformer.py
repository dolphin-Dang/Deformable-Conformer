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
    'sub_res_path': "./results/log_subject%d.txt",
    
    # model config
    'emb_size': 40,
    'encoder_depth': 6,
    'decoder_depth': 3,
    'n_classes': 4,
    
    'encoder_config': {
            'num_heads': 10,
            'drop_p': 0.5,
            'forward_expansion': 4,
            'forward_drop_p': 0.5
        },
    
    'decoder_config': {
            'num_heads': 10,
            'drop_p': 0.5,
            'forward_expansion': 4,
            'forward_drop_p': 0.5,
            'num_of_points': 10
        },
    
    'hidden_size_1': 256,
    'hidden_size_2': 32,
    'drop_p_1': 0.5,
    'drop_p_2': 0.3,
    
    # training config
    'lr': 0.002,
    'b1': 0.5,
    'b2': 0.999
}



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
        exp = ExP(i + 1)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        result_write.write('subject %d duration: '%(i+1) + str(endtime - starttime) + "\n")
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
