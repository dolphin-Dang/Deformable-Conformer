"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()
        self.emb_size = emb_size

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)), # shape: 1000 -> 976 (1000-25+1)
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # shape: 976 -> 61 ((976-75)/15+1)
            nn.Dropout(0.5),
        )

        # transpose, conv could enhance fiting ability slightly
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))
        # Rearrange('b e (h) (w) -> b (h w) e'),



    def forward(self, x: Tensor) -> Tensor:
        bs, _, _, _ = x.shape
        x = self.shallownet(x)

        # 1*1 conv
        x = self.projection(x)
        
        # reshape: (bs, embd, a, b) -> (bs, a*b, embd)
        x = x.permute(0,2,3,1).reshape(bs, -1, self.emb_size)
        # print(x.shape)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None, query: Tensor = None) -> Tensor:
        if query != None:
            queries = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads)
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class ClassificationHead2(nn.Module):
    def __init__(self, emb_size=40, n_classes=4):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes

        self.classification_mlps = nn.ModuleList()
        for _ in range(n_classes):
            mlp = nn.Sequential(
                nn.Linear(emb_size, 256),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 32),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(32, 4)                
            )
            self.classification_mlps.append(mlp)

    def forward(self, input):
        '''
        Input: (batch_size, n_classes, emb_size)
        '''    

        xs = torch.chunk(input, chunks=self.n_classes, dim=1)
        outputs = []
        for i, mlp in enumerate(self.classification_mlps):
            output = mlp(xs[i].squeeze(dim=1))
            output_max, _ = torch.max(output, dim=1, keepdim=True)
            outputs.append(output_max)
        output = torch.cat(outputs, dim=1)
        # return input, output
        return output


class DeformableCrossAttention(nn.Module):
    def __init__(self, num_heads, emb_size, drop_p=0.3, num_of_points=10):
        '''
        query: (bs, n_classes, emb_size)

        Use nn.Linear to get the reference points and weights.
        '''
        super().__init__()
        self.drop_p = drop_p
        self.num_of_points = num_of_points

        self.fc_pts = nn.Linear(emb_size, num_of_points)
        self.fc_w = nn.Linear(emb_size, num_of_points)

        self.att = MultiHeadAttention(emb_size, num_heads, drop_p)
        

    def forward(self, input, query):
        # print("*** DCA forward ***")
        bs, n, e = input.shape
        ref_pts_idx = self.fc_pts(query) # (bs, n_classes, num_of_points) point offset
        ref_pts_idx = torch.floor(
            torch.sigmoid(ref_pts_idx) * n
        ).long() # int [0, n-1]

        ref_weight = self.fc_w(query) # (bs, n_classes, num_of_points) point weight
        ref_weight = F.softmax(ref_weight, -1) # float [0,1]

        indices_tuple = ref_pts_idx.split(1, dim=1)
        indices_lists = [t.squeeze() for t in indices_tuple] # list of idx tensor (bs, num_of_points)

        weight_lists = ref_weight.split(1, dim=1)
        weight_lists = [t.squeeze() for t in weight_lists] # list of tensor

        # list of (bs, num_of_points, e) tensors with weight multiplied
        deform_tensors = []
        for i in range(len(indices_lists)):
            index = indices_lists[i].unsqueeze(-1).repeat(1,1,e)
            tmp_t = input.gather(1, index) # (bs, num_of_points, e)
            weights_tensor = weight_lists[i].unsqueeze(-1).repeat(1,1,e) # (bs, num_of_points, e)
            # print(weight_lists[i].shape)
            # print(tmp_t.shape)
            # print(weights_tensor.shape)
            deform_tensors.append(tmp_t * weights_tensor)

        att_ans_list = []
        for t in deform_tensors:
            att = self.att(x=t, mask=None, query=query) # (bs, num_of_points, e)
            att = torch.sum(att, dim=1) # (bs, e)
            att_ans_list.append(att)
        
        ans = torch.stack(att_ans_list, dim=1)
        return ans

class TransformerDecoderBlock(nn.Module):
    def __init__(self, emb_size, 
                num_heads=10, 
                n_classes=4, 
                drop_p=0.5, 
                forward_expansion=4, 
                forward_drop_p=0.5):
        '''
        n_classes == num of object queries
        '''
        super().__init__()
        self.p1 = ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p)
                ))
        # self.p2 = 
        self.ln = nn.LayerNorm(emb_size)
        self.deform_cross_att = DeformableCrossAttention(num_heads, emb_size, drop_p)
        self.dropout = nn.Dropout(drop_p)
        
        self.p3 = ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                ))

    def forward(self, feature, query):
        '''
        feature, query: (bs, n, emb)
        '''
        query = self.p1(query)
        query = self.ln(query)
        att = self.dropout(self.deform_cross_att(feature, query))
        att = self.p3(att)
        return att

class TransformerDecoder(nn.Module):
    def __init__(self, depth, n_classes=4, emb_size=40):
        super().__init__()
        self.depth = depth
        self.decoder_blocks = [TransformerDecoderBlock(emb_size, n_classes=n_classes).cuda() for _ in range(depth)]
        # TODO: try linearly project feature to object queries
        self.obj_query = nn.Parameter(torch.randn(n_classes, emb_size)).cuda()

    def forward(self, input):
        '''
        input: (bs, n, emb)
        self.query: (n, emb)
        '''
        bs, n, emb = input.shape
        batch_query = self.obj_query.unsqueeze(0).repeat(bs, 1, 1) #(bs, n, e)
        for i in range(self.depth):
            batch_query = self.decoder_blocks[i](input, batch_query)
        return batch_query


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, 
            encoder_depth=6, 
            decoder_depth=6,
            n_classes=4, **kwargs):
        '''
        input:
            emb_size: k the num of temporal conv filters
            depth: num of transformer encoder blocks
            n_class: output num of last fully-connected layer
        '''
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(encoder_depth, emb_size),
            # ClassificationHead(emb_size, n_classes)
            TransformerDecoder(decoder_depth, n_classes),
            ClassificationHead2(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 500
        self.c_dim = 4
        self.lr = 0.002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        # self.root = '/Data/strict_TE/'
        self.root = './data/rawMat/'

        
        res_path = "./results/log_subject%d.txt" % self.nSub
        dir_name = os.path.dirname(res_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        result_write = open(res_path, "w")
        self.log_write = open(res_path, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        summary(self.model, (1, 22, 1000))

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        # print("In interaug.")
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            # cls_idx = np.where(label == cls4aug + 1)
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_data = tmp_data.reshape(tmp_data.shape[0], -1, *tmp_data.shape[-2:])
            tmp_label = label[cls_idx]
            # print(timg.shape) # (288, 1, 22, 1000)
            # print(tmp_data.shape) # (72, 1, 22, 1000)
            # print(label.shape) # (288, 1)
            # print(tmp_label.shape) # (72,)

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            # print(f"tmp_aug_data.shape = {tmp_aug_data.shape}")
            # drf: get 8 slices of 8 random data from tmp_data, from the same time period idx
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    # print(f"ri, rj: {ri}, {rj}")
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        # aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_test_data(self):
        test_tmp = scipy.io.loadmat(self.root + 'se00%d.mat' % self.nSub)
        test_data = test_tmp['x']
        test_label = test_tmp['y']

        test_data = np.transpose(test_data, (2, 0, 1))
        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)

        testData = test_data
        testLabel = test_label.squeeze()

        return testData, testLabel
    
    def get_source_data(self):

        # train data
        # self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        # self.train_data = self.total_data['data']
        # self.train_label = self.total_data['label']

        self.total_data = scipy.io.loadmat(self.root + 's00%d.mat' % self.nSub)
        self.train_data = self.total_data['x'] # (22, 1000, 288)
        self.train_label = self.total_data['y'] # (1, 288)

        self.train_data = np.transpose(self.train_data, (2, 0, 1)) # (288, 22, 1000)
        self.train_data = np.expand_dims(self.train_data, axis=1) # (288, 1, 22, 1000)
        self.train_label = np.transpose(self.train_label) # (288, 1)
        
        self.allData = self.train_data
        self.allLabel = self.train_label.squeeze()

        shuffle_num = np.random.permutation(len(self.allData))
        # print(f"Shuffle num {shuffle_num}")
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        # self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        # self.test_data = self.test_tmp['data']
        # self.test_label = self.test_tmp['label']

        self.test_tmp = scipy.io.loadmat(self.root + 'se00%d.mat' % self.nSub)
        self.test_data = self.test_tmp['x']
        self.test_label = self.test_tmp['y']

        self.test_data = np.transpose(self.test_data, (2, 0, 1))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label.squeeze()

        # standardize
        # target_mean = np.mean(self.allData)
        # target_std = np.std(self.allData)
        # self.allData = (self.allData - target_mean) / target_std
        # self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        # print(self.allData.shape) # (288, 1, 22, 1000)
        # print(self.allLabel.shape) # (288, 1)
        # print(self.testData.shape)
        # print(self.testLabel.shape)
        # print(self.testLabel)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        # label = torch.from_numpy(label - 1)
        # print(img.shape)
        # print(label.shape)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        # test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                # print(aug_data.shape)
                # print(aug_label.shape)
                # print(self.allData.shape)
                # print(self.allLabel.shape)
                
                # print(label.shape)
                # print(aug_label.shape)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                
                outputs = self.model(img)
                # tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # out_epoch = time.time()

            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Cls = self.model(test_data)


                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred


        torch.save(self.model.module.state_dict(), './result_models/model_sub%d.pth'%self.nSub)
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():
    best = 0
    aver = 0
    
    res_path = "./results/sub_result.txt"
    dir_name = os.path.dirname(res_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    result_write = open(res_path, "w")


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
        result_write.write('subject %d duration: '%(i+1) + str(endtime - starttime))
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
