"""
classes in this file:
+ PatchEmbedding (CNN)
+ MultiheadAttention
+ ResidualAdd
+ FeedForwardBlock
# + GELU
+ TransformerEncoderBlock
+ TransformerEncoder
+ ClassificationHead
+ ClassificationHead2 (to do classification job with decoder object queries)
+ DeformableCrossAttention
+ TransformerDecoderBlock
+ TransformerDecoder
+ Conformer

Structure:

Conformer:
    PatchEmbedding
    TransformerEncoder
        TransformerEncoderBlock
            ResidualAdd
            MultiheadAttention
            FeedForwardBlock
    ClassificationHead


Deformable Conformer:
    PatchEmbedding
    TransformerEncoder
        TransformerEncoderBlock
            ResidualAdd
            MultiheadAttention
            FeedForwardBlock
    TransformerDecoder
        TransformerDecoderBlock
            ResidualAdd
            DeformableCrossAttention
                MultiheadAttention
            FeedForwardBlock            
    ClassificationHead2
"""

import os
import numpy as np
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.autograd as autograd

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt

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


# class GELU(nn.Module):
#     def forward(self, input: Tensor) -> Tensor:
#         return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


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
    def __init__(self, depth, emb_size, config=None):
        if config != None:
            super().__init__(*[TransformerEncoderBlock(emb_size, **config["encoder_config"]) for _ in range(depth)])
        else:
            super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=40, n_classes=4, config=None):
        super().__init__()
        
        hidden_size_1 = 256
        hidden_size_2 = 32
        drop_p_1 = 0.5
        drop_p_2 = 0.3
        if config != None:
            hidden_size_1 = config["hidden_size_1"]
            hidden_size_2 = config["hidden_size_2"]
            drop_p_1 = config["drop_p_1"]
            drop_p_2 = config["drop_p_2"]
            
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            # 2440 = 61*40
            # another hard code potential bug here
            nn.Linear(2440, hidden_size_1),
            nn.ELU(),
            nn.Dropout(drop_p_1),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ELU(),
            nn.Dropout(drop_p_2),
            nn.Linear(hidden_size_2, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        # return x, out
        return out


class ClassificationHead2(nn.Module):
    def __init__(self, emb_size=40, n_classes=4, config=None):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes

        hidden_size_1 = 256
        hidden_size_2 = 32
        drop_p_1 = 0.5
        drop_p_2 = 0.3
        if config != None:
            hidden_size_1 = config["hidden_size_1"]
            hidden_size_2 = config["hidden_size_2"]
            drop_p_1 = config["drop_p_1"]
            drop_p_2 = config["drop_p_2"]
            
        self.classification_mlps = nn.ModuleList()
        for _ in range(n_classes):
            mlp = nn.Sequential(
                nn.Linear(emb_size, hidden_size_1),
                nn.ELU(),
                nn.Dropout(drop_p_1),
                nn.Linear(hidden_size_1, hidden_size_2),
                nn.ELU(),
                nn.Dropout(drop_p_2),
                nn.Linear(hidden_size_2, n_classes)
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
    def __init__(self, num_heads, emb_size, drop_p=0.5, num_of_points=10):
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
                drop_p=0.5, 
                forward_expansion=4, 
                forward_drop_p=0.5,
                num_of_points=10):
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
        self.deform_cross_att = DeformableCrossAttention(num_heads, emb_size, drop_p, num_of_points)
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
    def __init__(self, depth, n_classes=4, emb_size=40, config=None):
        super().__init__()
        self.depth = depth
        if config != None:
            self.decoder_blocks = \
                [TransformerDecoderBlock(emb_size, **config["decoder_config"]).cuda() for _ in range(depth)]
        else:
            self.decoder_blocks = [TransformerDecoderBlock(emb_size).cuda() for _ in range(depth)]
        # try linearly project feature to object queries
        # hard code 61 here: a potential bug
        # self.obj_query_proj = nn.Linear(61, n_classes)
        
        # randomly initialize query
        self.obj_query = nn.Parameter(torch.randn(n_classes, emb_size)).cuda()

        '''
        note: two ways of query initialization hardly influence performance.
        '''
        
    def forward(self, input):
        '''
        input: (bs, n, emb)
        '''
        bs, n, emb = input.shape
        
        # batch_query = self.obj_query_proj(input.permute(0,2,1)).permute(0,2,1)
        batch_query = self.obj_query.unsqueeze(0).repeat(bs, 1, 1) #(bs, n_cls, e)
        for i in range(self.depth):
            batch_query = self.decoder_blocks[i](input, batch_query)
        return batch_query


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, 
            encoder_depth=6, 
            decoder_depth=3,
            n_classes=4, 
            config=None):
        '''
        input:
            emb_size: k the num of temporal conv filters
            depth: num of transformer encoder blocks
            n_class: output num of last fully-connected layer
        '''
        if config != None:
            emb_size = config["emb_size"]
            encoder_depth = config["encoder_depth"]
            decoder_depth = config["decoder_depth"]
            n_classes = config["n_classes"]
        
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(encoder_depth, emb_size, config),
            # ClassificationHead(emb_size, n_classes)
            TransformerDecoder(decoder_depth, n_classes, emb_size, config),
            ClassificationHead2(emb_size, n_classes)
        )
