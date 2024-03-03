"""
Author: dolphin-Dang

In this file we provide a class 'ModelRunner' to run the pretrained Deformable-Conformer model.
TO MAKE THINGS CLEAR: 
    If you don't want to change model structure,
    user only need to focus on ModelRunner class, and ignore other classes.

try:
    'from DeformableConformer import ModelRunner' to import the class.

ModelRunner provide two useful APIs:
    finetune: 
        Do short fine-tuning for cross-session tasks.
        Choices: (in config dictionary)
            Cover original .pth file every time or not.
            Use test set or not. 
                Without a test set, the ModelRunner will save the best training accuracy model.
                With a test set, the ModelRunner will save the best test accuracy model.
    inference: 
        Use preload model to do inference task.

You may want to check config dictionary for changing/learning some settings.

classes in this file:
    + PatchEmbedding (CNN)
    + MultiheadAttention
    + ResidualAdd
    + FeedForwardBlock
    + TransformerEncoderBlock
    + TransformerEncoder
    + ClassificationHead
    + DeformableCrossAttention
    + TransformerDecoderBlock
    + TransformerDecoder
    + DeformableConformer
    + ModelRunner



Deformable Conformer Structure:
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
    ClassificationHead
"""

import os
import numpy as np
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# important
config = {
    # model param
    'pretrained_pth': './DeformableConformer.pth',
    'cover_old_param': False, # default not to cover old param
    'new_file_name': './FineTunedParam.pth',

    # training config (adam)
    'batch_size': 30,
    'n_epochs': 100,
    'use_test_set': False, # do you use test set?
    'test_prop': 0.2, # how much data to test?
    'lr': 0.0002,
    'b1': 0.9,
    'b2': 0.999,


    # model config: no need to focus on
    'emb_size': 40,
    'proj_size': 40,
    'encoder_depth': 4,
    'decoder_depth': 2,
    'n_classes': 3,
    'channel': 14,
    'seq_len': 250,
    'num_queries': 3,

    'encoder_config': {
            'num_heads': 8,
            'drop_p': 0.5,
            'forward_expansion': 4,
            'forward_drop_p': 0.5,
            'num_of_points': 16
        },

    'decoder_config': {
            'num_heads': 8,
            'drop_p': 0.5,
            'forward_expansion': 4,
            'forward_drop_p': 0.5,
            'num_of_points': 24
        },

    'hidden_size_1': 256,
    'hidden_size_2': 64,
    'drop_p_1': 0.5,
    'drop_p_2': 0.3,
}


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, channel=22):
        # self.patch_size = patch_size
        super().__init__()
        self.emb_size = emb_size

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)), # shape: 1000 -> 976 (1000-25+1)
            nn.Conv2d(40, 40, (channel, 1), (1, 1)),
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
        assert self.emb_size % self.num_heads == 0, "Invalid head number!"

    def forward(self, x: Tensor, mask: Tensor = None, query: Tensor = None) -> Tensor:
        # print("*** MHA forward ***")
        if query != None:
            queries = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads)
        else:
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
        # print(att.shape)
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


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5,
                 num_of_points=8):
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

class ClassificationHead(nn.Module):
    def __init__(self, emb_size=40, n_classes=4, config=None):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes

        hidden_size_1 = 256
        hidden_size_2 = 64
        drop_p_1 = 0.5
        drop_p_2 = 0.3
        num_q = n_classes
        if config != None:
            hidden_size_1 = config["hidden_size_1"]
            hidden_size_2 = config["hidden_size_2"]
            drop_p_1 = config["drop_p_1"]
            drop_p_2 = config["drop_p_2"]
            num_q = config["num_queries"]
        
        self.classification_proj = nn.Sequential(
                nn.Linear(emb_size*num_q, hidden_size_1),
                nn.ELU(),
                nn.Dropout(drop_p_1),
                nn.Linear(hidden_size_1, hidden_size_2),
                nn.ELU(),
                nn.Dropout(drop_p_2),
                nn.Linear(hidden_size_2, 4)
            )

    def forward(self, input):
        '''
        Input: (batch_size, num_q, emb_size)
        '''
        output = self.classification_proj(input.reshape(input.size(0),-1))
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
        query_list = query.split(1, dim=1) # list of (bs, 1, emb)
        indices_list = []
        for t in query_list:
            idx = self.fc_pts(t).squeeze() # (bs, num_of_pts)
            idx = torch.floor(
                torch.sigmoid(idx) * n
            ).long() # int [0, n-1]
            indices_list.append(idx)
            
        weight_list = []
        for t in query_list:
            ref_w = self.fc_w(t).squeeze() # (bs, num_of_points)
            ref_w = F.softmax(ref_w, -1) # float [0,1]
            weight_list.append(ref_w)

        # list of (bs, num_of_points, e) tensors with weight multiplied
        deform_tensors = []
        for i in range(len(indices_list)):
            index = indices_list[i].unsqueeze(-1).repeat(1,1,e)
            tmp_t = input.gather(1, index) # (bs, num_of_points, e)
            weights_tensor = weight_list[i].unsqueeze(-1).repeat(1,1,e) # (bs, num_of_points, e)
            deform_tensors.append(tmp_t * weights_tensor)

        att_ans_list = []
        for i, t in enumerate(deform_tensors):
            att = self.att(x=t, mask=None, query=query_list[i]) # (bs, num_of_points, e)
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
        feature: (bs, n, emb)
        query: (bs, n_classes, emb)
        '''
        
        query = self.p1(query)
        query = self.ln(query)
        att = self.deform_cross_att(feature, query)
        att = self.dropout(att)
        att = self.p3(att)
        return att

class TransformerDecoder(nn.Module):
    def __init__(self, depth, n_classes=4, emb_size=40, config=None):
        super().__init__()
        self.depth = depth
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth):
            self.decoder_blocks.append(TransformerDecoderBlock(emb_size, **config["decoder_config"]))
        
        # randomly initialize query
        num_queries = n_classes
        if config!=None:
            num_queries = config["num_queries"]
        self.obj_query = nn.Parameter(torch.randn(num_queries, emb_size))

        '''
        note: two ways of query initialization hardly influence performance.
        '''
        
    def forward(self, input):
        '''
        input: (bs, n, emb)
        '''
        bs, n, emb = input.shape
        batch_query = self.obj_query.unsqueeze(0).repeat(bs, 1, 1) #(bs, n_cls, e)
        for i in range(self.depth):
            batch_query = self.decoder_blocks[i](input, batch_query)
        return batch_query
    

class DeformableConformer(nn.Module):
    def __init__(self, emb_size=40, 
            encoder_depth=6, 
            decoder_depth=3,
            n_classes=4, 
            config=None):
        
        super().__init__()
        channel = 22
        proj_size = 40
        if config != None:
            emb_size = config["emb_size"]
            encoder_depth = config["encoder_depth"]
            decoder_depth = config["decoder_depth"]
            n_classes = config["n_classes"]
            channel = config["channel"]
            proj_size = config["proj_size"]
            
        self.patch_embedding = PatchEmbedding(emb_size, channel)
        self.encoder = TransformerEncoder(encoder_depth, emb_size, config)
        self.decoder = TransformerDecoder(decoder_depth, n_classes, proj_size, config)
        self.classifier = ClassificationHead(proj_size, n_classes, config)
       
    def forward(self, input):
        emb = self.patch_embedding(input)
        feat = self.encoder(emb)
        queries = self.decoder(feat)
        ans = self.classifier(queries)
        return ans
    
class ModelRunner():
    def __init__(self):
        super().__init__()
        self.config = config
        self.start_epoch = 0
        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.b1 = config["b1"]
        self.b2 = config["b2"]
           
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.model = DeformableConformer(config=self.config).cuda()

        # load model first
        print(f"Loading pre-trained parameters: {config['pretrained_pth']}.")
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(config["pretrained_pth"])
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        
        not_loaded_keys = [k for k in model_dict.keys() if k not in pretrained_dict.keys()]
        if len(not_loaded_keys) == 0:
            print("All parameters loaded successfully.")
        else:
            print("The following parameters were not loaded:")
            for key in not_loaded_keys:
                print(key)
            
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
            
    def finetune(self, input):
        '''
        Input:
            input: a tuple (data, label)
            where,
                data: ndarray of shape (bs, 14, 250)
                label: ndarray of shape (bs,)
        Output:
            None.
        Side Effect:
            Change file 'DeformableConformer.pth' to a fine-tuned model parameter file.
        '''
        data, label = input
        bs, ch, seq = data.shape
        assert(ch == 14 and seq == 250)
        assert(label.shape == (bs,))

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        
        shuffle_num = np.random.permutation(len(label))
        data = data[shuffle_num, :, :]
        label = label[shuffle_num]
        
        if self.config["use_test_set"]:
            tot_data = len(label)
            test_data = data[int(tot_data*(1-self.config["test_prop"])):,:,:]
            test_label = label[int(tot_data*(1-self.config["test_prop"])):]
            data = data[:int(tot_data*(1-self.config["test_prop"])),:,:]
            label = label[:int(tot_data*(1-self.config["test_prop"]))]
            
            # test data
            test_data = Variable(test_data.type(self.Tensor))
            test_label = Variable(test_label.type(self.LongTensor))
            
        # fine-tune data
        dataset = torch.utils.data.TensorDataset(data, label)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True
        )

        best_acc = 0
        best_model_dict = 0
        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                
                outputs = self.model(img.unsqueeze(1))
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
            
            if self.config["use_test_set"]:
                self.model.eval()
                with torch.no_grad():
                    outputs_test = self.model(test_data.unsqueeze(1))

                loss_test = self.criterion_cls(outputs_test, test_label)
                
                y_pred = torch.max(outputs_test, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                
                if acc > best_acc:
                    best_acc = train_acc
                    best_model_dict = self.model.state_dict()
                    
                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)
            else:
                if train_acc > best_acc:
                    best_acc = train_acc
                    best_model_dict = self.model.state_dict()
                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc)
            
        # save model param
        if self.config["cover_old_param"]:
            torch.save(best_model_dict, './DeformableConformer.pth')
        else:
            torch.save(best_model_dict, self.config["new_file_name"])

        
    def inference(self, input):
        '''
        Input:
            input: ndarray of shape (1, 14, 250)
        Output:
            Predicted label: int.
        '''
        assert(input.shape == (1,14,250))
        input = torch.from_numpy(input)
        img = Variable(input.cuda().type(self.Tensor))
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(img.unsqueeze(1))
        y_pred = torch.max(output, 1)[1]
        return int(y_pred)
        
        
def main():
    left_raw2 = np.load('../../data/lyh_data/left_processed_v2(300).npy') # label: 0
    right_raw2 = np.load('../../data/lyh_data/right_processed_v2(300).npy') # label: 1
    leg_raw2 = np.load('../../data/lyh_data/leg_processed_v2(300).npy') # label: 2
    nothing_raw2 = np.load('../../data/lyh_data/nothing_processed_v2(300).npy') # label: 3
    eeg_raw2 = [left_raw2, right_raw2, leg_raw2, nothing_raw2]
    eeg_raw2 = [t[:14,:] for t in eeg_raw2]
    
    X = []
    y = []

    for i in range(config["n_classes"]):
        split_data = np.split(eeg_raw2[i], 1200, axis=1)
        X_raw = np.stack(split_data, axis=0) # (14, 30_0000) => (1200, 14, 250)
        X_raw = np.expand_dims(X_raw, axis=1) # (1200, 1, 14, 250)
        y_raw = np.array([i for j in range(1200)]) # (1200,) value = label
        X.append(X_raw)
        y.append(y_raw)

    X = np.concatenate(X)
    y = np.concatenate(y)

    shuffle_num = np.random.permutation(len(X))
    X = X[shuffle_num, :, :, :]
    y = y[shuffle_num]

    num = 100
    X_train = X[:num,:,:,:]
    y_train = y[:num]
    X_train = X_train.squeeze()

    print(X_train.shape)
    print(y_train.shape)
    
    
    model_runner = ModelRunner()
    # finetune test
    model_runner.finetune((X_train, y_train))
    
    # inference test
    correct = 0
    for i in range(20):
        output = model_runner.inference(X[1000+i])
        correct += 1 if output==y[1000+i] else 0
    print(correct)
        
    
if __name__ == '__main__':
    main()