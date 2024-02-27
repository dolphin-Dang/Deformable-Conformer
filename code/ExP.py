'''
ExP class.
The training logic.
'''


import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from models import DeformableConformer
from torch.backends import cudnn
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from CenterLoss import CenterLoss

import matplotlib.pyplot as plt

import scipy.io
import sys

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # arrange GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus)) # choose GPUs

class ExP():
    def __init__(self, nsub, config=None):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 500
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.channel = 22
        self.mode = "BCIC"
        self.n_classes = 4
        self.Lambda = 0.0005
        self.use_center_loss = False
        
        self.start_epoch = 0
        # self.root = '/Data/strict_TE/'
        self.root = './data/rawMat/'
        self.res_path = "./results/sub%d/log.txt" % self.nSub
        
        self.config = None
        self.config = config
        if self.config != None:
            self.batch_size = config["batch_size"]
            self.n_epochs = config["n_epochs"]
            self.lr = config["lr"]
            self.b1 = config["b1"]
            self.b2 = config["b2"]
            self.res_path = config["sub_res_path"] % self.nSub
            self.channel = config["channel"]
            self.mode = config["mode"]
            self.n_classes = config["n_classes"]
            self.Lambda = config["Lambda"]
            self.use_center_loss = config["use_center_loss"]
        
        dir_name = os.path.dirname(self.res_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        result_write = open(self.res_path, "w")
        self.log_write = open(self.res_path, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.center_loss = CenterLoss(num_classes=self.n_classes, feat_dim=self.n_classes*config["proj_size"], use_gpu=True)
        # self.center_loss = CenterLoss(num_classes=self.n_classes, feat_dim=61*40, use_gpu=True)
        
        self.model = DeformableConformer(config=self.config).cuda()

        # two-stage training
        if config != None and config["pretrained"] == True:
            print(f"Loading pre-trained parameters: {config['pretrained_pth']}.")
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(os.path.join(config["pretrained_pth"], 'sub%d/model.pth'%self.nSub))
            pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            # print(model_dict.keys())
            
            # freeze pretrained parameters
            # for param_name, param in self.model.named_parameters():
            #     if param_name in pretrained_dict:
            #         param.requires_grad = False
        
        
        if self.config:
            summary(self.model, (1, config['channel'], 1000))
        else:
            summary(self.model, (1, 22, 1000))
        
#         with open('summary.txt', 'w') as f:
#             original_stdout = sys.stdout 
#             sys.stdout = f

#             summary(self.model, (1, 22, 1000))
#             sys.stdout = original_stdout

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

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, self.channel, 1000))
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
    
    def get_lyh_data(self):
        print("Getting LYH data set.")
        
        # get all the data first
        cur_path = os.getcwd()
        print(cur_path)
        
        # train data
        left_raw = np.load('./data/lyh_data/left_processed_v3(500).npy') # label: 0
        right_raw = np.load('./data/lyh_data/right_processed_v3(500).npy') # label: 1
        leg_raw = np.load('./data/lyh_data/leg_processed_v3(500).npy') # label: 2
        nothing_raw = np.load('./data/lyh_data/nothing_processed_v3(500).npy') # label: 3
        eeg_raw = [left_raw, right_raw, leg_raw, nothing_raw]
        eeg_raw = [t[:14,:] for t in eeg_raw]
        
        # test data
        left_raw_test = np.load('./data/lyh_data/left_processed_v2(300).npy') # label: 0
        right_raw_test = np.load('./data/lyh_data/right_processed_v2(300).npy') # label: 1
        leg_raw_test = np.load('./data/lyh_data/leg_processed_v2(300).npy') # label: 2
        nothing_raw_test = np.load('./data/lyh_data/nothing_processed_v2(300).npy') # label: 3
        eeg_raw_test = [left_raw_test, right_raw_test, leg_raw_test, nothing_raw_test]
        eeg_raw_test = [t[:14,:] for t in eeg_raw_test]
        
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        
        for i in range(self.config["n_classes"]):
            # print(f"Data shape: {eeg_raw[i].shape}")
            # cross-session
#             split_data = np.split(eeg_raw[i], 500, axis=1)
#             X_raw = np.stack(split_data, axis=0) # (14, 50_0000) => (500, 14, 1000)
#             X_raw = np.expand_dims(X_raw, axis=1) # (500, 1, 14, 1000)
#             y_raw = np.array([i for j in range(500)]) # (500,) value = label
#             X_train.append(X_raw)
#             y_train.append(y_raw)
            
#             split_data_test = np.split(eeg_raw_test[i], 300, axis=1)
#             X_raw_test = np.stack(split_data_test, axis=0) # (14, 30_0000) => (300, 14, 1000)
#             X_raw_test = np.expand_dims(X_raw_test, axis=1) # (300, 1, 14, 1000)
#             y_raw_test = np.array([i for j in range(300)]) # (300,) value = label
#             X_test.append(X_raw_test)
#             y_test.append(y_raw_test)

            # in-session
            split_data = np.split(eeg_raw[i], 500, axis=1)
            X_raw = np.stack(split_data, axis=0) # (14, 50_0000) => (500, 14, 1000)
            X_raw = np.expand_dims(X_raw, axis=1) # (500, 1, 14, 1000)
            y_raw = np.array([i for j in range(500)]) # (500,) value = label
            train_prop = int(self.config["train_prop"]*len(X_raw))
            X_train.append(X_raw[:train_prop,:,:,:])
            X_test.append(X_raw[train_prop:,:,:,:])
            y_train.append(y_raw[:train_prop])
            y_test.append(y_raw[train_prop:])
 
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)

        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
    
        shuffle_num = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_num, :, :, :]
        y_train = y_train[shuffle_num]
        
        self.train_data = X_train
        self.train_label = y_train
        self.allData = np.concatenate([X_train, X_test])
        self.allLabel = np.concatenate([y_train, y_test])
            
        return X_train, y_train, X_test, y_test
    
    def get_source_data(self):
        print("Getting BCIC data set.")
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
        
        # smaller time span
        # self.train_data = self.train_data.reshape((288*4, 1, 22, 250))
        # self.train_label = np.repeat(self.train_label, 4, axis=0)
        
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
        
        # smaller time span
        # self.test_data = self.test_data.reshape((288*4, 1, 22, 250))
        # self.test_label = np.repeat(self.test_label, 4, axis=0)

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

        # some trackable history
        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []
        
        if self.mode == 'BCIC':
            img, label, test_data, test_label = self.get_source_data()
        elif self.mode == "LYH":
            img, label, test_data, test_label = self.get_lyh_data()

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
        # if self.use_center_loss:
        #     self.optimizer_feat = torch.optim.Adam(
        #         list(self.model.patch_embedding.parameters())+
        #             list(self.model.encoder.parameters())+
        #                 list(self.model.decoder.parameters()), 
        #         lr=self.lr, betas=(self.b1, self.b2))
        #     self.optimizer_cls = torch.optim.Adam(
        #         list(self.model.classifier.parameters()), 
        #         lr=self.lr, betas=(self.b1, self.b2))
        # else:
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
        best_acc_ep = 0
        
        def lambda_decend(Lambda, total_ep, ep):
            if ep > total_ep / 2:
                return 0
            else:
                return (1 - ep*2/total_ep) * Lambda
        
        
        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                
                # print(img.shape)
                # print(label.shape)
                
                if self.mode == 'BCIC':
                    # data augmentation
                    aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                    img = torch.cat((img, aug_data))
                    label = torch.cat((label, aug_label))
                
                # outputs = self.model(img)
                tok, outputs = self.model(img)
                
                loss = self.criterion_cls(outputs, label) 
                if self.use_center_loss:
                    center_loss = self.center_loss(tok.reshape(label.size(0), -1), label) * self.Lambda
                    loss = loss + center_loss

                    # for param in self.model.parameters():
                    #     print(param.grad)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

#                 loss = self.criterion_cls(outputs, label)
#                 if self.use_center_loss:
#                     center_loss = self.center_loss(tok.reshape(label.size(0), -1), label) * self.Lambda
#                     self.optimizer_cls.zero_grad()
#                     loss.backward(retain_graph=True)
#                     self.optimizer_feat.zero_grad()
#                     center_loss.backward()
                    
#                     self.optimizer_cls.step()
#                     self.optimizer_feat.step()
#                 else:
#                     self.optimizer.zero_grad()
#                     center_loss.backward()
#                     self.optimizer.step()
            # out_epoch = time.time()

            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)
                # Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                if self.use_center_loss:
                    center_loss = self.center_loss(Tok.reshape(test_label.size(0), -1), test_label)
                    loss_test = loss_test + self.Lambda * center_loss
                
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                train_acc_list.append(train_acc)
                test_acc_list.append(acc)
                train_loss_list.append(loss.detach().cpu().numpy())
                test_loss_list.append(loss_test.detach().cpu().numpy())
                    
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    best_acc_ep = e
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred
        
        torch.save(self.model.state_dict(), './results/sub%d/model.pth'%self.nSub)
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        self.log_write.write('Best accuracy appears in: ' + str(best_acc_ep) + " epoch\n")
        
        cm = confusion_matrix(Y_true.cpu(), Y_pred.cpu())
        self.log_write.write('\nconfusion_matrix:\n')
        self.log_write.write(str(cm))
        
        # draw the plot
        if train_acc_list:
            plt.figure()
            x = [i for i in range(1, len(train_acc_list) + 1)]
            plt.plot(x, train_acc_list, label="acc_train")
            plt.plot(x, test_acc_list, label="acc_test")
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.savefig(os.path.join(os.path.dirname(self.res_path), "acc.png"))


            plt.figure()
            x = [i for i in range(1, len(train_loss_list) + 1)]
            plt.plot(x, train_loss_list, label="loss_train")
            plt.plot(x, test_loss_list, label="loss_test")
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.savefig(os.path.join(os.path.dirname(self.res_path), "loss.png"))

        return bestAcc, averAcc, cm, best_acc_ep
        # writer.close()