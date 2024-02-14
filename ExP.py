'''
ExP class.
The training logic.
'''


import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from models import Conformer
from torch.backends import cudnn
from torchsummary import summary

import scipy.io

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # arrange GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus)) # choose GPUs

class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 2000
        self.c_dim = 4
        self.lr = 0.0002
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
                # Tok, Cls = self.model(test_data)
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

        dir_name = "./results/models"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(self.model.module.state_dict(), './results/models/model_sub%d.pth'%self.nSub)
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()