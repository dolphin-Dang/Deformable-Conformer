from conformer import PatchEmbedding, MultiHeadAttention, ExP
import torch
import scipy

# t = torch.ones((64, 1, 22, 1000))

# pe = PatchEmbedding(40)
# t = pe(t)
# mha = MultiHeadAttention(40, 10, 0.5)
# t = mha(t)

# data = scipy.io.loadmat("./data/rawMat/s001.mat")
# print(data.keys())

exp = ExP(1)
# _, _, data, label = exp.get_source_data()
# print(data.shape)
# print(label.shape)

exp.train()