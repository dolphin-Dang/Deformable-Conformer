import torch
import scipy

para_dict = torch.load("./results_old_2000ep/models/model_sub1.pth")
keys = para_dict.keys()
print(keys)