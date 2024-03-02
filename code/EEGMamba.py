import torch
from mamba.mamba import Mamba, MambaConfig
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from models import PatchEmbedding, TransformerEncoder, ClassificationHead

    
class EEGMamba(nn.Module):
    def __init__(self,
                mamba_depth=3,
                block_depth=3,
                n_classes=4, 
                config=None):
        
        super().__init__()
        channel = 22
        hidden_size_1 = 256
        hidden_size_2 = 32
        drop_p_1 = 0.5
        drop_p_2 = 0.3
        if config != None:
            hidden_size_1 = config["hidden_size_1"]
            hidden_size_2 = config["hidden_size_2"]
            drop_p_1 = config["drop_p_1"]
            drop_p_2 = config["drop_p_2"]
            n_classes = config["n_classes"]
            channel = config["channel"]
            mamba_depth = config["mamba_depth"]
            block_depth = config["block_depth"]
            
        self.mambas = nn.ModuleList()

        for i in range(block_depth):
            mamba_config = MambaConfig(d_model=channel, n_layers=mamba_depth)
            mamba = Mamba(mamba_config)
            self.mambas.append(mamba)
            
        self.down_sample = nn.AvgPool2d((15, 1), (3,1))
        self.fc = nn.Sequential(
            nn.Linear(31*22, hidden_size_1), # 1000 sample data
            # nn.Linear(440, hidden_size_1), # 250 sample data
            nn.ELU(),
            nn.Dropout(drop_p_1),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ELU(),
            nn.Dropout(drop_p_2),
            nn.Linear(hidden_size_2, n_classes)
        )
       
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = x.squeeze().permute(0,2,1)
        for mamba in self.mambas:
            x = mamba(x) + x
            x = self.down_sample(x)
        output = self.fc(x.reshape(bs, -1))
        return x, output