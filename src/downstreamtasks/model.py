import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.append("../")
import torch.nn.functional as F
from ABMIL import Attention

class ABMIL_Classification(nn.Module):
    def __init__(self, embed_dim: int ,
                 n_out: int = 8,):
        super().__init__()

        #self.aggregator = Aggregator(embed_dim, num_heads, num_seeds)
        self.encoder = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU())
        self.aggregator =  Attention(256, 256)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256,256),
            nn.Dropout(),
            nn.SiLU(),
            nn.Linear(256,n_out),
        )

        # for param in self.aggregator.parameters():
        #     param.requires_grad = True
    
    def forward(self,x): #,idxs):
        #return self.head(self.aggregator(x,idxs))
        embed = self.encoder(x)
        att = self.aggregator(embed)
        att = torch.softmax(att, dim=1)
        # assert att.sum()==att.shape[0], f'something wenxt wrong with {att}, sum was {att.sum()}' 
        w_embed = (att*embed).sum(-2)
        return self.head(w_embed), att



class Simple_Classification(nn.Module):
    def __init__(self,
                 input_dim: int , 
                 n_out: int = 8,
                 
                 ):
        super().__init__()
        self.class_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.Dropout(),
            nn.SiLU(),
            nn.Linear(256, n_out),
        )
        
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

         
    def dtype(self):
        return torch.float32

    def forward(self, image_features):
        #image_features =self.encoder.trunk(image) #trunk for deeper layers of biomedclip
        logits = self.class_head(image_features)
        return logits.squeeze(1)


