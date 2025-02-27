import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    n_latent = n_latent or (n_in + 1) // 2

    return nn.Sequential(nn.Linear(n_in, n_latent), nn.Tanh(), nn.Linear(n_latent, 1))