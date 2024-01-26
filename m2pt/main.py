import torch 
from torch import nn, Tensor
from einops import rearrange, repeat, reduce
from zeta.nn import MultiQueryAttention, FeedForward, Residual, PreNorm
from dataclasses import dataclass, field


class CrossModalReParametrization(nn.Module):
    dim: int
    l: float
    eps: float
    
    def __post__init__(self):
        self.fc_target = nn.Linear(self.dim, self.dim)
        self.fc_aux = nn.Linear(self.dim, self.dim)
        
    def forward(self, target: Tensor, aux: Tensor) -> Tensor:
        target, aux = self.fc_target(target), self.fc_aux(aux)
        
        return target + aux + self.l





@dataclass
class MPTransformerBlock(nn.Module):
    dim: int 
    dim_head: int
    heads: int
    ff_mult: int
    dropout: float
    
    def __post_init__(self):
        self.attn = MultiQueryAttention(
            self.dim,
            self.heads,
        )
        
        self.ffn = FeedForward(
            self.dim,
            self.dim,
            self.ff_mult
        )
        