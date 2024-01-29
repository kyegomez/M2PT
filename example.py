import torch
from torch import nn
from m2pt import MPTransformerBlock

model = MPTransformerBlock(
    dim=512,
    dim_head=64,
    heads=8,
    dropout=0.1,
    ff_mult=4,
    original_linear=nn.Linear(512, 512),
    auxiliar_linear=nn.Linear(512, 512),
    ffn_original_linear=nn.Linear,
    ffn_auxiliar_linear=nn.Linear,
    ffn_original_last_linear=nn.Linear,
    ffn_aux_last_linear=nn.Linear,
)

# 3D tensor B x S x D
x = torch.randn(1, 512, 512)

out = model(x)

print(out.shape)
