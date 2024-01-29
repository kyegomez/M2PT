import torch
from torch import nn
from m2pt import MPTransformerBlock

# Create an instance of the MPTransformerBlock class with the specified parameters
model = MPTransformerBlock(
    dim=512,  # Dimension of the input and output tensors
    dim_head=64,  # Dimension of each attention head
    heads=8,  # Number of attention heads
    dropout=0.1,  # Dropout rate
    ff_mult=4,  # Multiplier for the dimension of the feed-forward network
    original_linear=nn.Linear(512, 512),  # Linear layer for the original input tensor
    auxiliar_linear=nn.Linear(512, 512),  # Linear layer for the auxiliary input tensor
    ffn_original_linear=nn.Linear,  # Linear layer for the original input tensor in the feed-forward network
    ffn_auxiliar_linear=nn.Linear,  # Linear layer for the auxiliary input tensor in the feed-forward network
    ffn_original_last_linear=nn.Linear,  # Last linear layer for the original input tensor in the feed-forward network
    ffn_aux_last_linear=nn.Linear,  # Last linear layer for the auxiliary input tensor in the feed-forward network
)

# Create a 3D tensor with shape B x S x D
x = torch.randn(1, 512, 512)

# Pass the input tensor through the model
out = model(x)

# Print the shape of the output tensor
print(out.shape)
