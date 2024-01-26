import torch
from torch import nn, Tensor
from zeta.nn.attention.multihead_attention import MultiheadAttention


class CrossModalReParametrization(nn.Module):
    """
    A module for cross-modal reparametrization.

    Args:
        original_linear (nn.Linear): The original linear layer.
        auxiliary_linear (nn.Linear): The auxiliary linear layer.

    Attributes:
        cross_modal_scale (nn.Parameter): The scale parameter for cross-modal reparametrization.

    Methods:
        forward(x: Tensor) -> Tensor: Performs forward pass through the module.
        merge(): Merges the weights and biases of the original and auxiliary linear layers.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        auxiliary_linear: nn.Linear,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.auxiliary_linear = auxiliary_linear
        self.cross_modal_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        combined_weight = (
            self.original_linear.weight
            + self.cross_modal_scale * self.auxiliary_linear.weight
        )
        return nn.functional.linear(
            x, combined_weight, self.original_linear.bias
        )

    def merge(self):
        self.original_linear.weight.data.add_(
            self.cross_modal_scale.item()
            * self.auxiliary_linear.weight.data
        )
        if (
            self.original_linear.bias is not None
            and self.auxiliary_linear.bias is not None
        ):
            self.original_linear.bias.data.add_(
                self.cross_modal_scale.item()
                * self.auxiliary_linear.bias.data
            )


class MPTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        original_linear: nn.Linear,
        auxiliar_linear: nn.Linear,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.original_linear = original_linear
        self.auxiliar_linear = auxiliar_linear

        # Cross modal reparametrization
        self.reparametrization = CrossModalReParametrization(
            self.original_linear, self.auxiliar_linear
        )

        # Norm
        self.norm = nn.LayerNorm(self.dim)

        # Check for gpu
        self.is_cuda = torch.cuda.is_available()

        # Flash Attention
        self.mha = MultiheadAttention(
            dim,
            heads,
            dropout,
            subln=True,
        )

    def forward(self, x: Tensor):
        skip = x
        x = self.norm(x)

        # Cross Modal Reparametrization with the q, k, v
        q, k, v = (
            self.reparametrization(x),
            self.reparametrization(x),
            self.reparametrization(x),
        )
        print(f"All shapes: {q.shape}, {k.shape}, {v.shape}")

        # Attention
        # attn, _, _ = self.attn(q, k, v)
        # attn = self.flash_attn(q, k, v)
        attn = self.mha(q, k, v)

        # After attention projections
        attn_out = self.reparametrization(attn) + skip

        # Norm
        attn_out_norm = self.norm(attn_out)

        # Reparameterization agin
        norm_then_reparam = self.reparametrization(attn_out_norm)

        # Reparameterization agin
        reparam_them_reparam = self.reparametrization(
            norm_then_reparam
        )

        return reparam_them_reparam + attn_out_norm


model = MPTransformerBlock(
    dim=512,
    dim_head=64,
    heads=8,
    ff_mult=4,
    dropout=0.1,
    original_linear=nn.Linear(512, 512),
    auxiliar_linear=nn.Linear(512, 512),
)

# 3D tensor B x S x D
x = torch.randn(1, 512, 512)

out = model(x)

print(out.shape)
