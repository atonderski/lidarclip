from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        input_dim: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.num_heads = num_heads

    def _no_attention_no_pool(self, x: Tensor) -> Tensor:
        x = x[1:]  # remove the "fake" averaged token (HW+1)NC -> (HW)NC
        return self.c_proj(self.v_proj(x))  # (HW)NC

    def forward(
        self, x, no_pooling: bool = False, return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        key_padding_mask = (x.abs().sum(dim=-1) == 0).permute(1, 0)  # (HW)NC -> (HW)N -> N(HW)
        x = self.in_proj(x)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        if no_pooling:
            assert not return_attention, "Cannot return attention if we skip the attention pooling"
            return self._no_attention_no_pool(x), key_padding_mask
        else:
            query = x[0:1]
            attn_mask = None
        x, weights = F.multi_head_attention_forward(
            query=query,
            key=x[1:],
            value=x[1:],
            key_padding_mask=key_padding_mask,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=return_attention,
            attn_mask=attn_mask,
        )

        return x[0], weights
