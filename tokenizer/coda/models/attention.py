import math
import numbers

import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states

class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim = 16,
        norm_type = None,
        attn_dim = None,
    ):
        super().__init__()

        if attn_dim is None:
            attn_dim = hidden_dim

        self.to_q = nn.Linear(hidden_dim, attn_dim)
        self.to_k = nn.Linear(hidden_dim, attn_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)

        if norm_type is None:
            self.norm_q = None
            self.norm_k = None
        elif norm_type == 'layer_norm':
            self.norm_q = nn.LayerNorm(attn_dim)
            self.norm_k = nn.LayerNorm(attn_dim)
        elif norm_type == 'rms_norm':
            self.norm_q = RMSNorm(attn_dim, eps=1e-5)
            self.norm_k = RMSNorm(attn_dim, eps=1e-5)
        else:
            raise NotImplementedError

    def forward(
        self,
        hidden_states,
        codebook_hidden_states,
    ):
        B, C, H, W = hidden_states.shape
        N, _ = codebook_hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, -1).contiguous()

        query = self.to_q(hidden_states)
        key = self.to_k(codebook_hidden_states)
        value = self.to_v(codebook_hidden_states)

        scale_factor = 1 / math.sqrt(query.size(-1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.permute(0, 2, 1).reshape(B, -1, H, W)
        logits = einsum('b d h w, n d -> b n h w', query, key) * scale_factor

        soft_one_hot = F.softmax(logits, dim=1)

        dim = 1
        idx_N = soft_one_hot.max(dim, keepdim=True)[1]
        hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, idx_N, 1.0)
        one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

        z_q = einsum('b n h w, n d -> b d h w', one_hot, value)
        z_q_2 = einsum('b n h w, n d -> b d h w', hard_one_hot, value)

        return logits, idx_N, z_q, z_q_2
