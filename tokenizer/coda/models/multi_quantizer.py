from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from einops import reduce

from .attention import Attention

def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class CODAQuantizer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int,
        patch_size_list: List[int],
        beta: float = 0.25,
        entropy_temperature: float = 0.01,
        sample_minimization_weight=1.0, batch_maximization_weight=1.0,
        attn_norm_type: str = None,
        attn_dim: int = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.patch_size_list = patch_size_list
        self.codebooks = nn.Embedding(codebook_size, hidden_dim)
        self.attn = Attention(hidden_dim=hidden_dim, norm_type=attn_norm_type, attn_dim=attn_dim)
        self.beta = beta
        self.entropy_temperature = entropy_temperature
        self.codebook_size = codebook_size
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

    def compute_entropy_loss(
        self,
        logits,
        temperature=0.01,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        eps=1e-5,
    ):
        """
        Entropy loss of unnormalized logits

        logits: Affinities are over the last dimension

        https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
        LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
        """
        probs = F.softmax(logits / temperature, -1)
        log_probs = F.log_softmax(logits / temperature + eps, -1)

        avg_probs = reduce(probs, "... D -> D", "mean")

        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

        sample_entropy = -torch.sum(probs * log_probs, -1)
        sample_entropy = torch.mean(sample_entropy)

        loss = (sample_minimization_weight * sample_entropy) - (
            batch_maximization_weight * avg_entropy
        )

        return sample_entropy, avg_entropy, loss

    def forward(
        self,
        model_input: torch.Tensor,
    ) -> Tuple[List[torch.LongTensor], torch.FloatTensor]:
        bs, _, H, W = model_input.shape
        model_idx_list = []
        f_hat = torch.zeros_like(model_input, device=model_input.device)
        f_residual = model_input.clone()
        f_hat.requires_grad_(True)

        loss_list, entropy_loss_list = [], []
        for i, patch_size in enumerate(self.patch_size_list):
            f_reshape = F.interpolate(f_residual, size=(patch_size, patch_size), mode='area')

            logits, idx_N, z_q, z_q_2 = self.attn(f_reshape, self.codebooks.weight)

            sample_entropy, avg_entropy, entropy_loss = self.compute_entropy_loss(logits=logits.permute(0, 2, 3, 1).reshape(-1, self.codebook_size), temperature=self.entropy_temperature, sample_minimization_weight=self.sample_minimization_weight, batch_maximization_weight=self.batch_maximization_weight) # logits [b d h w] -> [b * h * w, n]

            quant_loss = torch.mean((z_q - f_reshape)**2) + torch.mean((z_q_2.detach()-f_reshape)**2) + self.beta * \
                    torch.mean((z_q_2 - f_reshape.detach()) ** 2)

            h_BChw = F.interpolate(z_q, size=(H, W), mode='bicubic').contiguous()

            f_hat = f_hat + h_BChw
            f_residual = f_residual - h_BChw
            model_idx_list.append(idx_N)

            loss_list.append(quant_loss)
            entropy_loss_list.append(entropy_loss)

        vq_loss = torch.mean(torch.mean(torch.stack(loss_list)))
        entropy_loss = torch.mean(torch.mean(torch.stack(entropy_loss_list)))

        return model_idx_list, f_hat, vq_loss, entropy_loss

