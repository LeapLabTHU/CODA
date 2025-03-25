import torch
import torch.nn as nn

from .vqperceptual import *
from .dino import DinoDisc


class LPIPSDiscriminatorCriterion(nn.Module):
    def __init__(
            self, 
            pretrain_path,
            l1_weight = 0.2,
            l2_weight = 1.0,
            perceptual_weight = 1.0, 
            discriminator_weight = 0.4,
            disc_loss="hinge"
        ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = discriminator_weight

        self.perceptual_loss = LPIPS().eval()

        self.discriminator = DinoDisc( # hard code DINO-s parameters
            dino_ckpt_path=pretrain_path,
            depth=12, key_depths=(2, 5, 8, 11),
            ks=9, norm_type='sbn', using_spec_norm=True, norm_eps=1e-6,
        )
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0].data.norm()
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0].data.norm()
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0].data.norm()
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0].data.norm()

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
            self, 
            inputs, 
            reconstructions, 
            optimizer_idx,
            last_layer = None, 
            kwargs = None,
        ):
        if optimizer_idx == 0:
            rec_loss = F.l1_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss_l1 = rec_loss.clone()
            rec_loss *= self.l1_weight
            if self.l2_weight > 0:
                rec_loss += F.mse_loss(reconstructions, inputs).mul_(self.l2_weight)

            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                nll_loss = rec_loss + self.perceptual_weight * p_loss

            nll_loss = torch.mean(nll_loss)

            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)

            loss = nll_loss +\
                    kwargs['warmup_disc_schedule'] * d_weight * g_loss

            log = {
                "total_loss": loss.clone().detach().mean(), 
                "rec_loss": rec_loss_l1.detach().mean(),
                "perceptual_loss": p_loss.detach().mean(),
                "nll_loss": nll_loss.detach().mean(),
                "g_loss": g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            d_loss = self.disc_loss(logits_real, logits_fake)
            acc_real, acc_fake = (logits_real.data > 0).float().mean().mul_(100), (logits_fake.data < 0).float().mean().mul_(100)

            log = {
                "disc_loss": d_loss.clone().detach().mean(),
                "logits_real": logits_real.detach().mean(),
                "logits_fake": logits_fake.detach().mean(),
                "acc_real": acc_real.detach(),
                "acc_fake": acc_fake.detach(),
            }
            return d_loss, log
