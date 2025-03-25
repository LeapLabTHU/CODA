from typing import List, Tuple
from torch import nn

from .lora import (
    LoRAConv2dLayer,
    LoRACompatibleConv,
)


def build_peft_from_vae(
    vae,
    rank: int,
    lora_module_list: List[str],
):
    for name, module in vae.named_modules():
        for lora_module in lora_module_list:
            if lora_module in name:
                if isinstance(module, nn.Conv2d):
                    # 1. change to LoRACompatibleConv
                    lora_compatible_conv = LoRACompatibleConv(
                        module.in_channels,
                        module.out_channels,
                        module.kernel_size,
                        stride = module.stride,
                        padding = module.padding,
                        dilation = module.dilation,
                        groups = module.groups,
                        bias = module.bias is not None,
                        padding_mode = module.padding_mode,
                    )
                    lora_compatible_conv.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_compatible_conv.bias.data = module.bias.data.clone()
                    module = lora_compatible_conv
                    module.requires_grad_(False)

                    # 2. inject lora conv
                    lora_conv = LoRAConv2dLayer(
                                    module.in_channels, 
                                    module.out_channels, 
                                    rank=rank, 
                                    kernel_size=module.kernel_size,
                                    stride=module.stride,
                                    padding=module.padding
                                )
                    lora_conv.requires_grad_(True)
                    module.set_lora_layer(lora_conv)

                    father_module = vae
                    for n in name.split(".")[:-1]:
                        father_module = getattr(father_module, n)
                    setattr(father_module, name.split(".")[-1], module)
                break
    return vae


def init_weights(model, conv_std_or_gain):
    print(f'[init_weights] {type(model).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if conv_std_or_gain > 0:
                nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
            else:
                nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None: nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None: nn.init.constant_(m.weight.data, 1.)
