#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import time
import gc
import logging
import math
import os
import shutil
import os.path as osp
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

from coda.utils import parse_config
from coda.models import (
    CODAQuantizer,
    build_peft_from_vae,
)
from coda.losses import (
    LPIPSDiscriminatorCriterion,
    build_disc_criterion,
)
from coda.datasets import (
    build_imagenet_dataset,
)
from coda.datasets.utils import normalize_01_into_pm1
from coda.utils import (
    VAE_SAVE_MODEL_NAME,
    QUANTIZER_SAVE_MODEL_NAME,
    DISCRIMINATOR_SAVE_MODEL_NAME,
    VAE_EMA_SAVE_MODEL_NAME,
    QUANTIZER_EMA_SAVE_MODEL_NAME,
    FIDCalculator,
)

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)

@torch.no_grad()
def log_validation(
    vae,
    multi_quantizer,
    config,
    accelerator,
    global_step
):
    logger.info(f"Running validation...")
    multi_quantizer.eval()

    transform = transforms.Compose([
        transforms.Resize(config.dataset_cfg.resolution, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((config.dataset_cfg.resolution, config.dataset_cfg.resolution)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ])

    # run inference
    image_logs = []
    for image in config.validation_images:
        img = Image.open(image).convert('RGB')
        pixel_values = transform(img)
        pixel_values = pixel_values.to(device=accelerator.device)
        pixel_values = pixel_values.view(1, *pixel_values.shape)

        model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor

        _, latents, _, _ = multi_quantizer(model_input)

        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]

        image_logs.append(
            {
                "images": [image],
            }
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                for idx, image in enumerate(images):
                    image = wandb.Image(image)
                    formatted_images.append(image)
            tracker.log({"validation": formatted_images})

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@torch.no_grad()
def eval(
    vae,
    multi_quantizer,
    config,
    accelerator,
    global_step,
):
    logger.info(f"Running evaluation...")
    multi_quantizer.eval()
    eval_batch_size = 8
    device = accelerator.device

    fid_calculator = FIDCalculator(accelerator, n_samples=config.n_evaluate_samples, test_bsz=eval_batch_size)

    eval_dataset = build_imagenet_dataset(split='val', **config.dataset_cfg)
    
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size = eval_batch_size,
        num_workers = config.dataloader_num_workers,
        pin_memory = True,
        drop_last = False,
        shuffle = True
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)

    for i, batch in enumerate(eval_dataloader):
        pixel_values = batch[0]
        pixel_values = pixel_values.to(device=device)

        # Convert images to latent space
        model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor

        _, latents, _, _ = multi_quantizer(model_input)

        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]

        res = fid_calculator.add(accelerator, image, pixel_values)
        if res is not None:
            break

    if res is None:
        res = fid_calculator.get_metrics(accelerator)
    if accelerator.is_main_process:
        # log rfid
        accelerator.log(res, step=global_step)
    accelerator.wait_for_everyone()

    del fid_calculator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(config):
    logging_dir = Path(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    )
    multi_quantizer = CODAQuantizer(
        hidden_dim = config.vae_latent_channels,
        codebook_size = config.codebook_size,
        patch_size_list = config.patch_size_list,
        beta = config.vq_commitment_beta,
        attn_norm_type = config.vq_norm_type,
        attn_dim = config.vq_attn_dim,
    )
    criterion = build_disc_criterion(config)

    vae.requires_grad_(False)
    multi_quantizer.requires_grad_(True)

    # inject lora
    vae = build_peft_from_vae(
        vae, rank=config.decoder_lora_rank, 
        lora_module_list=config.decoder_lora_module_list,
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    multi_quantizer.to(accelerator.device, dtype=weight_dtype)
    criterion.to(accelerator.device, dtype=torch.float32)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
            return
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                torch.save(vae_ema.state_dict(), osp.join(output_dir, VAE_EMA_SAVE_MODEL_NAME))
                torch.save(multi_quantizer_ema.state_dict(), osp.join(output_dir, QUANTIZER_EMA_SAVE_MODEL_NAME))

                if isinstance(model, AutoencoderKL):
                    torch.save(model.state_dict(), osp.join(output_dir, VAE_SAVE_MODEL_NAME))
                elif isinstance(model, CODAQuantizer):
                    torch.save(model.state_dict(), osp.join(output_dir, QUANTIZER_SAVE_MODEL_NAME))
                elif isinstance(model, LPIPSDiscriminatorCriterion):
                    torch.save(model.state_dict(), osp.join(output_dir, DISCRIMINATOR_SAVE_MODEL_NAME))
                else:
                    raise ValueError

                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
            return
        
        vae_ema.load_state_dict(torch.load(osp.join(input_dir, VAE_EMA_SAVE_MODEL_NAME), map_location='cpu'), strict=True)
        multi_quantizer_ema.load_state_dict(torch.load(osp.join(input_dir, QUANTIZER_EMA_SAVE_MODEL_NAME), map_location='cpu'), strict=True)

        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            if isinstance(model, AutoencoderKL):
                load_model = torch.load(osp.join(input_dir, VAE_SAVE_MODEL_NAME), map_location='cpu')
            elif isinstance(model, CODAQuantizer):
                load_model = torch.load(osp.join(input_dir, QUANTIZER_SAVE_MODEL_NAME), map_location='cpu')
            elif isinstance(model, LPIPSDiscriminatorCriterion):
                load_model = torch.load(osp.join(input_dir, DISCRIMINATOR_SAVE_MODEL_NAME), map_location='cpu')
            else:
                raise ValueError

            model.load_state_dict(load_model, strict=True)
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    model_parameters = list(filter(lambda p: p.requires_grad, vae.parameters())) + list(filter(lambda p: p.requires_grad, multi_quantizer.parameters()))
    params_to_optimize = [
        {"params": list(filter(lambda p: p.requires_grad, vae.parameters())), "lr": config.learning_rate, "weight_decay": config.weight_decay_g},
        {"params": list(filter(lambda p: p.requires_grad, multi_quantizer.parameters())), "lr": config.codebook_learning_rate, "weight_decay": config.weight_decay_g},
    ]

    model_parameters_d = list(filter(lambda p: p.requires_grad, criterion.parameters()))
    model_parameters_d_with_lr = {"params": model_parameters_d, "lr": config.learning_rate, "weight_decay": config.weight_decay_d}
    params_to_optimize_d = [model_parameters_d_with_lr]

    if accelerator.is_main_process:
        for i, param_set in enumerate(params_to_optimize):
            num_params = sum([p.numel() for p in param_set["params"]]) / 1e+6
            print(f"Trainable Params Set {i}: {num_params:02f}M")

    optimizer_class = torch.optim.AdamW
    optimizer_g = optimizer_class(
        params_to_optimize,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay_g,
        # eps=config.adam_epsilon,
    )
    optimizer_d = optimizer_class(
        params_to_optimize_d,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay_d,
        # eps=config.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    if config.dataset_type == 'imagenet':
        train_dataset = build_imagenet_dataset(**config.dataset_cfg)
    else:
        raise NotImplementedError

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=config.dataloader_shuffle,
        pin_memory=config.dataloader_pin_memory,
        drop_last=config.dataloader_drop_last,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_g = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer_g,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )
    lr_scheduler_d = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer_d,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    from copy import deepcopy
    vae_ema = deepcopy(vae).eval()
    multi_quantizer_ema = deepcopy(multi_quantizer).eval()

    def ema_update(config, vae, multi_quantizer, global_step, epoch):
        ema_ratio = min(config.ema_ratio, (global_step//2 + 1) / (global_step//2 + 10))
        for p_ema, p in zip(vae_ema.parameters(), vae.parameters()):
            if p.requires_grad:
                if epoch < config.ema_start_epoch:
                    p_ema.data.copy_(p.data)
                else:
                    p_ema.data.mul_(ema_ratio).add_(p.data, alpha=1-ema_ratio)
        for p_ema, p in zip(vae_ema.buffers(), vae.buffers()):
            p_ema.data.copy_(p.data)
        for p_ema, p in zip(multi_quantizer_ema.parameters(), multi_quantizer.parameters()):
            if p.requires_grad:
                if epoch < config.ema_start_epoch:
                    p_ema.data.copy_(p.data)
                else:
                    p_ema.data.mul_(ema_ratio).add_(p.data, alpha=1-ema_ratio)

    # Prepare everything with our `accelerator`.
    vae, multi_quantizer, criterion, optimizer_g, optimizer_d, train_dataloader, lr_scheduler_g, lr_scheduler_d = accelerator.prepare(
        vae, multi_quantizer, criterion, optimizer_g, optimizer_d, train_dataloader, lr_scheduler_g, lr_scheduler_d
    )
    if accelerator.num_processes > 1:
        criterion.module.discriminator.dino_proxy[0].to(accelerator.device)
    else:
        criterion.discriminator.dino_proxy[0].to(accelerator.device)
    vae_ema.to(accelerator.device)
    multi_quantizer_ema.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(copy.deepcopy(config))
        task_name = f'{config.tracker_task_name}_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()+8*3600))}' # beijing time

        accelerator.init_trackers(
            config.tracker_project_name,
            config=tracker_config,
            init_kwargs=dict(wandb=dict(name=task_name, entity="shuidi"))
        )

    # Train!
    if accelerator.is_main_process:
        with open(osp.join(config.output_dir, "exp_config.py"), "w") as f:
            f.write(config.pretty_text)
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            first_epoch = int(path.split("-")[1]) + 1

            initial_global_step = first_epoch * num_update_steps_per_epoch
            global_step = first_epoch * num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, config.num_train_epochs):
        multi_quantizer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [vae, multi_quantizer]

            with accelerator.accumulate(models_to_accumulate):
                pixel_values = batch[0].to(dtype=weight_dtype)

                warmup_disc_schedule = global_step / (config.num_train_epochs * num_update_steps_per_epoch)
                warmup_disc_schedule = min(warmup_disc_schedule, 1.0)
                fade_blur_schedule = min(1.0, global_step / (config.num_train_epochs * num_update_steps_per_epoch * 2))
                fade_blur_schedule = 1 - fade_blur_schedule

                criterion_kwargs = {
                    'warmup_disc_schedule': warmup_disc_schedule,
                    'fade_blur_schedule': fade_blur_schedule,
                }

                # Convert images to latent space
                vae_wo_ddp = unwrap_model(vae)
                model_input = vae_wo_ddp.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_wo_ddp.config.shift_factor) * vae_wo_ddp.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                _, latents, vq_loss, entropy_loss = multi_quantizer(model_input)

                latents = (latents / vae_wo_ddp.config.scaling_factor) + vae_wo_ddp.config.shift_factor
                images = unwrap_model(vae).decode(latents, return_dict=False)[0]

                loss, log_g = criterion(
                    pixel_values.float(), images.float(), 
                    optimizer_idx = 0,
                    last_layer = unwrap_model(vae).decoder.up_blocks[-1].resnets[-1].conv2.lora_layer.up.weight, # change to last peft out layer
                    kwargs = criterion_kwargs,
                )

                log_g.update({'vq_loss': vq_loss.detach().item(), 'entropy_loss': entropy_loss.detach().item()})
                loss += vq_loss * config.vq_loss_weight + entropy_loss * config.entropy_loss_weight

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model_parameters
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                optimizer_g.step()
                lr_scheduler_g.step()
                optimizer_g.zero_grad()

                if accelerator.sync_gradients:
                    ema_update(config, unwrap_model(vae), unwrap_model(multi_quantizer), global_step, epoch)

                loss_d, log_d = criterion(pixel_values.float(), images.float(), optimizer_idx=1, kwargs = criterion_kwargs)

                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    params_to_clip = model_parameters_d
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                optimizer_d.step()
                lr_scheduler_d.step()
                optimizer_d.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler_g.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                logs.update(log_g)
                logs.update(log_d)
                accelerator.log(logs, step=global_step)

                if global_step >= config.max_train_steps:
                    break

        if accelerator.is_main_process:
            if epoch % config.checkpointing_epochs == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if config.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(config.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= config.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(config.output_dir, f"checkpoint-{epoch}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

        if accelerator.is_main_process:
            if config.validation_images is not None and epoch % config.validation_epochs == 0:
                log_validation(
                    vae=vae_ema,
                    multi_quantizer=multi_quantizer_ema,
                    config=config,
                    accelerator=accelerator,
                    global_step=global_step,
                )
        eval(
            vae=vae_ema,
            multi_quantizer=multi_quantizer_ema,
            config=config,
            accelerator=accelerator,
            global_step=global_step+1,
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    config = parse_config()
    main(config)