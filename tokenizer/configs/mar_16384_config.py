_base_ = 'base.py'

pretrained_vae_path = 'checkpoints/mar_vae/kl16.safetensors'

decoder_lora_module_list = ['conv1', 'conv2', 'upsample', 'downsample']

resume_from_checkpoint = 'latest'
criterion_type = 'frozen_var' # ['patch-gan', 'style-gan']
criterion_cfg = dict(
    pretrain_path = 'checkpoints/dino_disc/dino_deitsmall16_pretrain.pth',
    l1_weight = 0.2,
    l2_weight = 1.0,
    perceptual_weight = 1.0, 
    discriminator_weight = 1.0,
    disc_loss = "hinge",
)
disc_warmup = 0.02 # gan warmup

dataset_type = "imagenet"
dataset_cfg = dict(
    resolution=256, 
    augmentation=True,
    data_path='PATH_TO_IMAGENET'
)

vq_commitment_beta = 0.05

patch_size_list = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16] # 128 resolution
codebook_size = 16384

decoder_lora_rank = 32
num_train_epochs = 10

learning_rate = 5e-4
codebook_learning_rate = 5e-4

train_batch_size = 16
gradient_accumulation_steps = 4

vq_attn_dim = 32

vq_norm_type = 'rms_norm'

validation_images = [
    'eval/0.png',
    'eval/1.png',
    'eval/2.png',
    'eval/3.png',
]

# logging
tracker_task_name = '{{fileBasenameNoExtension}}'
output_dir = 'work_dirs/train/{{fileBasenameNoExtension}}'
