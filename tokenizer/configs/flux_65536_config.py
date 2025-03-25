_base_ = 'base.py'

pretrained_vae_path = 'checkpoints/flux_vae'

decoder_lora_module_list = ['conv1', 'conv2']

resume_from_checkpoint = None
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

patch_size_list = [1, 2, 3, 4, 6, 9, 13, 18, 24, 32]
codebook_size = 65536

decoder_lora_rank = 32
num_train_epochs = 10

learning_rate = 5e-4
codebook_learning_rate = 5e-4

train_batch_size = 12
gradient_accumulation_steps = 3

vq_norm_type = 'rms_norm'
vq_attn_dim = 32

validation_images = [
    'eval/0.png',
    'eval/1.png',
    'eval/2.png',
    'eval/3.png',
]

# logging
tracker_task_name = '{{fileBasenameNoExtension}}'
output_dir = 'work_dirs/train/{{fileBasenameNoExtension}}'
