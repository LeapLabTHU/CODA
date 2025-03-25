### Model Settings
pretrained_vae_path = None
vae_latent_channels = 16

patch_size_list = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16] # 128 resolution
codebook_size = 16384
encoder_num_layers = 1
decoder_num_layers = 1

# LoRA Settings
decoder_lora_rank = 4
decoder_lora_module_list = ['conv1', 'conv2']

# Init settings
disc_init: float = 0.02 # <0: xavier_normal_(gain=abs(init)); >0: trunc_normal_(std=init)
do_disc_init: float = True

### Dataset Settings
dataset_type = "imagenet"
dataset_cfg = dict(resolution=128)
dataloader_shuffle = True
dataloader_pin_memory = True
dataloader_drop_last = True
dataloader_num_workers = 16

### Training Settings
seed = 42
report_to = "wandb"
tracker_project_name = "Cont-To-Disc_Tokenizer"
logging_dir = "logs"
checkpoints_total_limit = 20

# steps
train_batch_size = 32
num_train_epochs = 10
max_train_steps = None
checkpointing_epochs = 1
resume_from_checkpoint = "latest"
gradient_accumulation_steps = 2

# lr
learning_rate = 1e-5
codebook_learning_rate = 1e-5
quant_conv_learning_rate = 1e-5
scale_lr = False
lr_scheduler = "cosine"
lr_warmup_steps = 0
lr_num_cycles = 1
lr_power = 1.0

criterion_type = 'frozen_var' # ['patch-gan', 'style-gan']
criterion_cfg = dict(
    pretrain_path = 'checkpoints/dino_disc/dino_deitsmall16_pretrain.pth',
    l1_weight = 0.2,
    l2_weight = 1.0,
    perceptual_weight = 1.0, 
    discriminator_weight = 0.4,
    disc_loss = "hinge",
)
vq_loss_weight = 1.0
vq_commitment_beta = 0.25
entropy_loss_weight = 0.05
vq_attn_dim = None
disc_warmup = 0.22 # gan warmup
disc_start = 0.2 # gan warmup

# shared conv
share_quant_resi = 4
quant_resi = 0.5

# ema
ema_ratio = 0.9999
ema_start_epoch = 4

# optim
adam_beta1 = 0.5
adam_beta2 = 0.9
adam_weight_decay = 0.0
adam_epsilon = 0.0
max_grad_norm = 10.0
weight_decay_g = 0.005
weight_decay_d = 0.0005

vq_norm_type = None

### Validation Settings
validation_epochs = 1
validation_images = []
n_evaluate_samples = 50000

# logging
tracker_task_name = '{{fileBasenameNoExtension}}'
output_dir = 'work_dirs/train/{{fileBasenameNoExtension}}'

