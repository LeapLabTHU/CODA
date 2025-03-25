# Train CODA-MAR
PYTHONPATH=$(pwd) accelerate launch --config_file accelerate_cfg/1m1g_fp32.yaml train_scripts/train_mar.py configs/mar_16384_config.py

# Train CODA-FLUX
# PYTHONPATH=$(pwd) accelerate launch --config_file accelerate_cfg/1m1g_fp32.yaml train_scripts/train_flux.py configs/flux_65536_config.py
