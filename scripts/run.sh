#!/bin/bash

MODEL_NAME="$VAST/models/stable-diffusion-v1-5"
VAE_PATH="$VAST/models/stable-diffusion-v1-5"
DATASET_NAME="AgainstEntropy/kanji-20230110-main"

PROJECT_DIR="$VAST/codes/kanji"
SCRIPT_PATH="${PROJECT_DIR}/train_lcm_distill_lora_sd.py"
OUTPUT_DIR="${PROJECT_DIR}/kanji-lora-lcm-sd-15"

source $VAST/miniconda3/bin/activate kanji

accelerate launch ${SCRIPT_PATH} \
    --pretrained_teacher_model=${MODEL_NAME}  \
    --pretrained_vae_model_name_or_path=${VAE_PATH} \
    --output_dir=${OUTPUT_DIR} \
    --dataset_name=${DATASET_NAME} \
    --image_column="image" \
    --caption_column="text" \
    --mixed_precision="fp16" \
    --resolution=256 \
    --train_batch_size=16 \
    --dataloader_num_workers=8 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention \
    --lora_rank=64 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --loss_type="huber" \
    --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --checkpointing_steps=100 \
    --checkpoints_total_limit=10 \
    --validation_steps=50 \
    --seed=0 \
    --resume_from_checkpoint="latest" \
    --report_to="wandb" \
    # --gradient_checkpointing \
    # --push_to_hub
    # --use_8bit_adam \