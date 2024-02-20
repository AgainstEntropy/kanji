#!/bin/bash

MODEL_NAME="$VAST/models/stable-diffusion-v1-5"
VAE_PATH="$VAST/models/stable-diffusion-v1-5"
DATASET_NAME="AgainstEntropy/kanji-20230110-main"

PROJECT_DIR="$VAST/codes/kanji"
PROJECT_NAME="kanji-lora-lcm-sd-15"
OUTPUT_DIR="${PROJECT_DIR}/ckpts/${PROJECT_NAME}-1"
SCRIPT_PATH="${PROJECT_DIR}/train_lcm_distill_lora_sd.py"

source $VAST/miniconda3/bin/activate kanji

accelerate launch ${SCRIPT_PATH} \
    --pretrained_teacher_model=${MODEL_NAME}  \
    --pretrained_vae_model_name_or_path=${VAE_PATH} \
    --dataset_name=${DATASET_NAME} \
    --image_column="image" \
    --caption_column="text" \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --train_batch_size=32 \
    --dataloader_num_workers=8 \
    --gradient_accumulation_steps=1 \
    --lora_rank=128 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=200 \
    --loss_type="huber" \
    --adam_weight_decay=0.0 \
    --max_train_steps=5000 \
    --validation_steps=50 \
    --checkpointing_steps=50 \
    --checkpoints_total_limit=10 \
    --report_to="wandb" \
    --tracker_project_name=${PROJECT_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --resume_from_checkpoint="latest" \
    --seed=0
    # --gradient_checkpointing \
    # --push_to_hub
    # --use_8bit_adam \