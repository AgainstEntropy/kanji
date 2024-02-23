#!/bin/bash

SD_VERSION="4"
MODEL_NAME="$VAST/models/stable-diffusion-v1-${SD_VERSION}"
LORA_PATH="$VAST/codes/kanji/ckpts/kanji-lora-sd-14-0/checkpoint-10000/pytorch_lora_weights.safetensors"

# DATASET_NAME="AgainstEntropy/kanji-20230110-main"
DATASET_NAME="epts/kanji-full"

PROJECT_DIR="$VAST/codes/kanji"
PROJECT_NAME="kanji-lora-lcm-sd-1${SD_VERSION}"
OUTPUT_DIR="${PROJECT_DIR}/ckpts/${PROJECT_NAME}-1"
SCRIPT_PATH="${PROJECT_DIR}/train_lcm_distill_lora_sd.py"

# PROMPT_PREFIX="Japanese Kanji character, white background, "
PROMPT_PREFIX=""

source $VAST/miniconda3/bin/activate kanji

accelerate launch ${SCRIPT_PATH} \
    --prompt_prefix="${PROMPT_PREFIX}" \
    --pretrained_teacher_model=${MODEL_NAME} \
    --pretrained_teacher_lora=${LORA_PATH} \
    --dataset_name=${DATASET_NAME} \
    --interpolation_type="bicubic" \
    --image_column="image" \
    --caption_column="text" \
    --mixed_precision="no" \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --train_batch_size=32 \
    --dataloader_num_workers=8 \
    --gradient_accumulation_steps=1 \
    --lora_rank=64 \
    --lora_alpha=64 \
    --learning_rate=1e-6 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=200 \
    --loss_type="huber" \
    --huber_c=0.001 \
    --adam_weight_decay=0.0 \
    --max_train_steps=5000 \
    --validation_steps=50 \
    --checkpointing_steps=100 \
    --checkpoints_total_limit=10 \
    --report_to="wandb" \
    --tracker_project_name=${PROJECT_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --resume_from_checkpoint="latest" \
    --seed=0

    # --gradient_checkpointing \
    # --push_to_hub
    # --use_8bit_adam \