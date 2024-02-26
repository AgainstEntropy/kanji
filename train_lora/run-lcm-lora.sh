#!/bin/bash

MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATASET_NAME="epts/kanji-full"

PROJECT_DIR="$VAST/codes/kanji"
PROJECT_NAME="kanji-lora-lcm"
OUTPUT_DIR="${PROJECT_DIR}/ckpts/${PROJECT_NAME}-0"
SCRIPT_PATH="${PROJECT_DIR}/train_lora/train_lcm_distill_lora_sd.py"

LORA_PATH="${PROJECT_DIR}/ckpts/kanji-lora-0/pytorch_lora_weights.safetensors"

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
    --resolution=128 \
    --train_batch_size=64 \
    --lora_rank=64 \
    --lora_alpha=64 \
    --dataloader_num_workers=8 \
    --learning_rate=1e-6 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=200 \
    --loss_type="huber" \
    --huber_c=0.01 \
    --adam_weight_decay=0.0 \
    --max_train_steps=5000 \
    --validation_steps=100 \
    --checkpointing_steps=100 \
    --checkpoints_total_limit=10 \
    --report_to="wandb" \
    --tracker_project_name=${PROJECT_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --resume_from_checkpoint="latest" \
    --seed=0