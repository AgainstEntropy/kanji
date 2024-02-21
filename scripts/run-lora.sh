#!/bin/bash

MODEL_NAME="$VAST/models/stable-diffusion-v1-5"
# DATASET_NAME="AgainstEntropy/kanji-20230110-main"
DATASET_NAME="epts/kanji-full"

PROJECT_DIR="$VAST/codes/kanji"
PROJECT_NAME="kanji-lora-sd-15"
OUTPUT_DIR="${PROJECT_DIR}/ckpts/${PROJECT_NAME}-0"
SCRIPT_PATH="${PROJECT_DIR}/train_text_to_image_lora.py"

PROMPT_PREFIX=""

source $VAST/miniconda3/bin/activate kanji

accelerate launch ${SCRIPT_PATH} \
    --prompt_prefix=${PROMPT_PREFIX} \
    --pretrained_model_name_or_path=${MODEL_NAME}  \
    --dataset_name=${DATASET_NAME} \
    --image_column="image" \
    --caption_column="text" \
    --mixed_precision="no" \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --train_batch_size=16 \
    --dataloader_num_workers=8 \
    --lora_rank=16 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --max_train_steps=5000 \
    --validation_steps=50 \
    --validation_prompt="water" \
    --checkpointing_steps=100 \
    --checkpoints_total_limit=10 \
    --report_to="wandb" \
    --tracker_project_name=${PROJECT_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --resume_from_checkpoint="latest" \
    --seed=0
    # --lr_warmup_steps=200 \
    # --push_to_hub
