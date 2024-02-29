#!/bin/bash

source $VAST/miniconda3/bin/activate kanji

python app-kanji.py \
    --llama_model_id_or_path="/vast/yw7486/models/meta-llama/llama-2-7b-chat-hf" \
    --sd_model_id_or_path="/vast/yw7486/models/stable-diffusion-v1-5"  \
    --lora_path="/vast/yw7486/codes/kanji/lora_weights/kanji-lora-sd-v1-5" \
    --lcm_lora_path="/vast/yw7486/codes/kanji/lora_weights/kanji-lcm-lora-sd-v1-5" \
    --tmp_dir="./tmp"