#!/bin/bash

source $VAST/miniconda3/bin/activate kanji

python app-llama.py \
    --llama_model_id_or_path="/vast/yw7486/models/meta-llama/llama-2-7b-chat-hf"
