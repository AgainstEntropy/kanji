# Reproduce

## Setup environment

```bash
conda create -n kanji python=3.10
conda activate kanji
pip install -r requirements.txt
```

## Prepare the dataset

### Access form HuggingFace

There are many datasets in the format of (kanji image - text) pairs on HugginFace now. Choose whichever you like, for example, the dataset created by [enpitsu]((https://x.com/enpitsu/status/1610923494824628224?s=20)) (the original author of Kanji Generation), [epts/kanji-full](https://huggingface.co/datasets/epts/kanji-full).

### Create your own dataset

Or, if you would like to create a dataset on your own, check out the following useful links:

- [KANJIDIC Project](http://www.edrdg.org/wiki/index.php/KANJIDIC_Project), from which you can find and download `kanjidic2.xml`. This xml file contains the information of 13,108 kanji, such as their id and meanings in multiple languages.
- [KanjiVG](https://kanjivg.tagaini.net/), from which you can view kanji using a online viewer. You can also find and download seperated xml files of all kanji from its [release page](https://github.com/KanjiVG/kanjivg/releases).

Before getting on to build the dataset, we need to prepare `kanjidic2.xml` and a folder that contains seperated kanji svg files (e.g., `kanjivg-20230110-main`). 

```python
cd data/
python prepare_svg.py
python build_dataset.py
```

> Modify the hardcoded paths and dataset_id in these two `.py` files to match your case.

These two lines do the following things respectively:

1. Parse the `kanjidic2.xml` file and remove all stroke order numbers from the original svg files. This will also create a hashmap that map the kanji id to the English meanings of the corresponding kanji. The hashmap will be saved in the `id_to_text.json` file. We save the svg files without stroke order numbers in a new folder named `kanjivg-20230110-main-wo-num`.

2. Build the image-text paired dataset and upload it to HuggingFace.


## Finetune LDM for Kanji generation (with LoRA)

All the codes for finetuning LDM are in [train_text_to_image_lora.py](../train_lora/train_text_to_image_lora.py), which is a modified version of a [diffusers' example]((https://github.com/huggingface/diffusers/blob/3dd4168d4c96c429d2b74c2baaee0678c57578da/examples/text_to_image/train_text_to_image_lora.py)) for LoRA training.

Modify the `PROJECT_DIR` before running the following line:

    sh scripts/run-lora.sh

You will be prompted to log into your Wandb account the very first time you run these `.sh` scripts.

> With hyperparameters in [run-lora.sh](../scripts/run-lora.sh) unchanged(mainly `--resolution=128`, `--train_batch_size=64`, `--lora_rank=128`), it will need ~14G GPU memory.
>
> Note that by default we disable mixed precision training by setting `--mixed_precision="no"`. This doesn't add much to the memory usage, but avoid a lot of unexpected errors.

Model checkpoints will be save under `./ckpt/` folder with name `pytorch_lora_weights.safetensors` and log files will be found under `./wandb/` folders.


## Distill LDM with LCM-LoRA

To further expedite the image generation, we distill the finetuned LDM with [train_lcm_distill_lora_sd.py](../train_lora/train_lcm_distill_lora_sd.py), which is a modified version of a [diffusers' example](https://github.com/huggingface/diffusers/blob/3dd4168d4c96c429d2b74c2baaee0678c57578da/examples/consistency_distillation/train_lcm_distill_sd_wds.py).

    sh scripts/run-lcm-lora.sh

> With hyperparameters in [run-lcm-lora.sh](../scripts/run-lcm-lora.sh) unchanged(mainly `--resolution=128`, `--train_batch_size=64`, `--lora_rank=64`), it will need ~18G GPU memory.

## Test

Follow the code blocks in [test.ipynb](../notebooks/test.ipynb) to test yout training results.

### Generation with StreamDiffusion

Follow the code blocks in [stream.ipynb](../notebooks/stream.ipynb) to test streaming.
