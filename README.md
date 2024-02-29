# Kanji Streaming

This project aims to build an interesting dialogue system utilizing the characteristics of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion).

Users are not talking to a common Chatbot in English, but in a kanji-like fake language, where responses are rendered with diffusion-based models.

We build this system based on [StreamDiffusionIO](https://github.com/AgainstEntropy/StreamDiffusionIO), a modified version of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) that supports rendering text streams into image streams.


## Deploy

### Step0: Clone this repo

```shell
git clone https://github.com/AgainstEntropy/kanji.git
cd kanji
```

### Step1: Setup environment

```shell
conda create -n kanji python=3.10
conda activate kanji
pip install -r requirements.txt
```

### Step2: Install StreamDiffusionIO

#### For Users

```shell
pip install StreamDiffusionIO
```

#### For Developers

To pull the source code of [StreamDiffusionIO](https://github.com/AgainstEntropy/StreamDiffusionIO), one can either do

```shell
git submodule update --init --recursive
```

or 

```shell
git clone https://github.com/AgainstEntropy/StreamDiffusionIO.git
```

Then install StreamDiffusionIO in editable mode

```shell
pip install --editable StreamDiffusionIO/
```

> See repository of [StreamDiffusionIO](https://github.com/AgainstEntropy/StreamDiffusionIO) for more details.

### Step3: Download model weights

- [Llama-2](https://huggingface.co/meta-llama)
- [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [LoRA for kanji generation](https://huggingface.co/AgainstEntropy/kanji-lora-sd-v1-5): This LoRA enables sd-1.5 to generate kanji at resolution of 128 with text condition.
    - [LCM-LoRA for kanji generation](https://huggingface.co/AgainstEntropy/kanji-lcm-lora-sd-v1-5): This LCM-LoRA turns the LDM into a LCM.

### Step4: Serve with Gradio

Modify the arguments (e.g., model paths and `conda` installation path) in the [launching script](./demo/run-kanji.sh) to match your case before running the script.

```shell
cd demo
sh run-kanji.sh
```

> It will take ~18.5G GPU memory when using Llama-2-7b-chat and Stable-Diffusion-v1-5.


## Reproduce

Check out the [guide](./docs/REPRODUCE.md) on reproducing Kanji generation model used in this project.

## Acknowledgements & References

- Kanji Generation by [enpitsu](https://x.com/enpitsu/status/1610923494824628224?s=20)
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
- [Latent Consistency Models](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)