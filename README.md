# Kanji Streaming

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-space-yellow)](https://huggingface.co/spaces/AgainstEntropy/Kanji-Streaming)

This project aims to build an interesting dialogue system utilizing the characteristics of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion).

Users are not talking to a common Chatbot in English, but in a kanji-like fake language, where responses are rendered with diffusion-based models.

We build this system based on [StreamDiffusionIO](https://github.com/AgainstEntropy/StreamDiffusionIO), a modified version of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) that supports rendering text streams into image streams.

https://github.com/AgainstEntropy/kanji/assets/42559837/2a623697-94cd-406b-91bc-76c846bdc05a

## News

🔥 Mar 05, 2024 | Kanji Streaming is [reposted](https://x.com/enpitsu/status/1764915414063354236?s=20) by enpitsu (original author of Fake Kanji Generation) on X(twitter)!

⬆️ Mar 04, 2024 | We update the [demo](https://github.com/AgainstEntropy/kanji-streaming-demo/blob/c931afed2ed2dab39781969921d53322d28793e4/app-mixtral.py) and it now allows to chat with [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) with HF InferenceClient, which also significantly saves GPU memory usage (from ~18.5G ➡️ ~5G). Also checkout the demo deployed on [Huggingface Space](https://huggingface.co/spaces/AgainstEntropy/Kanji-Streaming)!

🔥 Mar 01, 2024 | Kanji Streaming is [reposted](https://x.com/Against_Entropy/status/1763305330027503856?s=20) by AK on X(twitter)!

🚀 Feb 29, 2024 | Kanji Streaming is released!

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

> [!TIP]
> See repository of [StreamDiffusionIO](https://github.com/AgainstEntropy/StreamDiffusionIO) for more details.

### Step3: Download model weights

- [Llama-2](https://huggingface.co/meta-llama)
- [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [LoRA for kanji generation](https://huggingface.co/AgainstEntropy/kanji-lora-sd-v1-5): This LoRA enables sd-1.5 to generate kanji at resolution of 128 with text condition.
    - [LCM-LoRA for kanji generation](https://huggingface.co/AgainstEntropy/kanji-lcm-lora-sd-v1-5): This LCM-LoRA turns the LDM into a LCM.

### Step4: Serve with Gradio

Run `git submodule update --init --recursive` to pull the code in `demo` folder.

Modify the arguments (e.g., model paths and `conda` installation path) in the launching scripts to match your case before running.

#### Serve with Mixtral-8x7B-Instruct-v0.1 (HF InferenceClient)

```shell
cd demo
sh run-app-mixtral.sh
```

#### Serve with Llama (run LLM locally)

```shell
cd demo
sh run-kanji-local_llama.sh
```

> [!TIP]
> It will take ~18.5G GPU memory when using Llama-2-7b-chat and Stable-Diffusion-v1-5.


## Reproduce

Check out the [guide](./docs/REPRODUCE.md) on reproducing Kanji generation model used in this project.

## Acknowledgements & References

- Kanji Generation by [enpitsu](https://x.com/enpitsu/status/1610923494824628224?s=20)
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
- [Latent Consistency Models](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)
