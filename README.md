# Streaming Kanji

This projects aims to build a novel dialogue system. 

Users are not talking to a common ChatBot in English, but in a kanji-like novel language, where responses are rendered with Diffusion-based models.

Thanks to the [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) pipeline, the system can stream character-level images, just like a normal dialog stream. We build this system based on [StreamDiffusionIO](https://github.com/AgainstEntropy/StreamDiffusionIO), a modified version of StreamDiffusion that supports using different text prompt on different samples respectively but consistently.

## Deploy the System

TBC.


## Reproduce

Check out the [guide](./docs/REPRODUCE.md) on reproducing all components of this system.

## Acknowledgements & References

- Kanji Generation by [enpitsu]((https://x.com/enpitsu/status/1610923494824628224?s=20))
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
- [Latent Consistency Models](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)