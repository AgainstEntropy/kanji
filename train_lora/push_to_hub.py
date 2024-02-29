import os

from diffusers.utils.hub_utils import (load_or_create_model_card,
                                       populate_model_card)
from huggingface_hub import create_repo, upload_folder


def save_model_card(
    repo_id: str, 
    images: list = None, 
    base_model: str = None, 
    dataset_name: str = None, 
    repo_folder: str = None,
):
    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. 
"""

    os.makedirs(repo_folder, exist_ok=True)

    if images is not None:
        img_str = ""
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"
        
        model_description += img_str

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
    )
    
    model_card = populate_model_card(model_card)

    model_card.save(os.path.join(repo_folder, "README.md"))


if __name__ == "__main__":
    repo_name = "kanji-lcm-lora-sd-v1-5"

    repo_id = create_repo(
        repo_id=repo_name, 
        exist_ok=True, 
        token="YOUR_TOKEN_HERE",
        private=True
    ).repo_id
    
    save_model_card(
        repo_id,
        images=None,
        base_model="runwayml/stable-diffusion-v1-5",
        dataset_name="epts/kanji-full",
        repo_folder=os.path.expandvars(f"$VAST/codes/kanji/lora_weights/{repo_name}"),
    )
    upload_folder(
        repo_id=repo_id,
        folder_path=os.path.expandvars(f"$VAST/codes/kanji/lora_weights/{repo_name}"),
    )