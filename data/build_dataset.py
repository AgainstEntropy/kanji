import io
import json
from pathlib import Path

import cairosvg
import datasets
from PIL import Image
from tqdm import tqdm


def load_data_to_dict(
    svg_list: list,
    id_to_text: dict,
) -> dict:
    data = {'image': [], 'text': []}

    for svg_file in tqdm(svg_list):
        kanji_id = svg_file.stem
        if kanji_id not in id_to_text:
            print(f"{kanji_id} not in kanjidic2")
            continue
        
        # Convert SVG to PNG using CairoSVG
        png_data = cairosvg.svg2png(url=str(svg_file), 
                                    dpi=192, scale=2,
                                    output_width=256, output_height=256)
        # Create a BytesIO object
        s = io.BytesIO(png_data)

        # Load the PNG data into a PIL Image object
        image = Image.open(s)

        data['image'].append(image)
        data['text'].append(id_to_text[kanji_id])

    return data


if __name__ == "__main__":

    id_to_text_json = Path("./id_to_text.json")
    with open(id_to_text_json, 'r') as f:
        id_to_text = json.load(f)
    print(f"Total {len(id_to_text)} characters")

    svg_folder = Path("./kanjivg-20230110-main-wo-num")
    svg_list = list(svg_folder.glob("*.svg"))
    print(f"Total {len(svg_list)} SVG files")

    data_dict = load_data_to_dict(svg_list, id_to_text)
    dataset = datasets.Dataset.from_dict(data_dict)
    print(dataset.features)

    dataset_id = "AgainstEntropy/kanji"
    dataset.push_to_hub(dataset_id, private=True)