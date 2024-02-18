import json
from pathlib import Path

from tqdm import tqdm

import cairosvg
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from PIL import Image
import io

import datasets


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

def build_hdf5_dataset():
    pass

def build_parquet_dataset(
    svg_list: list,
    id_to_text: dict,
    save_path: Path
):

    data = {'image': [], 'text': []}

    for svg_file in tqdm(svg_list[:10]):
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
        numpy_array = np.array(image)  # (256, 256, 4)

        data['image'].append(numpy_array)
        data['text'].append(id_to_text[kanji_id])

    # Create Pandas DataFrame
    df = pd.DataFrame(data)
    # Convert Pillow Images to bytes for Parquet
    # df['image_bytes'] = df['image'].apply(lambda img: img.tobytes())

    # df['image'] = df['image_bytes']
    # df = df.drop(columns=['image_bytes'])

    # Convert Pandas DataFrame to Arrow Table
    table = pa.Table.from_pandas(df[['image', 'text']])

    # Write the Arrow Table to a Parquet file
    pq.write_table(table, save_path)


def uopload_dataset_to_hf(
    svg_list: list,
    id_to_text: dict,
):
    data_dict = load_data_to_dict(svg_list, id_to_text)
    dataset = datasets.Dataset.from_dict(data_dict)
    print(dataset.features)
    dataset.push_to_hub("AgainstEntropy/kanji", private=True)


if __name__ == "__main__":

    id_to_text_json = Path("./data/id_to_text.json")
    with open(id_to_text_json, 'r') as f:
        id_to_text = json.load(f)
    print(f"Total {len(id_to_text)} characters")

    svg_folder = Path("./data/kanjivg-20230110-main-wo-num")
    svg_list = list(svg_folder.glob("*.svg"))
    print(f"Total {len(svg_list)} SVG files")

    uopload_dataset_to_hf(svg_list, id_to_text)

    # save_path = Path("./data/dataset.parquet")
    # build_parquet_dataset(svg_list, id_to_text, save_path)

    # build_hdf5_dataset()