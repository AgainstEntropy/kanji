import cairosvg
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


import io
from pathlib import Path


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