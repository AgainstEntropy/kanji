import json
from pathlib import Path

import xmltodict
from tqdm import tqdm

header_string = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd" [
<!ATTLIST g
xmlns:kvg CDATA #FIXED "http://kanjivg.tagaini.net"
kvg:element CDATA #IMPLIED
kvg:variant CDATA #IMPLIED
kvg:partial CDATA #IMPLIED
kvg:original CDATA #IMPLIED
kvg:part CDATA #IMPLIED
kvg:number CDATA #IMPLIED
kvg:tradForm CDATA #IMPLIED
kvg:radicalForm CDATA #IMPLIED
kvg:position CDATA #IMPLIED
kvg:radical CDATA #IMPLIED
kvg:phon CDATA #IMPLIED >
<!ATTLIST path
xmlns:kvg CDATA #FIXED "http://kanjivg.tagaini.net"
kvg:type CDATA #IMPLIED >
]>
"""


def parse_kanjidic2(
        kanjidic2_path: Path,
        save_path: Path = None,
) -> "dict[str, str]":
    with open(kanjidic2_path, "r") as f:
        lines = f.readlines()

    xml_string = ''.join(lines)
    xml_dict = xmltodict.parse(xml_string)

    id_to_text = dict()

    print("Parsing xml ...")
    for character in tqdm(xml_dict['kanjidic2']['character']):
        char_id = character['codepoint']['cp_value'][0]['#text']
        char_id = char_id.lower()
        if len(char_id) < 5:
            char_id = '0' + char_id
        try:
            meanings = character['reading_meaning']['rmgroup']['meaning']
            if not isinstance(meanings, list):
                meanings = [meanings]
            meaning_list = list(filter(lambda x: isinstance(x, str), meanings))
            if len(meaning_list) > 0:
                text = ', '.join(meaning_list)

                id_to_text[char_id] = text
            else:
                print(f"{char_id} has no meaning?")
        except KeyError:
            id_to_text[char_id] = ""
            print(f"{char_id} has no meaning?")

    print(f"Total {len(id_to_text)} characters")

    if save_path is not None:
        with open(save_path, 'w') as json_file:
            json.dump(id_to_text, json_file)

    return id_to_text


def parse_svg(svg_file: Path):
    with open(svg_file, "r") as f:
        svg_raw_string = ''.join(f.readlines())
    
    svg_raw_dict = xmltodict.parse(svg_raw_string)
    svg_raw_dict['svg']['g'] = svg_raw_dict['svg']['g'][0]  # remove stroke number

    svg_string = xmltodict.unparse(svg_raw_dict, full_document=False, pretty=True)
    svg_save_string = header_string + svg_string
    return svg_save_string


def remove_stroke_num(
    id_to_text: dict,
    original_dir: Path,
    save_dir: Path,
):
    ori_svg_list = list(original_dir.glob("*.svg"))
    print("Saving svg ...")

    for svg_file in tqdm(ori_svg_list):
        kanji_id = svg_file.stem
        if kanji_id not in id_to_text:
            print(f"{kanji_id} not in kanjidic2")
            continue

        svg_save_string = parse_svg(svg_file)
        save_svg = save_dir / svg_file.name
        with open(save_svg, "w") as f:
            f.write(svg_save_string)


if __name__ == "__main__":
    kanjidic2 = Path("./kanjidic2.xml")
    json_path = Path("./id_to_text.json")
    id_to_text = parse_kanjidic2(kanjidic2, save_path=json_path)

    original_dir = Path("./kanjivg-20230110-main")
    save_dir = Path("./kanjivg-20230110-main-wo-num")
    save_dir.mkdir(exist_ok=True)
    remove_stroke_num(id_to_text, original_dir, save_dir)
