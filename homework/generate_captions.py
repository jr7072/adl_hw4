from pathlib import Path

import fire
from tqdm import tqdm
import json
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info
)

import os

os.path

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """

    track_name = extract_track_info(info_path)
    kart_metadata = extract_kart_objects(
        info_path, view_index, img_width, img_height
    )

    # 1. Ego car
    # {kart_name} is the ego car.
    #
    # The ego car only exists in a view that contains at least one valid kart.
    captions = []
    if kart_metadata:

        ego_kart = next(kart for kart in kart_metadata if kart["is_center_kart"])
        ego_kart_x, ego_kart_y = ego_kart["center"]
        captions.append(
            {
                "caption": f"{ego_kart['kart_name']} is the ego car."
            }
        )

        # 4. Relative position
        # {kart_name} is {position} the ego car.
        #
        # Image coordinates increase downward, so a kart with a smaller y-coordinate
        # than the ego kart is in front of it.
        for kart in kart_metadata:

            if kart["is_center_kart"]:
                continue

            kart_x, kart_y = kart["center"]

            horizontal_position = "right of"
            if kart_x < ego_kart_x:
                horizontal_position = "left of"

            captions.append(
                {
                    "caption": f"{kart['kart_name']} is {horizontal_position} the ego car."
                }
            )
            
            vertical_position = "behind"
            if kart_y < ego_kart_y:
                vertical_position ="in front of"

            captions.append(
                {
                    "caption": f"{kart['kart_name']} is {vertical_position} the ego car."
                }
            )

    # 2. Counting
    # There are {num_karts} karts in the scenario.
    captions.append(
        {
            "caption": f"There are {len(kart_metadata)} karts in the scene."
        }
    )

    # 3. Track name
    # The track is {track_name}.
    captions.append(
        {
            "caption": f"The track is {track_name}."
        }
    )

    info_hex = info_path.split("/")[-1].replace("_info.json", "")
    for gen_caption in captions:
        str(info_path).split
        gen_caption["image_file"] = f"train/{info_hex}_0{view_index}_im.jpg"

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def generate_all_captions(
    split: str="train"
):

    data_path = Path(__file__).parent.parent / f"data/{split}"
    qa_data = list()

    info_files = list(data_path.glob("*_info.json"))
    for info_file in tqdm(info_files):
        for i in range(10):
            qa = generate_caption(str(info_file), i)
            qa_data += qa

    with open(data_path / f"{split}_captions.json", "w") as f:
        json.dump(qa_data, f)

def main():
    fire.Fire({"check": check_caption, "generate": generate_all_captions})


if __name__ == "__main__":
    main()
