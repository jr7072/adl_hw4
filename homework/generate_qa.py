import json
from pathlib import Path
import tqdm

import fire
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
# these are bounding boxes
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    with open(info_path, "r") as f:
        info_json = json.load(f)

    kart_names = info_json["karts"]

    detections = info_json["detections"]
    view_detections = np.array(detections[view_index])
    object_ids = view_detections[:, 0]
    kart_detections = view_detections[np.where(object_ids == 1)[0]]


    kart_list = list()
    glob_center_x, glob_center_y = (img_width / 2, img_height / 2)
    min_dist = np.inf
    center_kart = None

    valid_kart_idx = -1
    for kart in kart_detections:
        
        _, instance_id, x1, y1, x2, y2 = kart.tolist()

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        valid_kart_idx += 1

        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        sqrt_term = ((glob_center_x - center_x) ** 2) + \
                        ((glob_center_y - center_y) ** 2)
        dist_to_center = math.sqrt(sqrt_term)
        
        if dist_to_center < min_dist:
            min_dist = dist_to_center
            center_kart = valid_kart_idx

        forward_score = center_y
        if y2_scaled < img_height - 1:
            forward_score = (y2_scaled - y1_scaled)

        kart_obj = {
            "instance_id": instance_id,
            "kart_name": kart_names[instance_id],
            "center": (center_x, center_y),
            "forward_score": forward_score, # small weighted factor to simulate depth
            "is_center_kart": False
        }

        kart_list.append(kart_obj)
    
    if kart_list:
        kart_list[center_kart]["is_center_kart"] = True 
  
    return kart_list


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path, 'r') as f:
        info_json = json.load(f)
    
    return info_json["track"]


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """

    track_name = extract_track_info(info_path)
    kart_metadata = extract_kart_objects(info_path,
                                            view_index,
                                            img_width,
                                            img_height,
                                         )        
    
    if not kart_metadata:
        return list()

    center_kart_gen = filter(lambda x: x["is_center_kart"], kart_metadata)
    center_kart = list(center_kart_gen)[0]
    center_kart_x, center_kart_y = center_kart["center"]
    center_kart_forward_score = center_kart["forward_score"]

    # 1. Ego car question
    # What kart is the ego car?
    question_1 = {
        "question": "What kart is the ego car",
        "answer": center_kart["kart_name"]
    }

    # 2. Total karts question
    # How many karts are there in the scenario?
    question_2 = {
        "question": "How many karts are there in the scenario",
        "answer": str(len(kart_metadata))
    }

    # 3. Track information questions
    # What track is this?
    question_3 = {
        "question": "What track is this?",
        "answer": track_name
    }

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    relative_questions = list()
    left_of_ego = 0
    right_of_ego = 0
    back_of_ego = 0
    front_of_ego = 0

    for kart_obj in kart_metadata:
        
        if kart_obj["is_center_kart"]:
            continue

        kart_name = kart_obj["kart_name"]
        kart_x, kart_y = kart_obj["center"]
     

        if kart_x < center_kart_x:
            answer_4 = "left"
            left_of_ego += 1
        else:
            answer_4 = "right"
            right_of_ego += 1

        question_4 = {
            "question": f"Is {kart_name} to the left or right of the ego car?",
            "answer": answer_4
        }

        if kart_y > center_kart_y:
            answer_5 = "back"
            back_of_ego += 1

        else:
            answer_5 = "front"
            front_of_ego += 1
        
        question_5 = {
            "question": f"Is {kart_name} in front of or behind the ego car?",
            "answer": answer_5
        }

        question_6 = {
            "question": f"Where is {kart_name} relative to the ego car?",
            "answer": f"{answer_5} and {answer_4}"
        }

        relative_questions = [
            question_4,
            question_5,
            question_6
        ]


    # 5. Counting questions
    # How many karts are to the left of the ego car?
    question_7 = {
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left_of_ego)
    }

    # How many karts are to the right of the ego car?
    question_8 = {
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right_of_ego)
    }

    # How many karts are in front of the ego car?
    question_9 = {
        "question": "How many karts are in front of the ego car?",
        "answer": str(front_of_ego)
    }

    # How many karts are behind the ego car?
    question_10 = {
        "question": "How many karts are behind the ego car?",
        "answer": str(back_of_ego)
    }

    qa_list = [
        question_1,
        question_2,
        question_3,
        question_7,
        question_8,
        question_9,
        question_10
    ] + relative_questions

    info_hex = info_path.split("/")[-1].replace("_info.json", "")
    for qa in qa_list:

        str(info_path).split
        qa["image_file"] = f"train/{info_hex}_0{view_index}_im.jpg"
    
    return qa_list


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def generate_all_qa_pairs(
    split: str="train"
):

    data_path = Path(__file__).parent.parent / f"data/{split}"
    qa_data = list()

    info_files = list(data_path.glob("*_info.json"))
    for info_file in tqdm.tqdm(info_files):
        for i in range(10):
            qa = generate_qa_pairs(str(info_file), i)
            qa_data += qa

    with open(data_path / f"{split}_qa_pairs.json", "w") as f:
        json.dump(qa_data, f)
    

def main():
    fire.Fire({"check": check_qa_pairs, "generate": generate_all_qa_pairs})


if __name__ == "__main__":
    main()
