from PIL import Image
import torch
from torchvision import transforms
from .constants import VIT_MEAN, VIT_STD, IMAGENET_MEAN, IMAGENET_STD
from typing import Tuple
from .constants import BASE_OUTPUT, BASE_TRAIN_DATA, _GROUP_NAMES

def process_images(image_list):
    """
    Processes a list of image file paths by loading, converting to RGB if necessary, 
    applying a series of transformations (resize, center crop, normalization), 
    and stacking them into a single tensor suitable for model input.

    Args:
        image_list (list of str): List of file paths to the images to be processed.

    Returns:
        torch.Tensor: A 4D tensor of shape (1, N, C, H, W), where N is the number of images,
                      C is the number of channels (3 for RGB), and H, W are the height and width (384).
    """
    rag_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=VIT_MEAN,
            std=VIT_STD
        ),
    ])
    image_tensors=[]
    # print("imagelist",image_list)
    for img_path in image_list:
        # print("img_path",img_path)
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")  # 转换为 RGB 图像
        img = rag_transform(img)
        image_tensors.append(img)

        del img

        # 将多个图像的特征合并（平均池化）
    image_tensor = torch.stack(image_tensors)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def judge_label(judge_data, images):
    """
    Categorizes images based on their labels from the provided judge_data.

    Args:
        judge_data (list of dict): A list where each dict contains:
            - 'label': The label/category for a group of images.
            - 'image': A list of image paths associated with the label.
        images (list of list): A list of images to be judged, where each image is wrapped in a list (e.g., [[img1], [img2], ...]).

    Returns:
        tuple:
            - list: Unique labels found for the given images.
            - dict: A dictionary mapping each label to a list of image paths that belong to that label.

    Usage Example:
        judge_data = [
            {'label': 'cat', 'image': ['cat1.jpg', 'cat2.jpg']},
            {'label': 'dog', 'image': ['dog1.jpg']}
        ]
        images = [['cat1.jpg'], ['dog1.jpg']]
        labels, result = judge_label(judge_data, images)
        # labels -> ['cat', 'dog']
        # result -> {'cat': ['cat1.jpg'], 'dog': ['dog1.jpg']}
    """
    # images=flatten(images)
    result = {}
    labels = []
    for image in images:
        for entry in judge_data:
            if image in entry['image']:
                label = entry['label']
                labels.append(entry['label'])
                if label not in result:
                    result[label] = []
                result[label].append(image)
                break
    return list(set(labels)), result

def get_root_dir(label_id: int) -> Tuple[str, str]:
    """
    Retrieves the output directory and JSON data path for a given label ID.
    Args:
        label_id (int): The integer identifier for the label group.
    Returns:
        Tuple[str, str]: A tuple containing:
            - output_dir (str): The path to the output directory for the group.
            - json_path (str): The path to the JSON data file for the group.
    Raises:
        FileNotFoundError: If the provided label_id does not exist in _GROUP_NAMES.
    Usage:
        output_dir, json_path = get_root_dir(3)
    """
    try:
        group = _GROUP_NAMES[label_id]
    except KeyError:
        raise FileNotFoundError(f"Unknown label_id: {label_id!r}")
    
    output_dir = f"{BASE_OUTPUT}/{group}"
    json_path  = f"{BASE_TRAIN_DATA}/{group}.json"
    return output_dir, json_path