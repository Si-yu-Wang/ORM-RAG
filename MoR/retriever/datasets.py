import torch
import json
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset
import hashlib

from model.constants import VIT_MEAN, VIT_STD, IMAGENET_MEAN, IMAGENET_STD

class BBImageDataset(Dataset):
    def __init__(self,json_path,k=-1):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)  # 读取JSON文件
        self.k=k

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.k==-1:
            images = sample["image"]
        else:
            images = sample["image"][self.k]
        images = self.get_unique_image_list(images)

        if "label" in sample:
            label = sample["label"]
        else:
            raise KeyError("label not found in sample")

        report = sample["conversations"][1]["value"]
        image_tensors = []

        transform = T.Compose([
            T.Resize(384),
            T.CenterCrop(384),
            T.ToTensor(),
            T.Normalize(
                mean=VIT_MEAN,
                std=VIT_STD,
            ),
        ])
        for img_path in images:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = transform(img)
            image_tensors.append(img)

        image_tensor = torch.stack(image_tensors)  # shape: [k, 3, 384, 384]

        return image_tensor, label, images, report
    
    def get_unique_image_list(self, image_list):
        """
        Removes duplicate images from a list based on their content using MD5 hash.

        Args:
            image_list (list of str): List of image file paths.

        Returns:
            list of str: List of unique image file paths, where duplicate images (by content) are removed.

        Usage:
            unique_images = self.get_unique_image_list(['/path/to/img1.jpg', '/path/to/img2.jpg', ...])

        Note:
            - Images that cannot be read will be skipped with a warning.
            - Duplicate detection is based on the MD5 hash of the image file content, not the file name.
        """
        seen_hashes = set()
        unique_image_list = []

        for img_path in image_list:
            try:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
            except Exception as e:
                print(f"无法读取 {img_path}，已跳过：{e}")
                continue

            # 计算 MD5 指纹
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                # 已处理过同内容图片
                continue
            seen_hashes.add(img_hash)
            unique_image_list.append(img_path)

        return unique_image_list