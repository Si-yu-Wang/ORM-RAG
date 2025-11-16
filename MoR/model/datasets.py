import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import hashlib

class ImageDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        images = sample["image"]#[:2]
        images = self.get_unique_image_list(images)
        if len(images) > 20:
            images = random.sample(images, 20)
        if "label" in sample:
            label = sample["label"]
        else:
            label = -1
        # label = sample["label"]
        if label != -1:
            label = label % 2

        image_tensors = []
        for img_path in images:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
            image_tensors.append(img)

        image_tensor = torch.stack(image_tensors)  # 形状: [num_images, C, H, W]   [13, 3, 384, 384]
        return image_tensor, label, images
    
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

            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)
            unique_image_list.append(img_path)

        return unique_image_list
    
def collate_fn(batch):
    """
    Custom collate function for batching variable-length image sequences.

    Args:
        batch (list of tuples): Each tuple contains (imgs, label, _) where:
            - imgs (Tensor): A tensor of shape (N, C, H, W), where N is the number of images in the sequence,
              C is the number of channels, H and W are height and width.
            - label (int): The label associated with the sequence.
            - _ : Placeholder for any additional data (ignored).

    Returns:
        padded (Tensor): Batched images, zero-padded to the maximum sequence length in the batch.
            Shape: (B, N_max, C, H, W), where B is batch size and N_max is the max sequence length.
        masks (Tensor): Boolean mask indicating valid image positions (True for valid, False for padded).
            Shape: (B, N_max)
        labels (Tensor): Tensor of labels for each sequence in the batch.
            Shape: (B,)

    Usage:
        Use this function as the `collate_fn` argument in a PyTorch DataLoader to handle batches
        of variable-length image sequences. Example:

            dataloader = DataLoader(dataset, batch_size=..., collate_fn=collate_fn)
    """
    imgs_list, labels, _ = zip(*batch)
    B = len(imgs_list)
    C, H, W = imgs_list[0].shape[1:]
    N_max = max(img.shape[0] for img in imgs_list)

    padded = torch.zeros((B, N_max, C, H, W))
    masks  = torch.zeros((B, N_max), dtype=torch.bool)
    for i, imgs in enumerate(imgs_list):
        n = imgs.shape[0]
        padded[i, :n] = imgs
        masks[i, :n]  = 1

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, masks, labels