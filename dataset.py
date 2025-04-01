import os
import torch
import random
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
from torchvision import transforms
from transformers import BeitImageProcessor


def prepare_data(images_dir, masks_dir, split_ratio=0.9, seed=42):
    """
    Creates pairs of images and masks and splits them into train and validation sets.
    
    Args:
        images_dir (str): Directory containing input images
        masks_dir (str): Directory containing mask images
        split_ratio (float): Ratio for train/validation split (default 0.8)
    
    Returns:
        tuple: Lists of train and validation pairs
    """
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    pairs = list(zip(
        [os.path.join(images_dir, img) for img in image_files],
        [os.path.join(masks_dir, mask) for mask in mask_files]
    ))
    
    # Random shuffle
    random.seed(seed)
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    return train_pairs, val_pairs


def pad_to_divisible(img, target_size):
    """
    Pad the image with zeros so that both width and height become divisible by target_size.
    
    Args:
        img (PIL.Image.Image): Input image.
        target_size (int): The divisor for the output dimensions.
    
    Returns:
        PIL.Image.Image: The padded image.
    """
    width, height = img.size

    new_width = ((width + target_size - 1) // target_size) * target_size
    new_height = ((height + target_size - 1) // target_size) * target_size

    pad_width = new_width - width
    pad_height = new_height - height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    coords = (left, top, left + width, top + height)
    fill_color = (0, 0, 0) if img.mode == 'RGB' else 0

    padded_img = ImageOps.expand(img, border=(left, top, right, bottom), fill=fill_color)
    return padded_img, coords

def joint_transforms_train(crop_size, flip_prob=0.5):
    def transform(image, mask):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        if torch.rand(1).item() < flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        return image, mask
    return transform

def joint_transforms_test(crop_size):
    def transform(image, mask):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return image, mask
    return transform

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data_pairs, processor: BeitImageProcessor, joint_pil_transform=None):
        self.data_pairs = data_pairs
        self.joint_pil_transform = joint_pil_transform
        self.processor = processor

    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        image_path, mask_path = self.data_pairs[idx]
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, _ = pad_to_divisible(image, 384)
        mask, coords = pad_to_divisible(mask, 384)
        
        if self.joint_pil_transform:
            image, mask = self.joint_pil_transform(image, mask)

        image = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        mask = transforms.ToTensor()(mask)
        
        # if image.shape[2:] != mask.shape[2:]:
        #     image = F.resize(image, mask.shape[2:], Image.BILINEAR)

        return image_path, image, mask, coords