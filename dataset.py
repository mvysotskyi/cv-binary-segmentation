import os
import torch
from glob import glob

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageOps


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

    # Calculate new dimensions: smallest multiples of target_size that are >= current dimensions
    new_width = ((width + target_size - 1) // target_size) * target_size
    new_height = ((height + target_size - 1) // target_size) * target_size

    # Determine the padding required on each side
    pad_width = new_width - width
    pad_height = new_height - height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    # Select fill color: for RGB images use black (0,0,0), for others assume 0
    fill_color = (0, 0, 0) if img.mode == 'RGB' else 0

    # Pad the image
    padded_img = ImageOps.expand(img, border=(left, top, right, bottom), fill=fill_color)
    return padded_img

import random

def joint_random_crop_and_resize(crop_size, resize_size, flip_prob=0.5):
    def transform(image, mask):
        # Random scaling (between 0.9 and 1.1)
        # if random.random() < 0.5:
        #     scale_factor = 1.0 + 0.1 * torch.rand(1).item()  # Random value between 0.9 and 1.1
        #     new_size = [int(s * scale_factor) for s in image.size]
        #     image = F.resize(image, new_size, Image.BILINEAR)
        #     mask = F.resize(mask, new_size, Image.NEAREST)
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        # Random horizontal flip
        if torch.rand(1).item() < flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Resize
        image = F.resize(image, resize_size, Image.BILINEAR)
        mask = F.resize(mask, resize_size, interpolation=Image.NEAREST)
        return image, mask
    return transform


class SegmentationDataset(Dataset):
    default_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    default_mask_transforms = transforms.ToTensor()

    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*')))
        self.joint_transform = joint_random_crop_and_resize(crop_size=(224, 224), resize_size=(224, 224))
        self.transform = __class__.default_image_transforms if transform is None else transform
        self.mask_transform = __class__.default_mask_transforms if mask_transform is None else mask_transform
        self.train = True
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch between number of images and masks."

    def __len__(self):
        return len(self.image_paths)
    
    def set_train(self, train=True):
        self.train = train

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        mask = pad_to_divisible(mask, 224)
        image = pad_to_divisible(image, 224)

        if self.train:
            image, mask = self.joint_transform(image, mask)

        image = self.transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask
