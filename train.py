import torch
import torch.nn as nn

import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models

from dataset import SegmentationDataset

from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as F

from unet import UNet
from beit_seg import BeitMLASegmentation
# from dataset import transform
from copy import deepcopy

def get_dataloaders(image_dir, mask_dir, batch_size=4):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    num_workers = batch_size // 2
    full_dataset = SegmentationDataset(image_dir, mask_dir, transform=image_transforms)

    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataset.dataset.set_train(True)
    test_dataset.dataset = deepcopy(train_dataset.dataset)
    test_dataset.dataset.set_train(False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# def get_fcn_model(device: str = "cuda"):
#     model = models.segmentation.fcn_resnet101(pretrained=False, num_classes=1, aux_loss=True)
#     return model.to(device)

def get_fcn_model(device: str = "cuda"):
    model = models.segmentation.fcn_resnet101(pretrained=True, aux_loss=True)
    # model.requires_grad_(False)

    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    model.to(device)
    return model

def get_model_unet(device: str = "cuda"):
    model = UNet(in_channels=3, num_classes=1, depth=5)
    model.to(device)
    return model

def get_model_beit(device: str = "cuda"):
    model = BeitMLASegmentation(num_classes=1, model_name="microsoft/beit-base-patch16-224-pt22k")
    model = model.to(device)
    model.model.requires_grad_(False)
    return model

def call_beit(model, x):
    # x = 2 * (x - 0.5)
    # print(x.shape)
    # Scale the input to the range [0, 1]
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    # print(x.shape)
    inp = {"pixel_values": x}
    x = model(inp)

    return x


def dice_coef(pred, target, smooth=1e-7):
    """
    Computes the Dice coefficient.
    Args:
        pred (torch.Tensor): Predictions from the model (logits).
        target (torch.Tensor): Ground truth binary masks.
        smooth (float): Smoothing constant to avoid division by zero.
    Returns:
        dice (float): Dice coefficient.
    """
    # Apply sigmoid and threshold the predictions to obtain binary mask
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice


def sliding_window_inference(image, model, patch_size=256, stride=128, device='cuda'):
    """
    Perform patch-based inference on an image using a segmentation model with batching support.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        model (torch.nn.Module): Trained PyTorch segmentation model.
        patch_size (int): Size of each patch.
        stride (int): Step size for moving the window.
        device (str): Computation device ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: Segmentation mask of the same spatial dimensions as the input image.
    """
    # Get image dimensions
    H, W = image.shape[2], image.shape[3]
    output_mask = torch.zeros((1, 1, H, W), device=device)
    weight_matrix = torch.zeros((1, 1, H, W), device=device)

    # List to hold patches and their coordinates
    patches = []
    coords = []

    # Extract patches and store their top-left coordinates
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, :, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
    
    # Process patches in batches of 32
    batch_size = 32
    # model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = torch.cat(patches[i:i+batch_size], dim=0)
            # Get model predictions and apply sigmoid
            batch_preds = torch.sigmoid(model(batch_patches))
            # Iterate over predictions and add them back to the output mask
            for j, pred in enumerate(batch_preds):
                y, x = coords[i + j]
                output_mask[:, :, y:y+patch_size, x:x+patch_size] += pred.unsqueeze(0)
                weight_matrix[:, :, y:y+patch_size, x:x+patch_size] += 1

    # Average the overlapping regions
    output_mask /= torch.clamp(weight_matrix, min=1)
    final_mask = (output_mask > 0.5)

    return final_mask



def train(model, train_loader, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Training loop
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # images, masks = transform(images, masks)

            alpha = 0.01
            masks = (1 - alpha) * masks + alpha / 2

            # outputs = model(images)['out']
            outputs = call_beit(model, images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Evaluate on the test set
        model.eval()
        # test_loss = 0.0
        dice = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                # print(images.shape, masks.shape)
                # images, masks = transform(images, masks)
                if masks.shape[2:] != images.shape[2:]:
                    # outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                    continue

                # outputs = model(images)['out']
                # outputs = call_beit(model, images)
                # loss = criterion(outputs, masks)
                # test_loss += loss.item() * images.size(0)
                outputs = sliding_window_inference(images, lambda x: call_beit(model, x), patch_size=224, stride=112, device=device)
                # print(outputs.shape)


                dice += dice_coef(outputs, masks).item() * images.size(0)
        # test_loss /= len(test_loader.dataset)
        dice /= len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {8:.4f}, Dice Coefficient: {dice:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    image_dir = "seg-dataset/images"
    mask_dir = "seg-dataset/masks"
    batch_size = 16

    train_loader, test_loader = get_dataloaders(image_dir, mask_dir, batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model_beit(device)
    

    train(model, train_loader, test_loader)