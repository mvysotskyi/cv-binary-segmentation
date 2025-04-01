import torch

import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BeitImageProcessor

from dataset import SegmentationDataset, joint_transforms_test, joint_transforms_train, prepare_data, pad_to_divisible

from torchvision import transforms

from beit_seg import BeitMLASegmentation
from copy import deepcopy
from PIL import Image



def get_dataloader(image_dir, mask_dir, processor, batch_size=4):
    num_workers = batch_size

    train_pairs, test_pairs = prepare_data(image_dir, mask_dir)
    train_dataset = SegmentationDataset(train_pairs, processor, joint_pil_transform=joint_transforms_train((384, 384)))
    test_dataset = SegmentationDataset(test_pairs, processor, joint_pil_transform=joint_transforms_test((384, 384)))
    test_dataset_full = SegmentationDataset(test_pairs, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_full = DataLoader(test_dataset_full, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, test_loader_full

def get_model_beit(device: str = "cuda"):
    model = BeitMLASegmentation(num_classes=1, model_name="microsoft/beit-base-patch16-384")
    model = model.to(device)
    model.model.requires_grad_(False)
    return model

def dice_coef(pred, target, smooth=1e-7):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice


def sliding_window_inference(image_path, model, processor, patch_size=384, stride=192, device='cuda'):
    image = Image.open(image_path[0]).convert("RGB")
    image, _ = pad_to_divisible(image, patch_size)
    
    # Correct assignment: width (W) and height (H)
    W, H = image.size  # PIL returns (width, height)
    
    # Create the output mask with shape (1, 1, H, W)
    output_mask = torch.zeros((1, 1, H, W), device=device)
    weight_matrix = torch.zeros((1, 1, H, W), device=device)
    
    patches = []
    coords = []
    
    # Loop over the image using y for the vertical (height) dimension and x for horizontal (width)
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Crop using (left, upper, right, lower)
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patch = processor(images=patch, return_tensors="pt")["pixel_values"].to(device)
            patches.append(patch)
            coords.append((x, y))
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = torch.cat(patches[i:i+batch_size], dim=0)
            batch_preds = torch.sigmoid(model({"pixel_values": batch_patches}))
            for j, pred in enumerate(batch_preds):
                x, y = coords[i + j]
                # Note: in the tensor, dimension 2 is height and dimension 3 is width
                output_mask[:, :, y:y+patch_size, x:x+patch_size] += pred.unsqueeze(0)
                weight_matrix[:, :, y:y+patch_size, x:x+patch_size] += 1

    output_mask /= torch.clamp(weight_matrix, min=1)
    final_mask = (output_mask > 0.5)
    
    return final_mask


def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()


def train(model, train_loader, test_loader, test_loader_full):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = dice_loss
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_dice = 0.0

    # Training loop
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for _, images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            alpha = 0.01
            masks = (1 - alpha) * masks + alpha / 2

            outputs = model({"pixel_values": images})
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        scheduler.step()

        model.eval()
        test_loss = 0.0
        dice = 0.0
        with torch.no_grad():
            for _, images, masks, _ in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model({"pixel_values": images})
                loss = criterion(outputs, masks)
                test_loss += loss.item() * images.size(0)
                dice += dice_coef(outputs, masks).item() * images.size(0)

        test_loss = test_loss / len(test_loader.dataset)        
        epoch_loss = running_loss / len(train_loader.dataset)
        dice = dice / len(test_loader.dataset)

        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     print("Saving best model..., loss:", best_loss)
        #     best_model = deepcopy(model.state_dict())
        #     torch.save(best_model, "best_model.pth")

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Dice: {dice:.4f}")

        # dice = 0.0
        # print("Calculating Dice Coefficient on Test Set...")

        # with torch.no_grad():
        #     for images_paths, _, masks, coords in test_loader_full:
        #         # images = images.to(device)
        #         masks = masks.to(device)

        #         # print(masks.shape, images.shape)

        #         outputs = sliding_window_inference(images_paths, model, processor, patch_size=384, stride=192, device=device)
        #         # print(outputs.shape, masks.shape)
        #         x_start, y_start, x_end, y_end = coords
        #         masks_cut = masks[:, :, y_start:y_end, x_start:x_end].contiguous()
        #         outputs_cut = outputs[:, :, y_start:y_end, x_start:x_end].contiguous()

        #         dice += dice_coef(outputs_cut, masks_cut).item() * masks.size(0)

        # dice /= len(test_loader.dataset)
        # print("Dice Coefficient:", dice)

        # if dice > best_dice:
        #     best_dice = dice
        #     print("Saving best model..., Dice Coefficient:", best_dice)
        #     best_model = deepcopy(model.state_dict())
        #     torch.save(best_model, f"best_model_{dice:.4f}.pth")
    # print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {8:.4f}, Dice Coefficient: {dice:.4f}")


if __name__ == "__main__":
    image_dir = "seg-dataset/images"
    mask_dir = "seg-dataset/masks"
    batch_size = 16

    processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-384")
    train_loader, test_loader, test_loader_full = get_dataloader(image_dir, mask_dir, processor, batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model_beit(device)

    train(model, train_loader, test_loader, test_loader_full)