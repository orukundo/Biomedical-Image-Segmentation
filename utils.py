# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 07:32:18 2022

Modifications by orukundo@gmail.com Olivier Rukundo
"""
import torch
import torchvision
from dataset import FFGPROJECTDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoint_model.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):
    train_ds = FFGPROJECTDataset(
        trainingImages=train_dir,
        trainingMasks=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = FFGPROJECTDataset(
        trainingImages=val_dir,
        trainingMasks=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_dice_score = 0

    if len(loader) == 0:
        print("Loader is empty!")
        return None, None

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Calculate correct predictions
            correct = (preds == y).sum()
            total_correct += correct

            # Calculate total pixels
            pixels = torch.numel(preds)
            total_pixels += pixels

            # Calculate dice score
            tp = (preds * y).sum()
            fp = (preds * (1 - y)).sum()
            fn = ((1 - preds) * y).sum()
            dice_score = 2 * tp / (2 * tp + fp + fn)
            total_dice_score += dice_score

    if total_pixels == 0:
        print("Number of pixels is zero!")
        accuracy = 0.0
    else:
        accuracy = total_correct / total_pixels * 100

    dice_score = total_dice_score / len(loader)
    print(f"Got {total_correct}/{total_pixels} with acc {accuracy:.2f}")
    print(f"Dice score: {dice_score:.2f}")
    model.train()

    return accuracy, dice_score

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
