# Import necessary libraries and modules
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Define hyperparameters and settings
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use CUDA (GPU) if available, else use CPU
BATCH_SIZE = 2  # Batch size for training
NUM_EPOCHS = 100  # Number of training epochs
NUM_WORKERS = 2  # Number of data loading workers
IMAGE_HEIGHT = 512  # Desired height of input images
IMAGE_WIDTH = 512  # Desired width of input images
PIN_MEMORY = True  # Use pinned memory for faster data transfer
LOAD_MODEL = False  # Whether to load a pre-trained model
TRAIN_IMG_DIR = "C:/Users/../trainingImages"  # Directory for training images
TRAIN_MASK_DIR = "C:/Users/../trainingMasks"  # Directory for training masks
VAL_IMG_DIR = "C:/Users/../validationImages"  # Directory for validation images
VAL_MASK_DIR = "C:/Users/../validationMasks"  # Directory for validation masks

# Define the training function
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update the tqdm progress bar with the current loss
        loop.set_postfix(loss=loss.item())

# Define the main function for training
def main():
    # Define data augmentation and transformation pipelines for training and validation
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Initialize the UNet model, loss function, and optimizer
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer

    # Create data loaders for training and validation datasets
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    # Occasionally, load a pre-trained model checkpoint
    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint_model.tar"), model)

    # Evaluate model accuracy on the validation dataset
    check_accuracy(val_loader, model, device=DEVICE)

    # Initialize the gradient scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop over multiple epochs
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save the current model checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Evaluate model accuracy on the validation dataset
        check_accuracy(val_loader, model, device=DEVICE)
        
        with open("check_accuracy_values.txt", "a") as f:
         check_accuracy_value = check_accuracy(val_loader, model, device=DEVICE)
         f.write(f"Epoch {epoch + 1}: {check_accuracy_value}\n")

        # Save some example predictions to a folder
        output_folder = "saved_images/"
        os.makedirs(output_folder, exist_ok=True)
        save_predictions_as_imgs(val_loader, model, output_folder, device=DEVICE)
    
# Entry point of the script
if __name__ == "__main__":
    main()  # Call the main function to start training
