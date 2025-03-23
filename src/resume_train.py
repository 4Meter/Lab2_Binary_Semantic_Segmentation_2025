import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from utils import SegmentationTransform, dice_score
from oxford_pet import SimpleOxfordPetDataset


def resume_training(args):
    # Prepare dataset and dataloader
    transform = SegmentationTransform()
    train_data = SimpleOxfordPetDataset(args.data_path, mode="train", transform=transform)
    val_data = SimpleOxfordPetDataset(args.data_path, mode="valid")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    if args.model_type == 'Unet':
        model = UNet(c_in=3, c_out=2)
    elif args.model_type == 'Resnet34_Unet':
        model = ResNet34_UNet(c_in=3, c_out=2)
    else:
        print(f"model: {args.model_typel} not available. Do you mean Unet or Resnet34_Unet?")
        return

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # device select
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'The used device is {device}')
    
    model.to(device)
    # Load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("No checkpoint found. Exiting.")
        return

    
    best_dice = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    # Initialize history
    history = {
        "train_loss": [],
        "val_dice": [],
        "val_loss": [],
        "lr": []
    }
    
    # Resume training loop
    for epoch in range(args.resume_epoch, args.resume_epoch + args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            images = batch["image"].float().to(device)
            masks = batch["mask"].squeeze(1).long().to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(avg_loss)
        print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # Validation
        val_dice, val_loss = evaluate(model, val_data, device, args.batch_size)
        scheduler.step(val_dice)
        # Epoch result saving
        history["train_loss"].append(avg_loss)
        history["val_dice"].append(val_dice)
        history["val_loss"].append(val_loss)
        history["Last Epoch"] = epoch+1

        # Model saving + Early stopping
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"saved_models/{args.model_type}_best_model.pth")
            print("Model improved. Saved new best model.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    history["Best Dice Score"] = best_dice
    history["Best Epoch"] = best_epoch
    history["Start Epoch"] = args.resume_epoch
    history["batch_size"] = args.batch_size
    history["Model"] = args.model_type
    # Saving history to JSON
    with open(f"checkpoints/{args.model_type}_training_history.json", "w") as f:
        json.dump(history, f, indent=4) 
           
    print("Training complete!")
    print(f"Best Dice Score: {best_dice:.4f} at Epoch {best_epoch}")
    
    return history, best_dice, best_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/oxford-iiit-pet')
    parser.add_argument('--model_type', type=str, default='Unet',help='Unet or Resnet34_Unet')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=20, help='Start epoch index')
    parser.add_argument('--epochs', type=int, default=10, help='Number of additional epochs to train')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    history, best_dice, best_epoch = resume_training(args)
