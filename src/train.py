import argparse
import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from oxford_pet import load_dataset
from models.unet import UNet
from evaluate import evaluate
from utils import SegmentationTransform

def train(args):
    # IMP
    # implement the training function here
    transform = SegmentationTransform()
    train_dataset = load_dataset(args.data_path, mode='train',transform=transform)
    val_dataset = load_dataset(args.data_path, mode='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 模型、損失函數與優化器設定
    model = UNet(c_in=3, c_out=2) # TODO: 換成 UNet 或其他模型
    criterion = torch.nn.CrossEntropyLoss()
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
    
    best_dice = 0.0
    best_epoch = 0
    patience = 3
    patience_counter = 0

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    # Initialize history
    history = {
        "train_loss": [],
        "val_dice": []
    }

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].float().to(device)
            masks = batch["mask"].squeeze(1).long().to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Training Loss: {avg_loss:.4f}")

        # Validation
        val_dice = evaluate(model, val_dataset, device, args.batch_size)
        scheduler.step(val_dice)
        
        # Epoch result saving
        history["train_loss"].append(avg_loss)
        history["val_dice"].append(val_dice)

        # Model saving + Early stopping
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            print("Model improved. Saved new best model.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Saving history to JSON
    with open("checkpoints/training_history.json", "w") as f:
        json.dump(history, f, indent=4)  
        
    print("Training complete!")
    print(f"Best Dice Score: {best_dice:.4f} at Epoch {best_epoch}")
    return history, best_dice, best_epoch

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='dataset/oxford-iiit-pet',help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)