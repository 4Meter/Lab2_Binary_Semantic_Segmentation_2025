import argparse

import torch
from torch.utils.data import DataLoader
from oxford_pet import SimpleOxfordPetDataset, load_dataset
from models.unet import UNet

def train(args):
    # IMP
    # implement the training function here
    train_dataset = load_dataset(args.data_path, mode='train')
    val_dataset = load_dataset(args.data_path, mode='valid')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型、損失函數與優化器設定（此處僅放範例 placeholder）
    model = UNet(n_channels=3, n_classes=1) # TODO: 換成 UNet 或其他模型
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'The used device is {device}')
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss.item():.4f}")

    print("Training complete!")

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