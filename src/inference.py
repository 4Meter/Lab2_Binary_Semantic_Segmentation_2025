import argparse
import os
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from oxford_pet import SimpleOxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_type', type=str, default='Unet',help='Unet or Resnet34_Unet')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--visualize', action='store_true', help='visualize predictions')
    return parser.parse_args()

def inference(net, data, device, batch_size=4, visualize=False, model_path=None):
    net.to(device)
    net.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    total_dice = 0.0
    total_images = 0

    vis_images, vis_preds, vis_trues = [], [], []

    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing...")
        for batch in pbar:
            images = batch["image"].float().to(device)
            masks = batch["mask"].squeeze(1).long().to(device)  # class index labels: 0 or 1

            outputs = net(images)  # [B, 2, H, W]
            preds = torch.argmax(outputs, dim=1)  # [B, H, W], predicted class index

            preds_bin = (preds == 1).float()
            masks_bin = (masks == 1).float()

            batch_dice = dice_score(preds_bin, masks_bin)
            total_dice += batch_dice
            total_images += images.size(0)

            avg_dice = total_dice / (total_images / batch_size)
            pbar.set_postfix(avg_dice=f"{avg_dice:.4f}", images=total_images)

            if visualize and len(vis_images) < 10:
                remaining = 10 - len(vis_images)
                vis_images.extend(images[:remaining].cpu())
                vis_preds.extend(preds_bin[:remaining].cpu().unsqueeze(1))
                vis_trues.extend(masks_bin[:remaining].cpu().unsqueeze(1))

    avg_dice = total_dice / (total_images / batch_size)

    if visualize:
        vis_filename = f"checkpoints/vis_result_{timestamp}.png"
        print(f"Saved visualization: {vis_filename}")
        visualize_batch(vis_images, vis_preds, vis_trues, save_path=vis_filename, also_show=True)

    # Save result to CSV
    model_name = os.path.basename(model_path)
    model_type = type(net).__name__
    csv_path = "checkpoints/inference_result.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["time", "model_type", "model_file", "avg_dice"])
        writer.writerow([timestamp, model_type, model_name, f"{avg_dice:.4f}"])

    print("Inference Completed.")
    print(f"Test Dice Score: {avg_dice:.4f}")
    return avg_dice

def visualize_batch(images, preds, trues, save_path=None, also_show=False):
    fig, axs = plt.subplots(3, 10, figsize=(20, 6), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    row_titles = ["Image", "Prediction", "Ground Truth"]

    for row in range(3):
        for col in range(10):
            if row == 0:
                img = images[col].permute(1, 2, 0).numpy().astype(np.uint8)
                axs[row, col].imshow(img)
            elif row == 1:
                pred_mask = preds[col][0].numpy()
                axs[row, col].imshow(pred_mask, cmap='gray')
            else:
                true_mask = trues[col][0].numpy()
                axs[row, col].imshow(true_mask, cmap='gray')
            axs[row, col].axis('off')

    for row, title in enumerate(row_titles):
        fig.text(0.02, 0.82 - row * 0.31, title, va='center', ha='left', fontsize=14, rotation=90)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
    if save_path:
        plt.savefig(save_path)
    if also_show:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    args = get_args()

    # device select
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'The used device is {device}')

    dataset = SimpleOxfordPetDataset(args.data_path, mode='test')

    if args.model_type == 'Unet':
        model = UNet(c_in=3, c_out=2)
    elif args.model_type == 'Resnet34_Unet':
        model = ResNet34_UNet(c_in=3, c_out=2)
    else:
        print(f"model: {args.model_type} not available. Do you mean Unet or Resnet34_Unet?")

    model.load_state_dict(torch.load(args.model, map_location=device))

    avg_dice = inference(model, dataset, device, batch_size=args.batch_size, visualize=args.visualize, model_path=args.model)
