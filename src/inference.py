import argparse
import os
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from oxford_pet import SimpleOxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_type', type=str, default='Unet', help='Unet or Resnet34_Unet')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--visualize', action='store_true', help='visualize predictions'),
    parser.add_argument('--no_cbam', action='store_true', help='Disable CBAM in decoder blocks')
    return parser.parse_args()

def inference(net, data, device, batch_size=4, visualize=False, model_path=None):
    net.to(device)
    net.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    total_dice = 0.0
    total_images = 0

    vis_images, vis_preds, vis_trues = [], [], []

    checkpoint_dir = os.path.join("checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
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
                vis_preds.extend(preds_bin[:remaining].cpu())
                vis_trues.extend(masks_bin[:remaining].cpu())

    avg_dice = total_dice / (total_images / batch_size)

    if visualize:
        vis_filename = os.path.join(checkpoint_dir, f"vis_result_{timestamp}.png")
        csv_path = os.path.join(checkpoint_dir, "inference_result.csv")
        print(f"Saved visualization: {vis_filename}")
        visualize_overlay(vis_images, vis_preds, vis_trues, save_path=vis_filename, also_show=True)

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

def visualize_overlay(images, preds, trues, save_path=None, also_show=False):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    red_cmap = cm['Reds'].copy()
    green_cmap = cm['Greens'].copy()
    red_cmap.set_under(alpha=0.0)
    green_cmap.set_under(alpha=0.0)

    for i in range(10):
        img = images[i].permute(1, 2, 0).numpy().astype(np.uint8)
        pred_mask = preds[i].numpy()
        true_mask = trues[i].numpy()

        axs[i].imshow(img)
        axs[i].imshow(true_mask, cmap=green_cmap, alpha=0.8, vmin=0.01, vmax=1.0)  # Green for ground truth
        axs[i].imshow(pred_mask, cmap=red_cmap, alpha=0.5, vmin=0.01, vmax=1.0)    # Red for prediction
        axs[i].axis('off')
        axs[i].set_title(f"Sample {i+1}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if also_show:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    args = get_args()

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
        use_cbam = not args.no_cbam
        model = ResNet34_UNet(c_in=3, c_out=2, use_cbam=use_cbam)
    else:
        print(f"model: {args.model_type} not available. Do you mean Unet or Resnet34_Unet?")
        exit()

    model.load_state_dict(torch.load(args.model, map_location=device))

    avg_dice = inference(model, dataset, device, batch_size=args.batch_size, visualize=args.visualize, model_path=args.model)