from utils import dice_score
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

# IMP
def evaluate(net, data, device, batch_size=4):
    net.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    total_dice = 0.0
    total_loss = 0.0
    num_batches = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating..."):
            images = batch["image"].float().to(device)
            masks = batch["mask"].squeeze(1).long().to(device)  # class index labels: 0 or 1

            outputs = net(images)  # [B, 2, H, W]
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)  # [B, H, W], predicted class index

            # Convert preds and masks to binary masks for dice score
            preds_bin = (preds == 1).float()
            masks_bin = (masks == 1).float()

            batch_dice = dice_score(preds_bin, masks_bin)
            total_dice += batch_dice
            num_batches += 1

    avg_dice = total_dice / num_batches
    avg_loss = total_loss / num_batches
    print("Evaluation Completed.")
    print(f"Validation Dice Score: {avg_dice:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_dice, avg_loss
