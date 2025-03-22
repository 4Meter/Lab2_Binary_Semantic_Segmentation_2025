from utils import dice_score
import torch
from torch.utils.data import DataLoader

def evaluate(net, data, device):
    # implement the evaluation function here
    net.eval()
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)

            outputs = net(images)
            preds = torch.sigmoid(outputs)  # sigmoid
            preds = (preds > 0.5).float()   # binarization

            batch_dice = dice_score(preds, masks)
            total_dice += batch_dice
            num_batches += 1

    avg_dice = total_dice / num_batches
    print("Evaluation Completed.")
    print(f"Validation Dice Score: {avg_dice:.4f}")
    return avg_dice
