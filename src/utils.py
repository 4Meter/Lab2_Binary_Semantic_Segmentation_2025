import torch
import numpy as np

# implement the Dice score here
def dice_score(pred_mask, gt_mask, epsilon=1e-6):
    """
    pred_mask, gt_mask: can be torch.Tensor or numpy.ndarray
    Shape should be [B, 1, H, W]
    """
    if isinstance(pred_mask, np.ndarray):
        pred_mask = torch.from_numpy(pred_mask)
    if isinstance(gt_mask, np.ndarray):
        gt_mask = torch.from_numpy(gt_mask)

    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    gt_flat = gt_mask.view(gt_mask.size(0), -1)

    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()


