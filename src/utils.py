import torch
import numpy as np

# implement the Dice score here
def dice_score(pred_mask, gt_mask, epsilon=1e-6):
    """
    pred_mask, gt_mask: binary masks (float), shape [B, H, W] or [B, 1, H, W]
    """
    if isinstance(pred_mask, np.ndarray):
        pred_mask = torch.from_numpy(pred_mask)
    if isinstance(gt_mask, np.ndarray):
        gt_mask = torch.from_numpy(gt_mask)

    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    # if has shape [B, 1, H, W], squeeze channel dim
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)

    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    gt_flat = gt_mask.view(gt_mask.size(0), -1)

    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

import albumentations as A
from albumentations.pytorch import ToTensorV2

# transform wrapper class
class SegmentationTransform:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GridDistortion(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def __call__(self, image, mask, trimap=None):
        augmented = self.transform(image=image, mask=mask) # Only transform
        image = augmented['image']          # [C, H, W]
        mask = augmented['mask']            # [H, W]
        return {"image": image, "mask": mask, "trimap": trimap}