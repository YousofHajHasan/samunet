import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset


# ------------------------------------------------------
# Metric functions
# ------------------------------------------------------
def compute_iou(pred, mask):
    pred = (pred > 0.5).float()
    mask = (mask > 0.5).float()
    intersection = (pred * mask).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + mask.sum((1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean().item()


def compute_dice(pred, mask):
    pred = (pred > 0.5).float()
    mask = (mask > 0.5).float()
    intersection = (pred * mask).sum((1, 2, 3))
    dice = (2 * intersection + 1e-7) / (pred.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1e-7)
    return dice.mean().item()


# ------------------------------------------------------
# Main testing
# ------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--test_image_path", type=str, required=True)
parser.add_argument("--test_gt_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 1024)

model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()

os.makedirs(args.save_path, exist_ok=True)

# Running sums
mean_iou = []
mean_dice = []

for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()

        # Convert GT to torch tensor
        gt = np.asarray(gt, np.float32)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

        image = image.to(device)

        res, _, _ = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        pred_tensor = res.sigmoid()

        # ---- Compute metrics before saving ----
        iou = compute_iou(pred_tensor, gt_tensor)
        dice = compute_dice(pred_tensor, gt_tensor)
        mean_iou.append(iou)
        mean_dice.append(dice)

        # ---- Save prediction ----
        pred_np = pred_tensor.cpu().numpy().squeeze()
        pred_np = (pred_np * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.save_path, name.replace(".jpg", ".png")), pred_np)
        print(f"Saved {name} | IoU: {iou:.4f} | Dice: {dice:.4f}")

# ------------------------------------------------------
# Final results
# ------------------------------------------------------
print("\n===== Final Mean Metrics =====")
print(f"Mean IoU:  {np.mean(mean_iou):.4f}")
print(f"Mean Dice: {np.mean(mean_dice):.4f}")
