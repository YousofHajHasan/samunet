import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset
from typing import Dict, Tuple


def compute_detailed_metrics(pred: torch.Tensor, 
                             mask: torch.Tensor, 
                             num_classes: int,
                             class_names: list = None) -> Dict:
    """
    Compute detailed IoU and Dice metrics for a single image with per-class breakdown.
    
    Args:
        pred: [B, num_classes, H, W] - model logits/predictions
        mask: [B, H, W] - ground truth class indices
        num_classes: int - number of classes
        class_names: list - optional list of class names (e.g., ['Background', 'C1', 'C2', ...])
    
    Returns:
        Dictionary with detailed metrics
    """
    
    # Convert logits to class predictions
    pred_classes = torch.argmax(pred, dim=1)  # [B, H, W]
    
    # Remove batch dimension for single image analysis
    pred_classes = pred_classes.squeeze(0)  # [H, W]
    mask = mask.squeeze(0)  # [H, W]
    
    # Convert to numpy for easier analysis
    pred_np = pred_classes.cpu().numpy()
    mask_np = mask.cpu().numpy()
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' if i > 0 else 'Background' for i in range(num_classes)]
    
    # Initialize results
    results = {
        'overall': {},
        'per_class': {},
        'confusion_matrix': {},
        'pixel_counts': {}
    }
    
    # Get image dimensions
    H, W = pred_np.shape
    total_pixels = H * W
    
    # ========================================================================
    # Per-Class Metrics
    # ========================================================================
    class_ious = []
    class_dices = []
    
    print("=" * 80)
    print("DETAILED METRICS ANALYSIS")
    print("=" * 80)
    print(f"Image size: {H} x {W} = {total_pixels:,} pixels\n")
    
    print("-" * 80)
    print(f"{'Class':<15} {'IoU':<10} {'Dice':<10} {'TP':<10} {'FP':<10} {'FN':<10} {'GT':<10} {'Pred':<10}")
    print("-" * 80)
    
    for cls in range(num_classes):
        # Get binary masks for current class
        pred_cls = (pred_np == cls).astype(np.float32)
        mask_cls = (mask_np == cls).astype(np.float32)
        
        # True Positives, False Positives, False Negatives
        tp = np.sum((pred_cls == 1) & (mask_cls == 1))
        fp = np.sum((pred_cls == 1) & (mask_cls == 0))
        fn = np.sum((pred_cls == 0) & (mask_cls == 1))
        tn = np.sum((pred_cls == 0) & (mask_cls == 0))
        
        # Ground truth and prediction pixel counts
        gt_pixels = np.sum(mask_cls)
        pred_pixels = np.sum(pred_cls)
        
        # Compute IoU
        intersection = tp
        union = gt_pixels + pred_pixels - intersection
        iou = (intersection + 1e-7) / (union + 1e-7) if union > 0 else 0.0
        
        # Compute Dice
        dice = (2.0 * intersection + 1e-7) / (gt_pixels + pred_pixels + 1e-7) if (gt_pixels + pred_pixels) > 0 else 0.0
        
        # Store metrics
        if union > 0:  # Only count classes that appear in GT or prediction
            class_ious.append(iou)
            class_dices.append(dice)
        
        results['per_class'][class_names[cls]] = {
            'iou': float(iou),
            'dice': float(dice),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'gt_pixels': int(gt_pixels),
            'pred_pixels': int(pred_pixels),
            'precision': float(tp / (tp + fp + 1e-7)),
            'recall': float(tp / (tp + fn + 1e-7)),
        }
        
        # Print row
        print(f"{class_names[cls]:<15} {iou:<10.4f} {dice:<10.4f} {tp:<10} {fp:<10} {fn:<10} {gt_pixels:<10} {pred_pixels:<10}")
    
    print("-" * 80)
    
    # ========================================================================
    # Overall Metrics
    # ========================================================================
    mean_iou = np.mean(class_ious) if class_ious else 0.0
    mean_dice = np.mean(class_dices) if class_dices else 0.0
    
    # Pixel accuracy
    correct_pixels = np.sum(pred_np == mask_np)
    pixel_accuracy = correct_pixels / total_pixels
    
    results['overall'] = {
        'mean_iou': float(mean_iou),
        'mean_dice': float(mean_dice),
        'pixel_accuracy': float(pixel_accuracy),
        'correct_pixels': int(correct_pixels),
        'total_pixels': int(total_pixels)
    }
    
    print(f"\nOverall Metrics:")
    print(f"  Mean IoU:        {mean_iou:.4f}")
    print(f"  Mean Dice:       {mean_dice:.4f}")
    print(f"  Pixel Accuracy:  {pixel_accuracy:.4f} ({correct_pixels:,}/{total_pixels:,})")
    
    # ========================================================================
    # Confusion Summary
    # ========================================================================
    print(f"\nClass Distribution:")
    print(f"  Ground Truth:")
    for cls in range(num_classes):
        count = np.sum(mask_np == cls)
        pct = 100 * count / total_pixels
        print(f"    {class_names[cls]:<15}: {count:>8} pixels ({pct:>5.2f}%)")
    
    print(f"  Prediction:")
    for cls in range(num_classes):
        count = np.sum(pred_np == cls)
        pct = 100 * count / total_pixels
        print(f"    {class_names[cls]:<15}: {count:>8} pixels ({pct:>5.2f}%)")
    
    # ========================================================================
    # Misclassification Analysis
    # ========================================================================
    print(f"\nMisclassification Analysis:")
    total_errors = total_pixels - correct_pixels
    print(f"  Total errors: {total_errors:,} pixels ({100*total_errors/total_pixels:.2f}%)")
    
    # Find most common confusions
    confusions = []
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            if true_cls != pred_cls:
                count = np.sum((mask_np == true_cls) & (pred_np == pred_cls))
                if count > 0:
                    confusions.append((count, true_cls, pred_cls))
    
    confusions.sort(reverse=True)
    
    if confusions:
        print(f"  Top confusions:")
        for count, true_cls, pred_cls in confusions[:5]:
            pct = 100 * count / total_errors
            print(f"    {class_names[true_cls]} → {class_names[pred_cls]}: {count:>6} pixels ({pct:>5.2f}% of errors)")
    
    print("=" * 80)
    
    return results

# ------------------------------------------------------
# Multi-class Metric functions
# ------------------------------------------------------
def compute_iou(pred, mask, num_classes):
    """
    Compute mean IoU across all classes
    pred: [B, num_classes, H, W] - logits
    mask: [B, H, W] - class indices
    """
    # Convert logits to class predictions
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    iou_per_class = []
    
    for cls in range(num_classes):
        # Get binary masks for current class
        pred_cls = (pred == cls).float()
        mask_cls = (mask == cls).float()
        
        intersection = (pred_cls * mask_cls).sum()
        union = pred_cls.sum() + mask_cls.sum() - intersection
        
        if union > 0:
            iou = (intersection + 1e-7) / (union + 1e-7)
            iou_per_class.append(iou.item())
    
    # Return mean IoU and per-class IoU
    mean_iou = sum(iou_per_class) / max(len(iou_per_class), 1)
    return mean_iou, iou_per_class


def compute_dice(pred, mask, num_classes):
    """
    Compute mean Dice across all classes
    pred: [B, num_classes, H, W] - logits
    mask: [B, H, W] - class indices
    """
    # Convert logits to class predictions
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    dice_per_class = []
    
    for cls in range(num_classes):
        # Get binary masks for current class
        pred_cls = (pred == cls).float()
        mask_cls = (mask == cls).float()
        
        intersection = (pred_cls * mask_cls).sum()
        dice = (2. * intersection + 1e-7) / (pred_cls.sum() + mask_cls.sum() + 1e-7)
        
        if (pred_cls.sum() + mask_cls.sum()) > 0:
            dice_per_class.append(dice.item())
    
    # Return mean Dice and per-class Dice
    mean_dice = sum(dice_per_class) / max(len(dice_per_class), 1)
    return mean_dice, dice_per_class

# ------------------------------------------------------
# Helper function: Colorize segmentation for visualization
# ------------------------------------------------------
def colorize_segmentation(seg_map, num_classes):
    """
    Convert class indices to RGB colors for visualization
    seg_map: [H, W] numpy array with class indices
    Returns: [H, W, 3] RGB image
    """
    # Define a color palette (you can customize this)
    colors = [
        [0, 0, 0],       # Class 0: Black (background)
        [255, 0, 0],     # Class 1: Red
        [0, 255, 0],     # Class 2: Green
        [0, 0, 255],     # Class 3: Blue
        [255, 255, 0],   # Class 4: Yellow
        [255, 0, 255],   # Class 5: Magenta
        [0, 255, 255],   # Class 6: Cyan
    ]
    
    # Extend if needed
    while len(colors) < num_classes:
        colors.append([np.random.randint(0, 255) for _ in range(3)])
    
    h, w = seg_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls in range(num_classes):
        mask = seg_map == cls
        colored[mask] = colors[cls]
    
    return colored


# ------------------------------------------------------
# Main testing
# ------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--test_image_path", type=str, required=True, help="Path to test images")
parser.add_argument("--test_gt_path", type=str, required=True, help="Path to test ground truth masks")
parser.add_argument("--save_path", type=str, required=True, help="Path to save predictions")
parser.add_argument("--num_classes", type=int, default=7, help="Number of classes (including background)")
parser.add_argument("--input_size", type=int, default=1024, help="Input image size")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Number of classes: {args.num_classes}")

# Load test dataset
test_loader = TestDataset(args.test_image_path, args.test_gt_path, args.input_size)

# Load model
model = SAM2UNet(num_classes=args.num_classes).to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()


os.makedirs(args.save_path, exist_ok=True)

# Tracking metrics
mean_iou_list = []
mean_dice_list = []
per_class_iou_all = [[] for _ in range(args.num_classes)]
per_class_dice_all = [[] for _ in range(args.num_classes)]

print("\n" + "="*80)
print("Starting Testing")
print(f"Total test images: {test_loader.size}")
print("="*80)

for i in range(test_loader.size):
    try:
        with torch.no_grad():
            image, gt, name = test_loader.load_data()

        # Convert GT to torch tensor (keep as class indices)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).long().to(device)  # [1, H, W]

        image = image.to(device)

        # Get prediction
        res, _, _ = model(image)  # [1, num_classes, H, W]
        
        # Resize to match ground truth size
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        
        # Compute metrics
        mean_iou, iou_per_class = compute_iou(res, gt_tensor, args.num_classes)
        mean_dice, dice_per_class = compute_dice(res, gt_tensor, args.num_classes)
        
        mean_iou_list.append(mean_iou)
        mean_dice_list.append(mean_dice)
        
        # Store per-class metrics
        for cls_idx, (iou_val, dice_val) in enumerate(zip(iou_per_class, dice_per_class)):
            per_class_iou_all[cls_idx].append(iou_val)
            per_class_dice_all[cls_idx].append(dice_val)
        
        # Save prediction as colored segmentation map
        pred_class = torch.argmax(res, dim=1).squeeze().cpu().numpy()  # [H, W] with class indices
        # print(pred_class.shape)
        # print(pred_class.dtype)
        # print(pred_class)
        # Option 1: Save as class indices (grayscale)
        pred_gray = (pred_class * 40).astype(np.uint8)  # Scale for visibility
        imageio.imsave(os.path.join(args.save_path, name.replace(".jpg", "_classes.png")), pred_gray)
        
        # Option 2: Save as colored visualization
        colored_pred = colorize_segmentation(pred_class, args.num_classes)
        imageio.imsave(os.path.join(args.save_path, name.replace(".jpg", "_color.png")), colored_pred)
        
        print(f"[{i+1}/{test_loader.size}] {name} | IoU: {mean_iou:.4f} | Dice: {mean_dice:.4f}")

        if name == "339057-N_jpg.rf.79daa5b1fe03c2423daf349a005477ef.jpg":
            class_names = ['Background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
            results = compute_detailed_metrics(res, gt_tensor, args.num_classes, class_names)
            print(f"\nC1 IoU: {results['per_class']['C1']['iou']:.4f}")
            print(f"C1 Precision: {results['per_class']['C1']['precision']:.4f}")
            # print unique values 
            unique_vals = np.unique(pred_class)
            print(f"Unique predicted class indices in {name}: {unique_vals}")

    except Exception as e:
        print(f"Error processing image {i+1}: {e}")
        continue

# ------------------------------------------------------
# Final results
# ------------------------------------------------------
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nOverall Metrics:")
print(f"  Mean IoU:  {np.mean(mean_iou_list):.4f}")
print(f"  Mean Dice: {np.mean(mean_dice_list):.4f}")

print(f"\nPer-Class Metrics:")
print("-" * 80)
print(f"{'Class':<10} {'IoU':<15} {'Dice':<15} {'Samples':<10}")
print("-" * 80)
for cls in range(args.num_classes):
    if len(per_class_iou_all[cls]) > 0:
        avg_iou = np.mean(per_class_iou_all[cls])
        avg_dice = np.mean(per_class_dice_all[cls])
        samples = len(per_class_iou_all[cls])
        class_name = f"Class {cls}" if cls > 0 else "Background"
        print(f"{class_name:<10} {avg_iou:<15.4f} {avg_dice:<15.4f} {samples:<10}")
    else:
        class_name = f"Class {cls}" if cls > 0 else "Background"
        print(f"{class_name:<10} {'N/A':<15} {'N/A':<15} {0:<10}")
print("="*80)

# Save metrics to file
metrics_file = os.path.join(args.save_path, "test_metrics.txt")
with open(metrics_file, 'w') as f:
    f.write("Overall Metrics:\n")
    f.write(f"Mean IoU: {np.mean(mean_iou_list):.4f}\n")
    f.write(f"Mean Dice: {np.mean(mean_dice_list):.4f}\n\n")
    f.write("Per-Class Metrics:\n")
    f.write(f"{'Class':<10} {'IoU':<15} {'Dice':<15} {'Samples':<10}\n")
    for cls in range(args.num_classes):
        if len(per_class_iou_all[cls]) > 0:
            avg_iou = np.mean(per_class_iou_all[cls])
            avg_dice = np.mean(per_class_dice_all[cls])
            samples = len(per_class_iou_all[cls])
            class_name = f"Class {cls}" if cls > 0 else "Background"
            f.write(f"{class_name:<10} {avg_iou:<15.4f} {avg_dice:<15.4f} {samples:<10}\n")

print(f"\nMetrics saved to: {metrics_file}")


