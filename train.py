import os
import argparse
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet

# Argument configuration
parser = argparse.ArgumentParser("SAM2-UNet Multi-class Segmentation")
parser.add_argument("--hiera_path", type=str, required=True, help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True, help="path to the training images")
parser.add_argument("--train_mask_path", type=str, required=True, help="path to the training masks")
parser.add_argument("--val_image_path", type=str, required=True, help="path to the validation images")
parser.add_argument("--val_mask_path", type=str, required=True, help="path to the validation masks")
parser.add_argument('--save_path', type=str, required=True, help="path to store the checkpoints")
parser.add_argument("--num_classes", type=int, default=7, help="number of classes (including background)")
parser.add_argument("--epoch", type=int, default=20, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--input_size", default=1024, type=int, help="input image size (512 or 1024)")
args = parser.parse_args()


# Multi-class IoU computation
def compute_iou(pred, mask, num_classes):
    """
    Compute mean IoU across all classes
    pred: [B, num_classes, H, W] - logits
    mask: [B, H, W] - class indices
    """
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        mask_cls = (mask == cls).float()
        
        intersection = (pred_cls * mask_cls).sum()
        union = pred_cls.sum() + mask_cls.sum() - intersection
        
        if union > 0:
            iou = (intersection + 1e-7) / (union + 1e-7)
            iou_per_class.append(iou.item())
    
    return sum(iou_per_class) / max(len(iou_per_class), 1)


# Multi-class Dice computation
def compute_dice(pred, mask, num_classes):
    """
    Compute mean Dice across all classes
    pred: [B, num_classes, H, W] - logits
    mask: [B, H, W] - class indices
    """
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    dice_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        mask_cls = (mask == cls).float()
        
        intersection = (pred_cls * mask_cls).sum()
        dice = (2. * intersection + 1e-7) / (pred_cls.sum() + mask_cls.sum() + 1e-7)
        
        if (pred_cls.sum() + mask_cls.sum()) > 0:
            dice_per_class.append(dice.item())
    
    return sum(dice_per_class) / max(len(dice_per_class), 1)


# Multi-class segmentation loss
def structure_loss(pred, mask):
    """
    Multi-class segmentation loss
    pred: [B, num_classes, H, W] - logits
    mask: [B, H, W] - class indices (long tensor)
    """
    # Cross-entropy loss
    ce_loss = F.cross_entropy(pred, mask, reduction='mean')
    
    # Dice loss for multi-class
    pred_soft = F.softmax(pred, dim=1)  # [B, num_classes, H, W]
    
    # Convert mask to one-hot encoding
    num_classes = pred.shape[1]
    mask_one_hot = F.one_hot(mask, num_classes)  # [B, H, W, num_classes]
    mask_one_hot = mask_one_hot.permute(0, 3, 1, 2).float()  # [B, num_classes, H, W]
    
    # Dice loss
    intersection = (pred_soft * mask_one_hot).sum(dim=(2, 3))  # [B, num_classes]
    union = pred_soft.sum(dim=(2, 3)) + mask_one_hot.sum(dim=(2, 3))  # [B, num_classes]
    dice_loss = 1 - (2. * intersection + 1e-7) / (union + 1e-7)
    dice_loss = dice_loss.mean()
    
    return ce_loss + dice_loss


# Train the model for one epoch
def train_one_epoch(model, dataloader, optimizer, device, num_classes):
    model.train()
    epoch_loss, iou_score, dice_score = 0.0, 0.0, 0.0
    
    for i, batch in enumerate(dataloader):
        x = batch['image'].to(device)
        target = batch['label'].to(device)  # [B, H, W] - class indices
        
        optimizer.zero_grad()
        
        pred0, pred1, pred2 = model(x)  # Each: [B, num_classes, H, W]
        
        loss0 = structure_loss(pred0, target)
        loss1 = structure_loss(pred1, target)
        loss2 = structure_loss(pred2, target)
        loss = loss0 + loss1 + loss2
        
        # Compute metrics on final prediction
        iou = compute_iou(pred0, target, num_classes)
        dice = compute_dice(pred0, target, num_classes)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        iou_score += iou
        dice_score += dice
        
        # Print progress every 10 batches
        if (i + 1) % 10 == 0:
            print(f"  Batch [{i+1}/{len(dataloader)}] - Loss: {loss.item():.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    
    return avg_epoch_loss, avg_iou, avg_dice


# NEW: Validate the model
def validate(model, dataloader, device, num_classes):
    model.eval()
    val_loss, iou_score, dice_score = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            
            # Only use the main output (pred0) for validation
            pred0, _, _ = model(x)
            
            loss = structure_loss(pred0, target)
            iou = compute_iou(pred0, target, num_classes)
            dice = compute_dice(pred0, target, num_classes)
            
            val_loss += loss.item()
            iou_score += iou
            dice_score += dice
    
    avg_val_loss = val_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    
    return avg_val_loss, avg_iou, avg_dice


# Main training function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    
    # Load training dataset
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, args.input_size, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    # NEW: Load validation dataset
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, args.input_size, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = SAM2UNet(args.hiera_path, num_classes=args.num_classes).to(device)
    
    optimizer = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # NEW: Track best validation IoU (not training IoU)
    best_val_iou = 0.0
    best_model_path = None
    
    # NEW: Log file for tracking
    log_file = os.path.join(args.save_path, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_IoU,Train_Dice,Val_Loss,Val_IoU,Val_Dice\n")
    
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    for epoch in range(args.epoch):
        print(f"\nEpoch [{epoch + 1}/{args.epoch}]")
        print("-" * 80)
        
        # Training
        print("Training...")
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, optimizer, device, args.num_classes
        )
        
        # Validation
        print("Validating...")
        val_loss, val_iou, val_dice = validate(
            model, val_loader, device, args.num_classes
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print summary
        print(f"\n[Epoch {epoch + 1} Summary]")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_iou:.4f},{train_dice:.4f},"
                   f"{val_loss:.4f},{val_iou:.4f},{val_dice:.4f}\n")
        
        # NEW: Save model based on VALIDATION IoU, not training IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            
            # Delete previous best model
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_model_path = os.path.join(args.save_path, f'best_model_epoch-{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ NEW BEST MODEL! Val IoU: {best_val_iou:.4f} (saved to {best_model_path})")
        
        print("-" * 80)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Best Model: {best_model_path}")
    print(f"Training log: {log_file}")
    print("="*80)


if __name__ == "__main__":
    main(args)