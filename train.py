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
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True, help="path to the images used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True, help="path to the mask files for training")
parser.add_argument("--val_image_path", type=str, required=True, help="path to the validation images")
parser.add_argument("--val_mask_path", type=str, required=True, help="path to the validation masks")
parser.add_argument('--save_path', type=str, required=True, help="path to store the checkpoints")
parser.add_argument("--epoch", type=int, default=20, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


# Functions to compute IoU and Dice metrics
def compute_iou(pred, mask):
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum((1, 2, 3))
    union = (pred + mask).sum((1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean()


def compute_dice(pred, mask):
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum((1, 2, 3))
    dice = (2. * intersection + 1e-7) / (pred.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1e-7)
    return dice.mean()


# Custom loss function
def structure_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# Train the model for one epoch
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    epoch_loss, iou_score, dice_score = 0.0, 0.0, 0.0
    
    for i, batch in enumerate(dataloader):
        x = batch['image'].to(device)
        target = batch['label'].to(device)
        optimizer.zero_grad()
        
        pred0, pred1, pred2 = model(x)
        loss0 = structure_loss(pred0, target)
        loss1 = structure_loss(pred1, target)
        loss2 = structure_loss(pred2, target)
        loss = loss0 + loss1 + loss2
        
        iou = compute_iou(torch.sigmoid(pred0), target)
        dice = compute_dice(torch.sigmoid(pred0), target)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        iou_score += iou.item()
        dice_score += dice.item()
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    
    scheduler.step()
    
    return avg_epoch_loss, avg_iou, avg_dice


# Validate the model
def validate(model, dataloader, device):
    model.eval()
    val_loss, iou_score, dice_score = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            
            iou = compute_iou(torch.sigmoid(pred0), target)
            dice = compute_dice(torch.sigmoid(pred0), target)
            
            val_loss += loss.item()
            iou_score += iou.item()
            dice_score += dice.item()
    
    avg_val_loss = val_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    
    return avg_val_loss, avg_iou, avg_dice


# Main training function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training dataset and dataloader
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    # Validation dataset and dataloader
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 352, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    model = SAM2UNet(args.hiera_path).to(device)
    optimizer = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    best_iou = 0.0
    best_model_path = None
    
    for epoch in range(args.epoch):
        # Training
        avg_loss, avg_iou, avg_dice = train_one_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Epoch [{epoch + 1}/{args.epoch}] - Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}, Train Dice: {avg_dice:.4f}")
        
        # Validation
        val_loss, val_iou, val_dice = validate(model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}/{args.epoch}] - Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save model if the current validation IoU is the best
        if val_iou > best_iou:
            best_iou = val_iou
            best_model_path = os.path.join(args.save_path, f'best_model_epoch-{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)

            # Delete previous best model files to save space
            for file in os.listdir(args.save_path):
                if file.startswith("best_model_epoch-") and file != f'best_model_epoch-{epoch + 1}.pth':
                    os.remove(os.path.join(args.save_path, file))
            
            print(f"[Saving Best Model: Val IoU={best_iou:.4f} at {best_model_path}]")
        
        print("-" * 80)


if __name__ == "__main__":
    main(args)