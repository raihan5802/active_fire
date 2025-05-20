# train_active_fire.py - Modified version with improved training strategies
import argparse
import heapq
import os
import numpy as np
import torch
from monai.losses.dice import DiceLoss
from monai.metrics import MeanIoU, DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from satimg_dataset_processor.data_generator_torch import Normalize, FireDataset, WeightedFireDataLoader
from spatial_models.unetr.unetr import UNETR
# Import SwinUNETR model
from spatial_models.swinunetr.swinunetr import SwinUNETR
from sklearn.metrics import f1_score, jaccard_score
import matplotlib.pyplot as plt

# New focal loss implementation for small fire instances
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Use sigmoid to get probability
        inputs_soft = torch.sigmoid(inputs)
        
        # Flatten for easier processing
        batch_size = inputs.size(0)
        inputs_soft = inputs_soft.view(batch_size, -1)
        targets = targets.view(batch_size, -1).float()
        
        # Focal loss formula
        pt = torch.where(targets == 1, inputs_soft, 1 - inputs_soft)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        modulating_factor = (1.0 - pt) ** self.gamma
        
        # Calculate loss
        ce_loss = -torch.log(pt + 1e-10)  # Add epsilon for numerical stability
        loss = alpha_factor * modulating_factor * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# Combined loss for better fire detection
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(include_background=True, reduction='mean', sigmoid=True)
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal

# Time Series Consistency tracker for evaluation
class TimeSeriesConsistencyTracker:
    def __init__(self):
        self.detection_stability = []
        
    def update(self, preds, targets):
        """
        Calculate temporal consistency of predictions over time series
        Args:
            preds: predicted fire maps (batch, time, h, w)
            targets: ground truth fire maps (batch, time, h, w)
        """
        batch_size, time_length = preds.shape[0], preds.shape[1]
        
        for b in range(batch_size):
            # Calculate per-pixel temporal consistency
            if time_length <= 1:
                continue
                
            # For each fire pixel in any time frame, check consistency of predictions
            fire_pixels_any = torch.any(targets[b, :, :, :] > 0.5, dim=0)
            
            if torch.sum(fire_pixels_any) == 0:
                continue
                
            # Calculate how stable the predictions are over time for true fire pixels
            pred_changes = torch.sum(torch.abs(preds[b, 1:, :, :] - preds[b, :-1, :, :]), dim=0)
            pred_changes = pred_changes[fire_pixels_any]
            
            # Normalize by time length and number of pixels
            stability = 1.0 - torch.mean(pred_changes) / (time_length - 1)
            self.detection_stability.append(stability.item())
            
    def get_metrics(self):
        if len(self.detection_stability) == 0:
            return {"temporal_consistency": 0.0}
        
        return {"temporal_consistency": np.mean(self.detection_stability)}
        
    def reset(self):
        self.detection_stability = []

def train_active_fire(args):
    # Set up parameters
    model_name = args.model
    batch_size = args.batch_size
    num_heads = args.num_heads
    hidden_size = args.hidden_size
    ts_length = args.ts_length
    interval = args.interval
    n_channel = args.n_channel
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay if args.weight_decay else learning_rate / 10
    num_classes = 2
    max_epochs = args.epochs
    checkpoint_dir = args.checkpoint_dir
    root_path = args.data_path
    
    # Set up random seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define normalization transform
    transform = Normalize(
        mean=[18.76488, 27.441864, 20.584806, 305.99478, 294.31738, 14.625097, 276.4207, 275.16766],
        std=[15.911591, 14.879259, 10.832616, 21.761852, 24.703484, 9.878246, 40.64329, 40.7657]
    )
    
    # Set up data paths
    image_path = os.path.join(root_path, f'dataset_train/af_train_img_seqtoseq_alll_{ts_length}i_{interval}.npy')
    label_path = os.path.join(root_path, f'dataset_train/af_train_label_seqtoseq_alll_{ts_length}i_{interval}.npy')
    val_image_path = os.path.join(root_path, f'dataset_val/af_val_img_seqtoseq_alll_{ts_length}i_{interval}.npy')
    val_label_path = os.path.join(root_path, f'dataset_val/af_val_label_seqtoseq_alll_{ts_length}i_{interval}.npy')
    
    # Create datasets with weighted sampling for class imbalance
    train_dataset = FireDataset(
        image_path=image_path,
        label_path=label_path,
        ts_length=ts_length,
        transform=transform,
        n_channel=n_channel,
        label_sel=2,  # For active fire detection
        weighted_sampling=True  # Enable weighted sampling
    )
    
    # Use the custom weighted dataloader for training
    train_dataloader = WeightedFireDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataset = FireDataset(
        image_path=val_image_path,
        label_path=val_label_path,
        ts_length=ts_length,
        transform=transform,
        n_channel=n_channel,
        label_sel=2  # For active fire detection
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model based on selection
    image_size = (ts_length, 256, 256)
    if model_name == 'unetr3d':
        patch_size = (1, 16, 16)
        model = UNETR(
            in_channels=n_channel,
            out_channels=num_classes,
            img_size=image_size,
            feature_size=hidden_size,
            hidden_size=hidden_size*4,
            mlp_dim=hidden_size*8,
            num_heads=num_heads,
            patch_size=patch_size,
            spatial_dims=3,
            norm_name='batch'
        )
    elif model_name == 'swinunetr':
        window_size = (2, 7, 7)  # Adjusted for satellite imagery time series
        model = SwinUNETR(
            in_channels=n_channel,
            out_channels=num_classes,
            img_size=image_size,
            feature_size=hidden_size,
            window_size=window_size,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=3
        )
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')
    
    # Define combined loss function, metrics
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    mean_iou = MeanIoU(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    temporal_consistency = TimeSeriesConsistencyTracker()
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (cosine annealing)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max_epochs // 3,  # Restart every 1/3 of total epochs
        T_mult=1,
        eta_min=learning_rate / 100
    )
    
    # Enable mixed precision training
    scaler = GradScaler()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up best checkpoint tracking
    best_checkpoints = []
    top_n_checkpoints = 3
    
    # Initialize WandB if enabled
    if args.use_wandb:
        wandb.init(
            project="active-fire-detection",
            config={
                "model": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": max_epochs,
                "ts_length": ts_length,
                "num_heads": num_heads,
                "hidden_size": hidden_size,
                "n_channel": n_channel
            }
        )
    
    # Training loop with enhanced strategies
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_dataloader, total=len(train_dataloader))
        
        for i, batch in enumerate(train_bar):
            data_batch = batch['data']
            labels_batch = batch['labels']
            data_batch = data_batch.to(device)
            labels_batch = labels_batch.to(torch.long).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(data_batch)
                loss = criterion(outputs, labels_batch)
            
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.detach().item() * data_batch.size(0)
            train_bar.set_description(
                f"Epoch {epoch}/{max_epochs}, Loss: {train_loss/((i+1)* data_batch.size(0)):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Log intermediate metrics if using wandb
            if args.use_wandb and i % args.log_interval == 0:
                wandb.log({
                    "train_loss_step": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                })
        
        # Step learning rate scheduler at the end of epoch
        lr_scheduler.step()
        
        train_loss /= len(train_dataset)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        iou_values = []
        dice_values = []
        f1_values = []
        temporal_consistency.reset()
        val_bar = tqdm(val_dataloader, total=len(val_dataloader))
        
        with torch.no_grad():
            for j, batch in enumerate(val_bar):
                val_data_batch = batch['data']
                val_labels_batch = batch['labels']
                val_data_batch = val_data_batch.to(device)
                val_labels_batch = val_labels_batch.to(torch.long).to(device)
                
                outputs = model(val_data_batch)
                loss = criterion(outputs, val_labels_batch)
                
                # Post-process outputs
                outputs_post = torch.sigmoid(outputs)
                outputs_bin = (outputs_post > 0.5).float()
                
                # Calculate temporal consistency
                for b in range(outputs_bin.shape[0]):
                    temporal_consistency.update(
                        outputs_bin[b:b+1, 1].detach().cpu(), 
                        val_labels_batch[b:b+1, 1].detach().cpu()
                    )
                
                # Calculate metrics
                for b in range(outputs_bin.shape[0]):
                    for t in range(ts_length):
                        pred = outputs_bin[b, 1, t].detach().cpu().numpy().flatten()
                        target = val_labels_batch[b, 1, t].detach().cpu().numpy().flatten()
                        
                        if np.sum(target) > 0:  # Only calculate if there are fire pixels
                            f1 = f1_score(target, pred, zero_division=1)
                            f1_values.append(f1)
                
                # Calculate IoU and Dice
                outputs_list = [post_trans(o) for o in outputs]
                val_labels_list = val_labels_batch
                
                val_loss += loss.detach().item() * val_data_batch.size(0)
                iou_values.append(mean_iou(outputs_list, val_labels_list).mean().item())
                dice_values.append(dice_metric(y_pred=outputs_list, y=val_labels_list).mean().item())
                
                val_bar.set_description(
                    f"Epoch {epoch}/{max_epochs}, Val Loss: {val_loss / ((j + 1) * val_data_batch.size(0)):.4f}"
                )
        
        # Calculate validation metrics
        val_loss /= len(val_dataset)
        mean_iou_val = np.mean(iou_values) if iou_values else 0
        mean_dice_val = np.mean(dice_values) if dice_values else 0
        mean_f1_val = np.mean(f1_values) if f1_values else 0
        temporal_metrics = temporal_consistency.get_metrics()
        
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou_val:.4f}, Mean Dice: {mean_dice_val:.4f}, Mean F1: {mean_f1_val:.4f}")
        print(f"Temporal Consistency: {temporal_metrics['temporal_consistency']:.4f}")
        
        # Log validation metrics
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": mean_iou_val,
                "val_dice": mean_dice_val,
                "val_f1": mean_f1_val,
                "temporal_consistency": temporal_metrics['temporal_consistency'],
            })
        
        # Save top N checkpoints, but start saving after certain number of epochs
        if (len(best_checkpoints) < top_n_checkpoints or val_loss < best_checkpoints[0][0]) and epoch >= args.save_start_epoch:
            save_path = os.path.join(
                checkpoint_dir,
                f"model_{model_name}_epoch_{epoch + 1}_iou_{mean_iou_val:.4f}_dice_{mean_dice_val:.4f}_f1_{mean_f1_val:.4f}.pth"
            )
            
            if len(best_checkpoints) == top_n_checkpoints:
                _, remove_checkpoint = heapq.heappop(best_checkpoints)
                if os.path.exists(remove_checkpoint):
                    os.remove(remove_checkpoint)
            
            # Save model state
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss,
                'iou': mean_iou_val,
                'dice': mean_dice_val,
                'f1': mean_f1_val,
                'temporal_consistency': temporal_metrics['temporal_consistency'],
            }, save_path)
            
            heapq.heappush(best_checkpoints, (val_loss, save_path))
            best_checkpoints = heapq.nlargest(top_n_checkpoints, best_checkpoints)
    
    # Final model saving
    final_save_path = os.path.join(checkpoint_dir, f"model_{model_name}_final.pth")
    
    # Save model state
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
    }, final_save_path)
    
    print("Top N best checkpoints:")
    for _, checkpoint in best_checkpoints:
        print(checkpoint)
        
    # Close wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Active Fire Detection model')
    parser.add_argument('--model', type=str, default='swinunetr', help='Model to be executed (unetr3d or swinunetr)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension size')
    parser.add_argument('--ts_length', type=int, default=10, help='Time series length')
    parser.add_argument('--interval', type=int, default=3, help='Interval')
    parser.add_argument('--n_channel', type=int, default=8, help='Number of input channels')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--save_start_epoch', type=int, default=50, help='Start saving checkpoints from this epoch')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use gradient checkpointing to save memory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for wandb')
    parser.add_argument('--data_path', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='saved_models', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    train_active_fire(args)