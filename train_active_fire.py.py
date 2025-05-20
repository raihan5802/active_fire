# train_active_fire.py - Modified version of run_spatial_temp_model.py for active fire detection
import argparse
import heapq
import os
import numpy as np
import torch
from monai.losses.dice import DiceLoss
from monai.metrics import MeanIoU, DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from satimg_dataset_processor.data_generator_torch import Normalize, FireDataset
from spatial_models.unetr.unetr import UNETR
from sklearn.metrics import f1_score, jaccard_score
import matplotlib.pyplot as plt

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
    weight_decay = learning_rate / 10
    num_classes = 2
    max_epochs = args.epochs
    checkpoint_dir = args.checkpoint_dir
    root_path = args.data_path
    
    # Set up random seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
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
    
    # Create datasets and dataloaders
    train_dataset = FireDataset(
        image_path=image_path,
        label_path=label_path,
        ts_length=ts_length,
        transform=transform,
        n_channel=n_channel,
        label_sel=2  # For active fire detection
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = FireDataset(
        image_path=val_image_path,
        label_path=val_label_path,
        ts_length=ts_length,
        transform=transform,
        n_channel=n_channel,
        label_sel=2  # For active fire detection
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Define model
    image_size = (ts_length, 256, 256)
    patch_size = (1, 16, 16)
    
    if model_name == 'unetr3d':
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
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')
    
    # Define loss function, metrics, and optimizer
    criterion = DiceLoss(include_background=True, reduction='mean', sigmoid=True)
    mean_iou = MeanIoU(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up best checkpoint tracking
    best_checkpoints = []
    top_n_checkpoints = 3
    
    # Training loop
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
            
            # Forward pass
            outputs = model(data_batch)
            loss = criterion(outputs, labels_batch)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.detach().item() * data_batch.size(0)
            train_bar.set_description(
                f"Epoch {epoch}/{max_epochs}, Loss: {train_loss/((i+1)* data_batch.size(0)):.4f}"
            )
        
        train_loss /= len(train_dataset)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        iou_values = []
        dice_values = []
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
                outputs = [post_trans(i) for i in decollate_batch(outputs)]
                val_labels_batch = decollate_batch(val_labels_batch)
                
                val_loss += loss.detach().item() * val_data_batch.size(0)
                iou_values.append(mean_iou(outputs, val_labels_batch).mean().item())
                dice_values.append(dice_metric(y_pred=outputs, y=val_labels_batch).mean().item())
                
                val_bar.set_description(
                    f"Epoch {epoch}/{max_epochs}, Loss: {val_loss / ((j + 1) * val_data_batch.size(0)):.4f}"
                )
        
        val_loss /= len(val_dataset)
        mean_iou_val = np.mean(iou_values)
        mean_dice_val = np.mean(dice_values)
        
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou_val:.4f}, Mean Dice: {mean_dice_val:.4f}")
        
        # Save top N checkpoints
        if (len(best_checkpoints) < top_n_checkpoints or val_loss < best_checkpoints[0][0]) and epoch >= 50:
            save_path = os.path.join(
                checkpoint_dir,
                f"model_{model_name}_epoch_{epoch + 1}_iou_{mean_iou_val:.4f}_dice_{mean_dice_val:.4f}.pth"
            )
            
            if len(best_checkpoints) == top_n_checkpoints:
                _, remove_checkpoint = heapq.heappop(best_checkpoints)
                if os.path.exists(remove_checkpoint):
                    os.remove(remove_checkpoint)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'iou': mean_iou_val,
                'dice': mean_dice_val,
            }, save_path)
            
            heapq.heappush(best_checkpoints, (val_loss, save_path))
            best_checkpoints = heapq.nlargest(top_n_checkpoints, best_checkpoints)
    
    # Final model saving
    final_save_path = os.path.join(checkpoint_dir, f"model_{model_name}_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, final_save_path)
    
    print("Top N best checkpoints:")
    for _, checkpoint in best_checkpoints:
        print(checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Active Fire Detection model')
    parser.add_argument('--model', type=str, default='unetr3d', help='Model to be executed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension size')
    parser.add_argument('--ts_length', type=int, default=10, help='Time series length')
    parser.add_argument('--interval', type=int, default=3, help='Interval')
    parser.add_argument('--n_channel', type=int, default=8, help='Number of input channels')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_path', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='saved_models', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    train_active_fire(args)