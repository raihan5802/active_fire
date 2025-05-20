# analyze_feature_importance.py
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from satimg_dataset_processor.data_generator_torch import Normalize, FireDataset
from spatial_models.unetr.unetr import UNETR
from sklearn.metrics import f1_score

def analyze_feature_importance(args):
    # Set up parameters
    model_name = args.model
    checkpoint_path = args.checkpoint
    batch_size = args.batch_size
    num_heads = args.num_heads
    hidden_size = args.hidden_size
    ts_length = args.ts_length
    interval = args.interval
    n_channel = args.n_channel
    num_classes = 2
    root_path = args.data_path
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define normalization transform
    transform = Normalize(
        mean=[18.76488, 27.441864, 20.584806, 305.99478, 294.31738, 14.625097, 276.4207, 275.16766],
        std=[15.911591, 14.879259, 10.832616, 21.761852, 24.703484, 9.878246, 40.64329, 40.7657]
    )
    
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Set up validation dataset for feature importance analysis
    val_image_path = os.path.join(root_path, f'dataset_val/af_val_img_seqtoseq_alll_{ts_length}i_{interval}.npy')
    val_label_path = os.path.join(root_path, f'dataset_val/af_val_label_seqtoseq_alll_{ts_length}i_{interval}.npy')
    
    val_dataset = FireDataset(
        image_path=val_image_path,
        label_path=val_label_path,
        ts_length=ts_length,
        transform=transform,
        n_channel=n_channel,
        label_sel=2  # For active fire detection
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define band names for reference
    band_names = [
        "I1 (Blue)",
        "I2 (Near IR)",
        "I3 (Mid IR)",
        "I4 (Thermal IR)",
        "I5 (Thermal IR)",
        "M11 (Mid IR)",
        "Night I1",
        "Night I2"
    ]
    
    # Feature importance analysis
    feature_importance = np.zeros(n_channel)
    
    # Evaluate performance when zeroing out each band
    for band_idx in range(n_channel):
        print(f"Analyzing importance of band {band_idx} ({band_names[band_idx]})...")
        
        f1_scores = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                data_batch = batch['data']
                labels_batch = batch['labels']
                
                # Create a copy with the current band zeroed out
                zeroed_data = data_batch.clone()
                zeroed_data[:, band_idx] = 0
                
                zeroed_data = zeroed_data.to(device)
                labels_batch = labels_batch.to(device)
                
                # Forward pass
                outputs = model(zeroed_data)
                outputs = torch.sigmoid(outputs)
                
                # Calculate F1 score for this batch
                outputs_binary = (outputs[:, 1] > 0.5).cpu().numpy()
                labels = (labels_batch[:, 1] > 0).cpu().numpy()
                
                # Calculate F1 score for each sample and timestep
                for i in range(outputs_binary.shape[0]):
                    for t in range(outputs_binary.shape[1]):
                        f1 = f1_score(
                            labels[i, t].flatten(), 
                            outputs_binary[i, t].flatten(), 
                            zero_division=1.0
                        )
                        f1_scores.append(f1)
        
        # Calculate average F1 score with this band zeroed out
        avg_f1_zeroed = np.mean(f1_scores)
        
        # Calculate baseline performance (all bands)
        baseline_f1_scores = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                data_batch = batch['data']
                labels_batch = batch['labels']
                
                data_batch = data_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                # Forward pass
                outputs = model(data_batch)
                outputs = torch.sigmoid(outputs)
                
                # Calculate F1 score for this batch
                outputs_binary = (outputs[:, 1] > 0.5).cpu().numpy()
                labels = (labels_batch[:, 1] > 0).cpu().numpy()
                
                # Calculate F1 score for each sample and timestep
                for i in range(outputs_binary.shape[0]):
                    for t in range(outputs_binary.shape[1]):
                        f1 = f1_score(
                            labels[i, t].flatten(), 
                            outputs_binary[i, t].flatten(), 
                            zero_division=1.0
                        )
                        baseline_f1_scores.append(f1)
        
        avg_baseline_f1 = np.mean(baseline_f1_scores)
        
        # The importance is the difference in performance
        feature_importance[band_idx] = avg_baseline_f1 - avg_f1_zeroed
        
        print(f"Band {band_idx} ({band_names[band_idx]}): Baseline F1 = {avg_baseline_f1:.4f}, "
              f"Zeroed F1 = {avg_f1_zeroed:.4f}, Importance = {feature_importance[band_idx]:.4f}")
    
    # Normalize importance scores
    if np.sum(feature_importance) > 0:
        feature_importance = feature_importance / np.sum(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_channel), feature_importance)
    plt.xlabel('Band')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importance for Active Fire Detection')
    plt.xticks(range(n_channel), band_names, rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    
    # Save results to text file
    with open(os.path.join(output_dir, "feature_importance.txt"), "w") as f:
        f.write("Feature Importance Analysis for Active Fire Detection\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        
        for i in range(n_channel):
            f.write(f"Band {i} ({band_names[i]}): {feature_importance[i]:.4f}\n")
    
    # Create a summary of band importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    print("\nBand Importance Ranking:")
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. {band_names[idx]} (Band {idx}): {feature_importance[idx]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Feature Importance for Active Fire Detection')
    parser.add_argument('--model', type=str, default='unetr3d', help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension size')
    parser.add_argument('--ts_length', type=int, default=10, help='Time series length')
    parser.add_argument('--interval', type=int, default=3, help='Interval')
    parser.add_argument('--n_channel', type=int, default=8, help='Number of input channels')
    parser.add_argument('--data_path', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='feature_importance', help='Directory to save results')
    
    args = parser.parse_args()
    analyze_feature_importance(args)