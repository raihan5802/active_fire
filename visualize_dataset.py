import os
import numpy as np
import matplotlib.pyplot as plt
from satimg_dataset_processor.data_generator_torch import DataGenerator, FireDataset, Normalize

def visualize_dataset(data_path, label_path, output_dir, ts_length=10, n_channel=8):
    """
    Visualize the Active Fire Detection dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define normalization transform
    transform = Normalize(
        mean=[18.76488, 27.441864, 20.584806, 305.99478, 294.31738, 14.625097, 276.4207, 275.16766],
        std=[15.911591, 14.879259, 10.832616, 21.761852, 24.703484, 9.878246, 40.64329, 40.7657]
    )
    
    # Create dataset
    dataset = FireDataset(
        image_path=data_path, 
        label_path=label_path, 
        ts_length=ts_length,
        transform=transform,
        n_channel=n_channel,
        label_sel=2  # For active fire detection
    )
    
    # Get number of samples to visualize (max 5)
    num_samples = min(5, len(dataset))
    
    for i in range(num_samples):
        sample = dataset[i]
        data = sample['data']  # Shape: [n_channel, ts_length, height, width]
        labels = sample['labels']  # Shape: [2, ts_length, height, width]
        
        # Visualize each timestep
        for t in range(ts_length):
            plt.figure(figsize=(15, 10))
            
            # Visualize each spectral band
            for c in range(min(6, n_channel)):  # Visualize up to 6 bands
                plt.subplot(2, 3, c+1)
                # Normalize for visualization
                band_data = data[c, t].numpy()
                vmin, vmax = np.nanpercentile(band_data, [2, 98])
                plt.imshow(band_data, cmap='viridis', vmin=vmin, vmax=vmax)
                plt.title(f'Band {c+1}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i}_timestep_{t}_bands.png'))
            plt.close()
            
            # Visualize fire labels
            plt.figure(figsize=(10, 5))
            
            # Show active fire label
            plt.subplot(1, 2, 1)
            # Use band I4 (index 3) as background
            background = data[3, t].numpy()
            background = (background - np.nanmin(background)) / (np.nanmax(background) - np.nanmin(background))
            plt.imshow(background, cmap='gray')
            
            # Overlay active fire label
            fire_mask = labels[1, t].numpy()  # Class 1 is fire
            plt.imshow(np.ma.masked_where(fire_mask == 0, fire_mask), 
                      cmap='hot', alpha=0.7, vmin=0, vmax=1)
            plt.title('Active Fire Overlay')
            plt.axis('off')
            
            # Show only active fire
            plt.subplot(1, 2, 2)
            plt.imshow(fire_mask, cmap='hot', vmin=0, vmax=1)
            plt.title('Active Fire Mask')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i}_timestep_{t}_fire.png'))
            plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    root_path = 'dataset'
    ts_length = 10
    interval = 3
    
    # Visualize training data
    train_image_path = os.path.join(root_path, f'dataset_train/af_train_img_seqtoseq_alll_{ts_length}i_{interval}.npy')
    train_label_path = os.path.join(root_path, f'dataset_train/af_train_label_seqtoseq_alll_{ts_length}i_{interval}.npy')
    
    if os.path.exists(train_image_path) and os.path.exists(train_label_path):
        visualize_dataset(
            train_image_path, 
            train_label_path, 
            'visualization/training',
            ts_length=ts_length
        )
    else:
        print(f"Training data not found at {train_image_path}")
    
    # Visualize validation data
    val_image_path = os.path.join(root_path, f'dataset_val/af_val_img_seqtoseq_alll_{ts_length}i_{interval}.npy')
    val_label_path = os.path.join(root_path, f'dataset_val/af_val_label_seqtoseq_alll_{ts_length}i_{interval}.npy')
    
    if os.path.exists(val_image_path) and os.path.exists(val_label_path):
        visualize_dataset(
            val_image_path, 
            val_label_path, 
            'visualization/validation',
            ts_length=ts_length
        )
    else:
        print(f"Validation data not found at {val_image_path}")