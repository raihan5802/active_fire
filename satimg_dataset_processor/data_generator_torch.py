# Updated satimg_dataset_processor/data_generator_torch.py
import os
import random
import numpy as np
import torch
from monai.metrics import MeanIoU, DiceMetric
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample, label, augmentation=True):
        # Basic augmentations (flips and rotations)
        if augmentation:
            # Original augmentations
            hflip = bool(np.random.random() > 0.5)
            vflip = bool(np.random.random() > 0.5)
            rotate = int(np.floor(np.random.random() * 4))
            
            if hflip:
                sample = TF.hflip(sample)
                label = TF.hflip(label)

            if vflip:
                sample = TF.vflip(sample)
                label = TF.vflip(label)

            if rotate != 0:
                angle = rotate * 90
                sample = TF.rotate(sample, angle)
                label = TF.rotate(label, angle)
                
            # Enhanced augmentations
            
            # 1. Brightness/contrast variations
            if np.random.random() > 0.7:
                brightness_factor = np.random.uniform(0.8, 1.2)
                for i in range(sample.shape[0]):
                    # Apply only to visible bands (indices 0, 1, 2)
                    if i < 3:
                        sample[i] = sample[i] * brightness_factor
            
            # 2. Random masking of timestamps (improve robustness to missing data)
            if np.random.random() > 0.8:
                # Randomly mask out some time points
                time_mask_idx = np.random.randint(0, sample.shape[1])
                # Create a copy of nearby time point
                if time_mask_idx > 0:
                    replace_idx = time_mask_idx - 1
                else:
                    replace_idx = time_mask_idx + 1
                    
                if replace_idx < sample.shape[1]:
                    sample[:, time_mask_idx] = sample[:, replace_idx]
            
            # 3. Simulated cloud coverage (only for optical bands)
            if np.random.random() > 0.7:
                # Create a cloud mask (small elliptical shape)
                h, w = sample.shape[2], sample.shape[3]
                center_h = np.random.randint(0, h)
                center_w = np.random.randint(0, w)
                
                # Define cloud size
                cloud_h = np.random.randint(h // 10, h // 4)
                cloud_w = np.random.randint(w // 10, w // 4)
                
                # Create cloud mask
                y, x = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
                cloud_mask = ((x*x)/(cloud_w*cloud_w) + (y*y)/(cloud_h*cloud_h)) <= 1
                
                # Apply cloud to visible and near-IR bands (0, 1, 2)
                cloud_value = 0.8  # High reflectance for clouds
                for i in range(3):  # Visible bands
                    temp = sample[i].clone()
                    for t in range(temp.shape[0]):
                        # Only apply to some timestamps
                        if np.random.random() > 0.5:
                            temp[t][cloud_mask] = cloud_value
                    sample[i] = temp

        # Standardize channels
        for i in range(len(self.mean)):
            sample[i, :, ...] = (sample[i, :, ...] - self.mean[i]) / self.std[i]
        
        return sample, label

class FireDataset(Dataset):
    def __init__(self, image_path, label_path, ts_length=8, transform=None, n_channel=8, label_sel=0, weighted_sampling=True):
        self.image_path, self.label_path = image_path, label_path
        self.num_samples = np.load(self.image_path).shape[0]
        self.transform = transform
        self.n_channel = n_channel
        self.label_sel = label_sel
        self.ts_length = ts_length
        if 'train' in self.label_path:
            self.augmentation = True
        else:
            self.augmentation = False
            
        # For weighted sampling
        self.weighted_sampling = weighted_sampling
        if weighted_sampling and 'train' in self.label_path:
            # Calculate sample weights based on fire presence
            self.sample_weights = self._calculate_sample_weights()
        
    def _calculate_sample_weights(self):
        """
        Calculate sample weights to address class imbalance.
        Samples with fire pixels get higher weights.
        """
        label_data = np.load(self.label_path, mmap_mode='r')
        
        # Extract fire labels (label_sel=2 for active fire detection)
        fire_labels = label_data[:, self.label_sel, :, :, :]
        
        # Calculate fire presence ratio per sample
        sample_weights = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            # Count fire pixels in each sample
            fire_pixels = np.sum(fire_labels[i] > 0)
            total_pixels = fire_labels[i].size
            
            # Calculate fire ratio 
            fire_ratio = fire_pixels / total_pixels
            
            # Apply weight (giving more weight to samples with fire)
            if fire_ratio > 0:
                # Samples with fire get higher weights (up to 10x for samples with many fire pixels)
                sample_weights[i] = 1.0 + min(9.0, fire_ratio * 100.0)
            else:
                sample_weights[i] = 1.0
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
        
        return sample_weights

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load a chunk of data from disk
        data_chunk, label_chunk = self.load_data(idx)
        if self.transform:
            data_chunk, label_chunk = self.transform(data_chunk, label_chunk, self.augmentation)
        sample = {
            'data': data_chunk,
            'labels': label_chunk,
        }

        return sample

    # define a function to load a batch of data from disk
    def load_data(self, indices):
        # load a chunk of data from disk
        data_chunk = np.load(self.image_path, mmap_mode='r')[indices]
        label_chunk = np.load(self.label_path, mmap_mode='r')[indices]

        if self.n_channel==6:
            img_dataset = data_chunk[2:, :, :, :]
        else:
            img_dataset = data_chunk[:, :, :, :]
        label_dataset = label_chunk[[self.label_sel], :, :, :]
        # 0 NIFC 1 VIIRS AF ACC 2 combine
        y_dataset = np.zeros((2, self.ts_length, 256,256))
        # y_dataset = np.where(label_dataset > 0, 1, 0)
        # y_dataset = np.where(af_dataset > 0, 2, y_dataset)
        y_dataset[0, :, :, :] = label_dataset[..., :] == 0
        y_dataset[1, :, :, :] = label_dataset[..., :] > 0

        x_array, y_array = img_dataset, y_dataset
        x_array_copy = x_array.copy()
        # convert the data to a PyTorch tensor
        x = torch.squeeze(torch.from_numpy(x_array_copy))
        y = torch.squeeze(torch.from_numpy(y_array)).long()

        return x, y

class WeightedFireDataLoader(DataLoader):
    """
    Custom DataLoader that uses weighted sampling to address class imbalance
    """
    def __init__(self, dataset, batch_size=4, shuffle=True, **kwargs):
        if shuffle and hasattr(dataset, 'weighted_sampling') and dataset.weighted_sampling:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=dataset.sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            super().__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)
        else:
            super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

if __name__ == '__main__':
    root_path = '/home/z/h/zhao2/TS-SatFire/dataset/'
    mode = 'ba'
    interval = 3
    ts_length = 6
    image_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
    label_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
    transform = Normalize(mean = [18.76488,27.441864,20.584806,305.99478,294.31738,14.625097,276.4207,275.16766],
                        std = [15.911591,14.879259,10.832616,21.761852,24.703484,9.878246,40.64329,40.7657])
    train_dataset = FireDataset(image_path=image_path, label_path=label_path, transform=transform, ts_length=ts_length, n_channel=8)
    train_dataloader = WeightedFireDataLoader(train_dataset, batch_size=4, shuffle=True)
    print(next(iter(train_dataloader)).get('data').shape)