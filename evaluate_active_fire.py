# evaluate_active_fire.py - Enhanced version with improved metrics
import argparse
import os
import numpy as np
import torch
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.data import DataLoader
from satimg_dataset_processor.data_generator_torch import Normalize, FireDataset
from spatial_models.unetr.unetr import UNETR
from spatial_models.swinunetr.swinunetr import SwinUNETR
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

def calculate_size_based_metrics(y_true, y_pred):
    """
    Calculate metrics based on fire cluster size
    
    Args:
        y_true: Ground truth fire mask
        y_pred: Predicted fire mask
    
    Returns:
        Dictionary of metrics for different fire sizes
    """
    from scipy import ndimage
    
    # Create size categories
    size_categories = {
        'small': (0, 25),      # Small fires (up to 25 pixels)
        'medium': (25, 100),   # Medium fires (25-100 pixels)
        'large': (100, np.inf) # Large fires (100+ pixels)
    }
    
    # Find connected components in ground truth
    labeled_true, num_features_true = ndimage.label(y_true)
    
    if num_features_true == 0:
        return {f"{size}_f1": 0.0 for size in size_categories}
    
    # Measure area of each fire cluster
    fire_sizes = ndimage.sum(y_true, labeled_true, range(1, num_features_true + 1))
    
    # Initialize metrics
    size_metrics = {}
    
    # Calculate metrics for each size category
    for size_name, (min_size, max_size) in size_categories.items():
        # Create mask for current size category
        size_mask = np.zeros_like(y_true)
        
        for i, size in enumerate(fire_sizes):
            if min_size <= size < max_size:
                size_mask[labeled_true == (i + 1)] = 1
        
        # If no fires in this category, skip
        if np.sum(size_mask) == 0:
            size_metrics[f"{size_name}_f1"] = np.nan
            continue
        
        # Calculate metrics for this size category
        intersection = np.logical_and(size_mask, y_pred).sum()
        union = np.logical_or(size_mask, y_pred).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
            
        precision = intersection / np.sum(y_pred) if np.sum(y_pred) > 0 else 0.0
        recall = intersection / np.sum(size_mask) if np.sum(size_mask) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        size_metrics[f"{size_name}_f1"] = f1
    
    return size_metrics

def analyze_false_positives(thermal_band, label, pred, output_dir, test_id, timestep):
    """
    Analyze false positives to identify common error patterns
    
    Args:
        thermal_band: Thermal IR band (I4)
        label: Ground truth fire mask
        pred: Predicted fire mask
        output_dir: Directory to save visualizations
        test_id: Test fire event ID
        timestep: Current timestep
    """
    # Create confusion matrix categories
    tp = np.logical_and(pred == 1, label == 1)  # True Positive
    fp = np.logical_and(pred == 1, label == 0)  # False Positive
    fn = np.logical_and(pred == 0, label == 1)  # False Negative
    
    # Skip if no false positives or false negatives
    if np.sum(fp) == 0 and np.sum(fn) == 0:
        return
    
    # Analyze thermal characteristics
    if np.sum(fp) > 0:
        fp_thermal_values = thermal_band[fp]
        fp_mean = np.mean(fp_thermal_values)
        fp_std = np.std(fp_thermal_values)
    else:
        fp_mean, fp_std = 0, 0
        
    if np.sum(tp) > 0:
        tp_thermal_values = thermal_band[tp]
        tp_mean = np.mean(tp_thermal_values)
        tp_std = np.std(tp_thermal_values)
    else:
        tp_mean, tp_std = 0, 0
    
    # Create visualization with analysis
    plt.figure(figsize=(14, 8))
    
    # Original thermal image
    plt.subplot(221)
    img_norm = (thermal_band - np.min(thermal_band)) / (np.max(thermal_band) - np.min(thermal_band))
    plt.imshow(img_norm, cmap='gray')
    plt.title(f"Thermal Band (I4)")
    plt.axis('off')
    
    # Confusion map
    plt.subplot(222)
    confusion_map = np.zeros((*thermal_band.shape, 3))
    confusion_map[tp] = [0, 1, 0]  # Green for TP
    confusion_map[fp] = [1, 0, 0]  # Red for FP
    confusion_map[fn] = [0, 0, 1]  # Blue for FN
    plt.imshow(img_norm, cmap='gray')
    plt.imshow(confusion_map, alpha=0.5)
    plt.title(f"Error Analysis")
    plt.axis('off')
    
    # Thermal histograms
    plt.subplot(223)
    if np.sum(tp) > 0 and np.sum(fp) > 0:
        plt.hist(tp_thermal_values, bins=20, alpha=0.5, color='green', label=f'True Fire (μ={tp_mean:.1f}, σ={tp_std:.1f})')
        plt.hist(fp_thermal_values, bins=20, alpha=0.5, color='red', label=f'False Positive (μ={fp_mean:.1f}, σ={fp_std:.1f})')
        plt.legend()
    plt.title("Thermal Value Distribution")
    plt.xlabel("Thermal Value")
    plt.ylabel("Pixel Count")
    
    # Thermal thresholds
    plt.subplot(224)
    fp_edge_mask = np.zeros_like(thermal_band, dtype=bool)
    fn_edge_mask = np.zeros_like(thermal_band, dtype=bool)
    
    # Check if false positives are near true fire boundaries (potential edge cases)
    if np.sum(fp) > 0 and np.sum(tp) > 0:
        from scipy import ndimage
        # Dilate true fire mask to find boundary region
        dilated = ndimage.binary_dilation(tp, iterations=2)
        boundary = np.logical_and(dilated, ~tp)
        # Find FPs that are in boundary
        fp_edge_mask = np.logical_and(fp, boundary)
        fp_edge_pct = np.sum(fp_edge_mask) / np.sum(fp) * 100 if np.sum(fp) > 0 else 0
        
        # Find FNs that are near true fire
        fn_edge_mask = np.logical_and(fn, boundary)
        fn_edge_pct = np.sum(fn_edge_mask) / np.sum(fn) * 100 if np.sum(fn) > 0 else 0
        
        plt.bar(['Edge FP', 'Other FP', 'Edge FN', 'Other FN'], 
               [np.sum(fp_edge_mask), np.sum(fp) - np.sum(fp_edge_mask),
                np.sum(fn_edge_mask), np.sum(fn) - np.sum(fn_edge_mask)])
        plt.title(f"Error Types\nEdge FP: {fp_edge_pct:.1f}%, Edge FN: {fn_edge_pct:.1f}%")
    
    # Save the error analysis visualization
    os.makedirs(os.path.join(output_dir, 'error_analysis'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'error_analysis', f"{test_id}_t{timestep}_errors.png"), 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    return {
        'fp_mean_thermal': fp_mean,
        'fp_std_thermal': fp_std,
        'tp_mean_thermal': tp_mean,
        'tp_std_thermal': tp_std,
        'fp_edge_pct': fp_edge_pct if np.sum(fp) > 0 else 0,
        'fn_edge_pct': fn_edge_pct if np.sum(fn) > 0 else 0
    }

def calculate_temporal_consistency(preds_time_series, labels_time_series):
    """
    Calculate temporal consistency metrics
    
    Args:
        preds_time_series: Predictions over time
        labels_time_series: Ground truth over time
        
    Returns:
        Dictionary of temporal metrics
    """
    if preds_time_series.shape[0] <= 1:
        return {'temporal_consistency': 0.0}
    
    # Calculate pixel-wise changes in predictions
    pred_changes = np.abs(preds_time_series[1:] - preds_time_series[:-1])
    
    # Focus on pixels that are fire in any time step
    fire_pixels_any = np.any(labels_time_series > 0, axis=0)
    
    if np.sum(fire_pixels_any) == 0:
        return {'temporal_consistency': 0.0}
    
    # Calculate average change rate for fire pixels
    pred_changes_fire = pred_changes[:, fire_pixels_any]
    stability = 1.0 - np.mean(pred_changes_fire)
    
    # Calculate temporal metrics for expanding/decreasing fires
    increasing_fire = np.sum(labels_time_series[1:], axis=(1, 2)) >= np.sum(labels_time_series[:-1], axis=(1, 2))
    
    # Separate metrics for growing and shrinking fires
    growing_stability = 1.0 - np.mean(pred_changes[increasing_fire]) if np.any(increasing_fire) else 0.0
    shrinking_stability = 1.0 - np.mean(pred_changes[~increasing_fire]) if np.any(~increasing_fire) else 0.0
    
    return {
        'temporal_consistency': stability,
        'growing_stability': growing_stability,
        'shrinking_stability': shrinking_stability
    }

def evaluate_active_fire(args):
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
            use_checkpoint=False,
            spatial_dims=3
        )
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle potential differences in state_dict keys (DataParallel vs regular model)
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        # If checkpoint was saved with DataParallel but we're using a regular model
        if not isinstance(model, torch.nn.DataParallel):
            # Remove 'module.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Normal case or model wrapped in DataParallel but checkpoint wasn't
        if isinstance(model, torch.nn.DataParallel):
            # Add 'module.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = 'module.' + k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # Post-processing
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # Test IDs
    test_ids = ['elephant_hill_fire', 'eagle_bluff_fire', 'double_creek_fire','sparks_lake_fire', 'lytton_fire', 
                'chuckegg_creek_fire', 'swedish_fire', 'sydney_fire', 'thomas_fire', 'tubbs_fire', 
                'carr_fire', 'camp_fire', 'creek_fire', 'blue_ridge_fire', 'dixie_fire', 'mosquito_fire', 'calfcanyon_fire']
    
    results = {}
    overall_metrics = {
        'f1': [], 
        'iou': [], 
        'precision': [], 
        'recall': [],
        'small_f1': [],
        'medium_f1': [],
        'large_f1': [],
        'temporal_consistency': []
    }
    
    # Track error patterns
    error_patterns = {
        'fp_mean_thermal': [],
        'fp_edge_pct': []
    }
    
    # Evaluate each test fire event
    for test_id in test_ids:
        print(f"Evaluating {test_id}...")
        
        # Set up data paths
        test_image_path = os.path.join(root_path, f'dataset_test/af_{test_id}_img_seqtoseql_{ts_length}i_{interval}.npy')
        test_label_path = os.path.join(root_path, f'dataset_test/af_{test_id}_label_seqtoseql_{ts_length}i_{interval}.npy')
        
        if not os.path.exists(test_image_path) or not os.path.exists(test_label_path):
            print(f"Test data not found for {test_id}. Skipping...")
            continue
        
        # Create dataset and dataloader
        test_dataset = FireDataset(
            image_path=test_image_path,
            label_path=test_label_path,
            ts_length=ts_length,
            transform=transform,
            n_channel=n_channel,
            label_sel=2  # For active fire detection
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluation metrics
        f1_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        temporal_metrics_all = []
        size_based_metrics = {
            'small_f1': [],
            'medium_f1': [],
            'large_f1': []
        }
        
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                test_data_batch = batch['data']
                test_labels_batch = batch['labels']
                test_data_batch = test_data_batch.to(device)
                test_labels_batch = test_labels_batch.to(device)
                
                # Forward pass
                outputs = model(test_data_batch)
                
                # Post-process outputs
                outputs_post = torch.sigmoid(outputs)
                outputs_bin = (outputs_post > 0.5).float()
                
                # Calculate metrics for each sample in batch
                for k in range(test_data_batch.shape[0]):
                    # Store predictions and labels for temporal consistency
                    pred_time_series = outputs_bin[k, 1].cpu().numpy()
                    label_time_series = test_labels_batch[k, 1].cpu().numpy() > 0
                    
                    # Calculate temporal consistency
                    temp_metrics = calculate_temporal_consistency(pred_time_series, label_time_series)
                    temporal_metrics_all.append(temp_metrics['temporal_consistency'])
                    
                    # Calculate metrics for each timestep
                    for t in range(ts_length):
                        output_ti = outputs_bin[k, 1, t].cpu().numpy()
                        label_ti = test_labels_batch[k, 1, t].cpu().numpy() > 0
                        
                        # Skip if no fire in ground truth
                        if np.sum(label_ti) == 0:
                            continue
                        
                        # Calculate metrics
                        f1 = f1_score(label_ti.flatten(), output_ti.flatten(), zero_division=1.0)
                        iou = jaccard_score(label_ti.flatten(), output_ti.flatten(), zero_division=1.0)
                        precision = precision_score(label_ti.flatten(), output_ti.flatten(), zero_division=1.0)
                        recall = recall_score(label_ti.flatten(), output_ti.flatten(), zero_division=1.0)
                        
                        f1_scores.append(f1)
                        iou_scores.append(iou)
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        
                        # Calculate size-based metrics
                        size_metrics = calculate_size_based_metrics(label_ti, output_ti)
                        for size_name, value in size_metrics.items():
                            if not np.isnan(value):
                                if size_name in size_based_metrics:
                                    size_based_metrics[size_name].append(value)
                        
                        # Error analysis for sample cases
                        if i == 0 and k == 0 and t % 2 == 0:  # Every other timestep for first batch
                            # Get thermal band (Band I4)
                            thermal_band = test_data_batch[k, 3, t].cpu().numpy()
                            
                            # Analyze errors
                            error_data = analyze_false_positives(
                                thermal_band, label_ti, output_ti, 
                                output_dir, test_id, t
                            )
                            
                            if error_data:
                                for key, val in error_data.items():
                                    if key in error_patterns:
                                        error_patterns[key].append(val)
                        
                        # Create visualization
                        if i == 0 and k == 0 and t % 2 == 0:  # Visualize every other timestep
                            plt.figure(figsize=(16, 12))
                            
                            # Display input image (band I4 - thermal)
                            plt.subplot(2, 3, 1)
                            img = test_data_batch[k, 3, t].cpu().numpy()
                            img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
                            plt.imshow(img_norm, cmap='gray')
                            plt.title("Input Image (Thermal IR - Band I4)")
                            plt.axis('off')
                            
                            # Near IR band (Band I2)
                            plt.subplot(2, 3, 2)
                            img_nir = test_data_batch[k, 1, t].cpu().numpy()
                            img_nir_norm = (img_nir - np.min(img_nir)) / (np.max(img_nir) - np.min(img_nir))
                            plt.imshow(img_nir_norm, cmap='gray')
                            plt.title("Near IR (Band I2)")
                            plt.axis('off')
                            
                            # Display ground truth
                            plt.subplot(2, 3, 3)
                            plt.imshow(img_norm, cmap='gray')
                            plt.imshow(np.ma.masked_where(label_ti == 0, label_ti), cmap='hot', alpha=0.7)
                            plt.title("Ground Truth")
                            plt.axis('off')

                            # Display prediction
                            plt.subplot(2, 3, 4)
                            plt.imshow(img_norm, cmap='gray')
                            plt.imshow(np.ma.masked_where(output_ti == 0, output_ti), cmap='hot', alpha=0.7)
                            plt.title(f"Prediction (F1: {f1:.3f}, IoU: {iou:.3f})")
                            plt.axis('off')
                            
                            # Show confusion map (TP, FP, FN)
                            plt.subplot(2, 3, 5)
                            plt.imshow(img_norm, cmap='gray')
                            
                            # True Positive (Green)
                            tp_mask = np.logical_and(output_ti == 1, label_ti == 1)
                            if np.any(tp_mask):
                                plt.imshow(np.ma.masked_where(tp_mask == 0, tp_mask), cmap='Greens', alpha=0.7)
                            
                            # False Positive (Red)
                            fp_mask = np.logical_and(output_ti == 1, label_ti == 0)
                            if np.any(fp_mask):
                                plt.imshow(np.ma.masked_where(fp_mask == 0, fp_mask), cmap='Reds', alpha=0.7)
                            
                            # False Negative (Blue)
                            fn_mask = np.logical_and(output_ti == 0, label_ti == 1)
                            if np.any(fn_mask):
                                plt.imshow(np.ma.masked_where(fn_mask == 0, fn_mask), cmap='Blues', alpha=0.7)
                            
                            plt.title(f"Error Analysis")
                            plt.legend([
                                plt.Rectangle((0, 0), 1, 1, fc="g", alpha=0.7),
                                plt.Rectangle((0, 0), 1, 1, fc="r", alpha=0.7),
                                plt.Rectangle((0, 0), 1, 1, fc="b", alpha=0.7)
                            ], ['True Positive', 'False Positive', 'False Negative'], 
                               loc='lower right')
                            plt.axis('off')
                            
                            # Size-based metrics
                            plt.subplot(2, 3, 6)
                            size_metrics_dict = {k: v for k, v in size_metrics.items() if not np.isnan(v)}
                            if size_metrics_dict:
                                plt.bar(range(len(size_metrics_dict)), list(size_metrics_dict.values()))
                                plt.xticks(range(len(size_metrics_dict)), [k.split('_')[0].capitalize() for k in size_metrics_dict.keys()])
                                plt.title("F1 Score by Fire Size")
                                plt.ylim(0, 1.0)
                            else:
                                plt.text(0.5, 0.5, "No size metrics available", 
                                         horizontalalignment='center',
                                         verticalalignment='center')
                                plt.title("Size Metrics")
                                plt.axis('off')
                            
                            # Save the visualization
                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, f"{test_id}_timestep_{t}_evaluation.png"), 
                                       bbox_inches='tight', dpi=150)
                            plt.close()
        
        # Calculate average metrics for this fire event
        if f1_scores:
            avg_f1 = np.mean(f1_scores)
            avg_iou = np.mean(iou_scores)
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            avg_temporal = np.mean(temporal_metrics_all) if temporal_metrics_all else 0.0
            
            # Size-based metrics
            size_avgs = {}
            for size_name, values in size_based_metrics.items():
                if values:
                    size_avgs[size_name] = np.mean(values)
                else:
                    size_avgs[size_name] = np.nan
            
            results[test_id] = {
                'f1_score': avg_f1,
                'iou_score': avg_iou,
                'precision': avg_precision,
                'recall': avg_recall,
                'temporal_consistency': avg_temporal,
                **size_avgs
            }
            
            # Update overall metrics
            overall_metrics['f1'].append(avg_f1)
            overall_metrics['iou'].append(avg_iou)
            overall_metrics['precision'].append(avg_precision)
            overall_metrics['recall'].append(avg_recall)
            overall_metrics['temporal_consistency'].append(avg_temporal)
            
            for size_name, values in size_based_metrics.items():
                if values:
                    overall_metrics[size_name].append(np.mean(values))
            
            print(f"{test_id}: F1={avg_f1:.4f}, IoU={avg_iou:.4f}, P={avg_precision:.4f}, R={avg_recall:.4f}, TC={avg_temporal:.4f}")
            
            # Print size-based metrics if available
            size_str = ", ".join([f"{k.split('_')[0]}={v:.4f}" for k, v in size_avgs.items() if not np.isnan(v)])
            if size_str:
                print(f"Size metrics: {size_str}")
        else:
            print(f"No fire pixels detected in {test_id}. Skipping metrics calculation.")
    
    # Calculate overall metrics
    if results:
        overall_f1 = np.mean(overall_metrics['f1'])
        overall_iou = np.mean(overall_metrics['iou'])
        overall_precision = np.mean(overall_metrics['precision'])
        overall_recall = np.mean(overall_metrics['recall'])
        overall_temporal = np.mean(overall_metrics['temporal_consistency'])
        
        # Size-based overall metrics
        size_overall = {}
        for size_name in ['small_f1', 'medium_f1', 'large_f1']:
            if overall_metrics[size_name]:
                size_overall[size_name] = np.mean(overall_metrics[size_name])
            else:
                size_overall[size_name] = np.nan
        
        print("\nOverall Results:")
        print(f"Average F1 Score: {overall_f1:.4f}")
        print(f"Average IoU Score: {overall_iou:.4f}")
        print(f"Average Precision: {overall_precision:.4f}")
        print(f"Average Recall: {overall_recall:.4f}")
        print(f"Average Temporal Consistency: {overall_temporal:.4f}")
        
        # Print size-based metrics if available
        size_str = ", ".join([f"{k.split('_')[0]} F1={v:.4f}" for k, v in size_overall.items() if not np.isnan(v)])
        if size_str:
            print(f"Size metrics: {size_str}")
        
        # Save results to file
        with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n\n")
            f.write("Individual Results:\n")
            
            for test_id, metrics in results.items():
                f.write(f"{test_id}: F1={metrics['f1_score']:.4f}, IoU={metrics['iou_score']:.4f}, " +
                        f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, " +
                        f"TC={metrics['temporal_consistency']:.4f}\n")
                
                # Add size-based metrics if available
                size_str = ", ".join([f"{k.split('_')[0]}={v:.4f}" for k, v in metrics.items() 
                                      if k.endswith('_f1') and k != 'f1_score' and not np.isnan(v)])
                if size_str:
                    f.write(f"   Size metrics: {size_str}\n")
            
            f.write("\nOverall Results:\n")
            f.write(f"Average F1 Score: {overall_f1:.4f}\n")
            f.write(f"Average IoU Score: {overall_iou:.4f}\n")
            f.write(f"Average Precision: {overall_precision:.4f}\n")
            f.write(f"Average Recall: {overall_recall:.4f}\n")
            f.write(f"Average Temporal Consistency: {overall_temporal:.4f}\n")
            
            # Add size-based metrics if available
            size_str = ", ".join([f"{k.split('_')[0]} F1={v:.4f}" for k, v in size_overall.items() if not np.isnan(v)])
            if size_str:
                f.write(f"Size metrics: {size_str}\n")
        
        # Create visualization for overall results
        plt.figure(figsize=(10, 6))
        
        # Create bar chart for overall metrics
        metrics_names = ['F1', 'IoU', 'Precision', 'Recall', 'Temporal\nConsistency']
        metrics_values = [overall_f1, overall_iou, overall_precision, overall_recall, overall_temporal]
        
        plt.bar(range(len(metrics_names)), metrics_values)
        plt.ylim(0, 1.0)
        plt.xticks(range(len(metrics_names)), metrics_names)
        plt.title(f"Overall Performance ({model_name})")
        plt.ylabel("Score")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(output_dir, "overall_metrics.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create a heatmap for fire event comparison
        plt.figure(figsize=(12, 8))
        
        # Convert results to DataFrame for better visualization
        metrics_df = pd.DataFrame({
            'F1 Score': [results[id]['f1_score'] for id in results],
            'IoU': [results[id]['iou_score'] for id in results],
            'Precision': [results[id]['precision'] for id in results],
            'Recall': [results[id]['recall'] for id in results],
            'Temporal Consistency': [results[id]['temporal_consistency'] for id in results]
        }, index=[id.replace('_fire', '') for id in results])
        
        # Plot heatmap
        sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.3f', vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        plt.title(f"Performance Across Fire Events ({model_name})")
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "fire_comparison_heatmap.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Fire size performance comparison
        if any(not np.isnan(size_overall[k]) for k in size_overall):
            plt.figure(figsize=(8, 5))
            
            # Extract valid size metrics
            valid_sizes = [k for k in size_overall if not np.isnan(size_overall[k])]
            size_values = [size_overall[k] for k in valid_sizes]
            size_labels = [k.split('_')[0].capitalize() for k in valid_sizes]
            
            plt.bar(range(len(size_labels)), size_values)
            plt.ylim(0, 1.0)
            plt.xticks(range(len(size_labels)), size_labels)
            plt.title(f"Performance by Fire Size ({model_name})")
            plt.ylabel("F1 Score")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.savefig(os.path.join(output_dir, "size_comparison.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Analyze error patterns
        if error_patterns['fp_mean_thermal']:
            plt.figure(figsize=(10, 5))
            
            # Get test IDs with error data
            error_ids = [id.replace('_fire', '') for id in test_ids if any(id in t for t in [os.path.basename(path) for path in os.listdir(os.path.join(output_dir, 'error_analysis'))])]
            
            # Scatter plot of false positive rates vs thermal values
            plt.scatter([error_patterns['fp_mean_thermal'][i] for i in range(len(error_patterns['fp_mean_thermal']))], 
                        [error_patterns['fp_edge_pct'][i] for i in range(len(error_patterns['fp_edge_pct']))],
                        s=80, alpha=0.7)
            
            # Add fire ID labels
            for i in range(len(error_ids)):
                if i < len(error_patterns['fp_mean_thermal']):
                    plt.annotate(error_ids[i], 
                                (error_patterns['fp_mean_thermal'][i], error_patterns['fp_edge_pct'][i]),
                                xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel("Mean Thermal Value of False Positives")
            plt.ylabel("% of False Positives at Fire Edges")
            plt.title("Error Pattern Analysis")
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(output_dir, "error_pattern_analysis.png"), dpi=150, bbox_inches='tight')
            plt.close()
    
    else:
        print("No test data was evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Active Fire Detection model')
    parser.add_argument('--model', type=str, default='swinunetr', help='Model name (unetr3d or swinunetr)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension size')
    parser.add_argument('--ts_length', type=int, default=10, help='Time series length')
    parser.add_argument('--interval', type=int, default=3, help='Interval')
    parser.add_argument('--n_channel', type=int, default=8, help='Number of input channels')
    parser.add_argument('--data_path', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_output', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    evaluate_active_fire(args)