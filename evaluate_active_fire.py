# evaluate_active_fire.py
import argparse
import os
import numpy as np
import torch
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.data import DataLoader
from satimg_dataset_processor.data_generator_torch import Normalize, FireDataset
from spatial_models.unetr.unetr import UNETR
from sklearn.metrics import f1_score, jaccard_score
import matplotlib.pyplot as plt

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
    
    # Post-processing
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # Test IDs
    test_ids = ['elephant_hill_fire', 'eagle_bluff_fire', 'double_creek_fire','sparks_lake_fire', 'lytton_fire', 
                'chuckegg_creek_fire', 'swedish_fire', 'sydney_fire', 'thomas_fire', 'tubbs_fire', 
                'carr_fire', 'camp_fire', 'creek_fire', 'blue_ridge_fire', 'dixie_fire', 'mosquito_fire', 'calfcanyon_fire']
    
    results = {}
    overall_f1 = 0
    overall_iou = 0
    
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
        
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                test_data_batch = batch['data']
                test_labels_batch = batch['labels']
                test_data_batch = test_data_batch.to(device)
                test_labels_batch = test_labels_batch.to(device)
                
                # Forward pass
                outputs = model(test_data_batch)
                outputs = [post_trans(i) for i in decollate_batch(outputs)]
                outputs = torch.stack(outputs).cpu().numpy()
                
                # Calculate metrics for each timestep
                for k in range(test_data_batch.shape[0]):
                    for t in range(ts_length):
                        output_ti = outputs[k, 1, t, :, :] > 0.5
                        label = test_labels_batch[k, 1, t, :, :].cpu().numpy() > 0
                        
                        # Calculate F1 and IoU scores
                        f1_ts = f1_score(label.flatten(), output_ti.flatten(), zero_division=1.0)
                        iou_ts = jaccard_score(label.flatten(), output_ti.flatten(), zero_division=1.0)
                        
                        f1_scores.append(f1_ts)
                        iou_scores.append(iou_ts)
                        
                        # Create visualization
                        if i == 0 and k == 0 and t % 2 == 0:  # Visualize every other timestep for the first batch
                            plt.figure(figsize=(12, 4))
                            
                            # Display input image (band I4 - thermal)
                            plt.subplot(1, 3, 1)
                            img = test_data_batch[k, 3, t].cpu().numpy()
                            img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
                            plt.imshow(img_norm, cmap='gray')
                            plt.title("Input Image (Band I4)")
                            plt.axis('off')
                            
                            # Display ground truth
                            plt.subplot(1, 3, 2)
                            plt.imshow(img_norm, cmap='gray')
                            plt.imshow(np.ma.masked_where(label == 0, label), cmap='hot', alpha=0.7)
                            plt.title("Ground Truth")
                            plt.axis('off')

                            # Display prediction
                            plt.subplot(1, 3, 3)
                            plt.imshow(img_norm, cmap='gray')
                            plt.imshow(np.ma.masked_where(output_ti == 0, output_ti), cmap='hot', alpha=0.7)
                            plt.title("Prediction")
                            plt.axis('off')
                            
                            # Show confusion map (TP, FP, FN)
                            plt.figure(figsize=(8, 6))
                            plt.imshow(img_norm, cmap='gray')
                            
                            # True Positive (Green)
                            tp_mask = np.logical_and(output_ti == 1, label == 1)
                            if np.any(tp_mask):
                                plt.imshow(np.ma.masked_where(tp_mask == 0, tp_mask), cmap='Greens', alpha=0.7)
                            
                            # False Positive (Red)
                            fp_mask = np.logical_and(output_ti == 1, label == 0)
                            if np.any(fp_mask):
                                plt.imshow(np.ma.masked_where(fp_mask == 0, fp_mask), cmap='Reds', alpha=0.7)
                            
                            # False Negative (Blue)
                            fn_mask = np.logical_and(output_ti == 0, label == 1)
                            if np.any(fn_mask):
                                plt.imshow(np.ma.masked_where(fn_mask == 0, fn_mask), cmap='Blues', alpha=0.7)
                            
                            plt.title(f"Confusion Map (F1: {f1_ts:.3f}, IoU: {iou_ts:.3f})")
                            plt.legend([
                                plt.Rectangle((0, 0), 1, 1, fc="g", alpha=0.7),
                                plt.Rectangle((0, 0), 1, 1, fc="r", alpha=0.7),
                                plt.Rectangle((0, 0), 1, 1, fc="b", alpha=0.7)
                            ], ['True Positive', 'False Positive', 'False Negative'], 
                               loc='lower right')
                            plt.axis('off')
                            
                            # Save the visualization
                            plt.savefig(os.path.join(output_dir, f"{test_id}_timestep_{t}_evaluation.png"), 
                                       bbox_inches='tight', dpi=150)
                            plt.close()
        
        # Calculate average metrics for this fire event
        avg_f1 = np.mean(f1_scores)
        avg_iou = np.mean(iou_scores)
        
        results[test_id] = {
            'f1_score': avg_f1,
            'iou_score': avg_iou
        }
        
        overall_f1 += avg_f1
        overall_iou += avg_iou
        
        print(f"{test_id}: F1 Score = {avg_f1:.4f}, IoU Score = {avg_iou:.4f}")
    
    # Calculate overall metrics
    if test_ids:
        overall_f1 /= len(test_ids)
        overall_iou /= len(test_ids)
        
        print("\nOverall Results:")
        print(f"Average F1 Score: {overall_f1:.4f}")
        print(f"Average IoU Score: {overall_iou:.4f}")
        
        # Save results to file
        with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n\n")
            f.write("Individual Results:\n")
            
            for test_id in results:
                f.write(f"{test_id}: F1 Score = {results[test_id]['f1_score']:.4f}, IoU Score = {results[test_id]['iou_score']:.4f}\n")
            
            f.write("\nOverall Results:\n")
            f.write(f"Average F1 Score: {overall_f1:.4f}\n")
            f.write(f"Average IoU Score: {overall_iou:.4f}\n")
        
        # Create bar chart for all test events
        plt.figure(figsize=(14, 8))
        
        # Extract data for plotting
        ids = list(results.keys())
        f1_values = [results[id]['f1_score'] for id in ids]
        iou_values = [results[id]['iou_score'] for id in ids]
        
        x = np.arange(len(ids))
        width = 0.35
        
        # Create grouped bar chart
        plt.bar(x - width/2, f1_values, width, label='F1 Score')
        plt.bar(x + width/2, iou_values, width, label='IoU Score')
        
        plt.axhline(y=overall_f1, color='blue', linestyle='--', alpha=0.7, label=f'Avg F1: {overall_f1:.3f}')
        plt.axhline(y=overall_iou, color='orange', linestyle='--', alpha=0.7, label=f'Avg IoU: {overall_iou:.3f}')
        
        plt.ylabel('Score')
        plt.title('Model Performance on Test Fire Events')
        plt.xticks(x, ids, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=150)
        plt.close()
    else:
        print("No test data was evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Active Fire Detection model')
    parser.add_argument('--model', type=str, default='unetr3d', help='Model name')
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