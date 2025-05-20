# main.py
import os
import argparse
import subprocess

def main(args):
    # Create directories
    os.makedirs('dataset/dataset_train', exist_ok=True)
    os.makedirs('dataset/dataset_val', exist_ok=True)
    os.makedirs('dataset/dataset_test', exist_ok=True)
    os.makedirs('visualization', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('evaluation_output', exist_ok=True)
    os.makedirs('feature_importance', exist_ok=True)
    
    # Step 1: Generate datasets
    if args.generate_data:
        print("\n=== Generating Training Dataset ===")
        subprocess.run([
            'python', 'dataset_gen_afba.py',
            '-mode', 'train',
            '-ts', str(args.ts_length),
            '-it', str(args.interval),
            '-uc', 'af',
            '-data_path', args.data_path,
            '-save_path', 'dataset'
        ])
        
        print("\n=== Generating Validation Dataset ===")
        subprocess.run([
            'python', 'dataset_gen_afba.py',
            '-mode', 'val',
            '-ts', str(args.ts_length),
            '-it', str(args.interval),
            '-uc', 'af',
            '-data_path', args.data_path,
            '-save_path', 'dataset'
        ])
        
        print("\n=== Generating Test Dataset ===")
        subprocess.run([
            'python', 'dataset_gen_afba.py',
            '-mode', 'test',
            '-ts', str(args.ts_length),
            '-it', str(args.interval),
            '-uc', 'af',
            '-data_path', args.data_path,
            '-save_path', 'dataset'
        ])
    
    # Step 2: Visualize dataset
    if args.visualize:
        print("\n=== Visualizing Dataset ===")
        subprocess.run(['python', 'visualize_dataset.py'])
    
    # Step 3: Train model
    if args.train:
        print("\n=== Training Model ===")
        subprocess.run([
            'python', 'train_active_fire.py',
            '--model', args.model,
            '--batch_size', str(args.batch_size),
            '--num_heads', str(args.num_heads),
            '--hidden_size', str(args.hidden_size),
            '--ts_length', str(args.ts_length),
            '--interval', str(args.interval),
            '--n_channel', str(args.n_channel),
            '--learning_rate', str(args.learning_rate),
            '--epochs', str(args.epochs),
            '--seed', str(args.seed),
            '--data_path', 'dataset',
            '--checkpoint_dir', 'saved_models'
        ])
    
    # Step 4: Evaluate model
    if args.evaluate:
        # Find best checkpoint
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            # Find the checkpoint with highest IoU in the name
            checkpoint_files = [f for f in os.listdir('saved_models') if f.endswith('.pth')]
            if checkpoint_files:
                # Sort by IoU score in filename
                best_checkpoint = sorted(checkpoint_files, 
                                          key=lambda x: float(x.split('iou_')[1].split('_')[0]) 
                                          if 'iou_' in x else 0, 
                                          reverse=True)[0]
                checkpoint_path = os.path.join('saved_models', best_checkpoint)
            else:
                print("No checkpoint found for evaluation. Please specify with --checkpoint.")
                return
        
        print(f"\n=== Evaluating Model using checkpoint: {checkpoint_path} ===")
        subprocess.run([
            'python', 'evaluate_active_fire.py',
            '--model', args.model,
            '--checkpoint', checkpoint_path,
            '--batch_size', str(args.batch_size),
            '--num_heads', str(args.num_heads),
            '--hidden_size', str(args.hidden_size),
            '--ts_length', str(args.ts_length),
            '--interval', str(args.interval),
            '--n_channel', str(args.n_channel),
            '--data_path', 'dataset',
            '--output_dir', 'evaluation_output'
        ])
    
    # Step 5: Analyze feature importance
    if args.analyze_features:
        # Use the same checkpoint as evaluation
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            # Find the checkpoint with highest IoU in the name
            checkpoint_files = [f for f in os.listdir('saved_models') if f.endswith('.pth')]
            if checkpoint_files:
                # Sort by IoU score in filename
                best_checkpoint = sorted(checkpoint_files, 
                                          key=lambda x: float(x.split('iou_')[1].split('_')[0]) 
                                          if 'iou_' in x else 0, 
                                          reverse=True)[0]
                checkpoint_path = os.path.join('saved_models', best_checkpoint)
            else:
                print("No checkpoint found for feature analysis. Please specify with --checkpoint.")
                return
        
        print(f"\n=== Analyzing Feature Importance using checkpoint: {checkpoint_path} ===")
        subprocess.run([
            'python', 'analyze_feature_importance.py',
            '--model', args.model,
            '--checkpoint', checkpoint_path,
            '--batch_size', str(args.batch_size),
            '--num_heads', str(args.num_heads),
            '--hidden_size', str(args.hidden_size),
            '--ts_length', str(args.ts_length),
            '--interval', str(args.interval),
            '--n_channel', str(args.n_channel),
            '--data_path', 'dataset',
            '--output_dir', 'feature_importance'
        ])
    
    print("\n=== All tasks completed! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Fire Detection Pipeline')
    
    # Task selection
    parser.add_argument('--generate_data', action='store_true', help='Generate datasets')
    parser.add_argument('--visualize', action='store_true', help='Visualize dataset')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--analyze_features', action='store_true', help='Analyze feature importance')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='unetr3d', help='Model name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension size')
    parser.add_argument('--ts_length', type=int, default=10, help='Time series length')
    parser.add_argument('--interval', type=int, default=3, help='Interval')
    parser.add_argument('--n_channel', type=int, default=8, help='Number of input channels')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Paths
    parser.add_argument('--data_path', type=str, default='/path/to/data', help='Path to data directory')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # If --all is specified, run all steps
    if args.all:
        args.generate_data = True
        args.visualize = True
        args.train = True
        args.evaluate = True
        args.analyze_features = True
    
    main(args)