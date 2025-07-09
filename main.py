import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import from our modules
from data_preprocessing import preprocess_mri_data
from model import get_model
from train import train_model, evaluate_model, plot_training_history, save_model
from augmentation import get_augmentation_transforms
from hyperparameter_tuning import grid_search

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Alzheimer's Disease Prediction from MRI Scans")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing MRI data')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['cnn', 'resnet'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer to use')
    
    # Augmentation and regularization
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Hyperparameter tuning
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_loader, val_loader, test_loader, class_names, num_classes = preprocess_mri_data(
        args.data_dir
    )
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    if args.tune:
        # Perform hyperparameter tuning
        print("Performing hyperparameter tuning...")
        param_grid = {
            'model_name': ['cnn', 'resnet'] if args.model == 'both' else [args.model],
            'optimizer': ['adam', 'sgd'],
            'learning_rate': [0.1, 0.01, 0.001],
            'use_scheduler': [True, False]
        }
        
        best_params, results = grid_search(
            param_grid, dataloaders, num_epochs=5, device=device
        )
        
        # Use best parameters
        model_name = best_params['model_name']
        optimizer_name = best_params['optimizer']
        learning_rate = best_params['learning_rate']
        use_scheduler = best_params['use_scheduler']
    else:
        # Use provided parameters
        model_name = args.model
        optimizer_name = args.optimizer
        learning_rate = args.lr
        use_scheduler = True
    
    # Create model
    print(f"Creating {model_name} model...")
    model = get_model(model_name, pretrained=args.pretrained)
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Define optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Define scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    print("Training model...")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        num_epochs=args.epochs, device=device
    )
    
    # Plot training history
    plot_training_history(history)
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Save metrics
    import json
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save model
    save_model(model, os.path.join(args.output_dir, 'models'), f'alzheimer_{model_name}.pth')
    
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
