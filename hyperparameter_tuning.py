import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

# Import from our modules
from data_preprocessing import preprocess_mri_data
from model import get_model
from train import train_model, evaluate_model

def grid_search(param_grid, dataloaders, num_epochs=5, device=None):
    """
    Perform grid search for hyperparameter tuning
    
    Args:
        param_grid: Dictionary with hyperparameter names and values
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        num_epochs: Number of epochs to train for each combination
        device: Device to train on
    
    Returns:
        best_params: Best hyperparameters
        results: All results from grid search
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate all combinations of hyperparameters
    grid = list(ParameterGrid(param_grid))
    print(f"Grid search with {len(grid)} combinations")
    
    results = []
    best_val_acc = 0
    best_params = None
    
    for i, params in enumerate(grid):
        print(f"\nCombination {i+1}/{len(grid)}")
        print(f"Parameters: {params}")
        
        # Create model
        model = get_model(params['model_name'])
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        elif params['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], 
                                 momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {params['optimizer']}")
        
        # Define scheduler
        scheduler = None
        if params['use_scheduler']:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Train model
        model, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler,
            num_epochs=num_epochs, device=device
        )
        
        # Get validation accuracy
        val_acc = max(history['val_acc'])
        
        # Record results
        result = {
            'params': params,
            'val_acc': val_acc,
            'history': history
        }
        results.append(result)
        
        # Update best parameters
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            
        print(f"Validation accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc - 0.001:  # Account for floating point errors
            print("New best!")
    
    print("\nGrid search complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return best_params, results

def plot_grid_search_results(results, param_name):
    """
    Plot grid search results for a specific parameter
    
    Args:
        results: Results from grid search
        param_name: Parameter name to plot
    """
    # Group results by parameter value
    param_values = sorted(set(result['params'][param_name] for result in results))
    accuracies = []
    
    for value in param_values:
        # Get all results with this parameter value
        matching_results = [r for r in results if r['params'][param_name] == value]
        # Calculate average accuracy
        avg_acc = np.mean([r['val_acc'] for r in matching_results])
        accuracies.append(avg_acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, accuracies, 'o-')
    plt.title(f'Effect of {param_name} on Validation Accuracy')
    plt.xlabel(param_name)
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    
    # Format x-axis for learning rate
    if param_name == 'learning_rate':
        plt.xscale('log')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = preprocess_mri_data("./data")
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Define parameter grid
    param_grid = {
        'model_name': ['cnn', 'resnet'],
        'optimizer': ['adam', 'sgd'],
        'learning_rate': [0.1, 0.01, 0.001],
        'use_scheduler': [True, False]
    }
    
    # Perform grid search
    best_params, results = grid_search(param_grid, dataloaders, num_epochs=3, device=device)
    
    # Plot results
    for param_name in param_grid.keys():
        plot_grid_search_results(results, param_name)
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    
    # Create model
    model = get_model(best_params['model_name'])
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    if best_params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'], 
                             momentum=0.9)
    
    # Define scheduler
    scheduler = None
    if best_params['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        num_epochs=10, device=device
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
