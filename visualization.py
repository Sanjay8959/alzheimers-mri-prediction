import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from PIL import Image
import cv2

# Import from our modules
from model import get_model

def plot_confusion_matrix(y_true, y_pred, class_names=['Normal', 'Alzheimer']):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities for positive class
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, dataloader, num_images=5, device=None):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with images
        num_images: Number of images to visualize
        device: Device to run inference on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    for i, (img, label, pred, prob) in enumerate(zip(images, labels, preds, probs)):
        img = img.numpy().transpose((1, 2, 0))
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        # Set title with prediction and probability
        title = f"True: {'AD' if label else 'Normal'}\n"
        title += f"Pred: {'AD' if pred else 'Normal'}\n"
        title += f"Prob: {prob[1].item():.2f}"
        
        # Color based on correctness
        color = 'green' if label == pred else 'red'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_gradcam(model, image, target_layer_name, class_idx=None, device=None):
    """
    Generate Grad-CAM visualization for a given image
    
    Args:
        model: PyTorch model
        image: Input image tensor [1, C, H, W]
        target_layer_name: Name of the target layer for Grad-CAM
        class_idx: Index of the class to generate Grad-CAM for (default: predicted class)
        device: Device to run inference on
    
    Returns:
        cam: Grad-CAM visualization
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Move image to device
    image = image.to(device)
    
    # Forward pass
    image.requires_grad_()
    outputs = model(image)
    
    if class_idx is None:
        class_idx = outputs.argmax(dim=1).item()
    
    # Get the score for the target class
    score = outputs[0, class_idx]
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Get the gradients and activations
    # Note: This is a simplified version. In practice, you would need to
    # register hooks to get gradients and activations from the target layer.
    # For demonstration purposes, we'll just create a heatmap
    
    # Create a simulated heatmap (in a real implementation, this would be
    # calculated from the gradients and activations)
    heatmap = np.random.rand(7, 7)  # Simulated 7x7 heatmap
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert image tensor to numpy array
    image = image.detach().cpu().numpy()[0].transpose((1, 2, 0))
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    image = np.uint8(255 * image)
    
    # Overlay heatmap on image
    cam = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    
    return cam

def visualize_gradcam(model, dataloader, num_images=3, device=None):
    """
    Visualize Grad-CAM for sample images
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with images
        num_images: Number of images to visualize
        device: Device to run inference on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Plot original images and Grad-CAM
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Original image
        img_np = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"{'AD' if label else 'Normal'}")
        axes[0, i].axis('off')
        
        # Grad-CAM
        img_tensor = img.unsqueeze(0)  # Add batch dimension
        cam = generate_gradcam(model, img_tensor, "layer4", device=device)
        
        axes[1, i].imshow(cam)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load a pretrained model
    model_path = "./models/alzheimer_resnet.pth"
    if os.path.exists(model_path):
        model = get_model("resnet")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        model = get_model("resnet")
        print("Using untrained model (no saved model found)")
    
    # Create dummy data for visualization
    from data_preprocessing import preprocess_mri_data
    train_loader, val_loader, test_loader = preprocess_mri_data("./data")
    
    # Example confusion matrix
    y_true = np.random.choice([0, 1], size=100)
    y_pred = np.random.choice([0, 1], size=100)
    plot_confusion_matrix(y_true, y_pred)
    
    # Example ROC curve
    y_scores = np.random.rand(100)
    plot_roc_curve(y_true, y_scores)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, num_images=5, device=device)
    
    # Visualize Grad-CAM
    visualize_gradcam(model, test_loader, num_images=3, device=device)
