import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import torchvision.transforms as transforms

# Import from our modules
from model import get_model

def preprocess_image(image_path):
    """
    Preprocess a single image for inference
    
    Args:
        image_path: Path to the image file
    
    Returns:
        tensor: Preprocessed image tensor
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    try:
        # Try loading as a 2D image first
        image = Image.open(image_path).convert('RGB')
    except:
        # If it's a NIfTI file, extract a middle slice
        try:
            nifti_img = nib.load(image_path)
            img_data = nifti_img.get_fdata()
            
            # Extract middle slice from sagittal view
            slice_idx = img_data.shape[0] // 2
            slice_data = img_data[slice_idx, :, :]
            
            # Normalize to 0-255 and convert to PIL
            slice_data = ((slice_data - slice_data.min()) / 
                         (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            image = Image.fromarray(slice_data).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading {image_path}: {e}")
    
    # Apply transformations
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

def predict(model, image_tensor, device=None):
    """
    Make a prediction for a single image
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
    
    Returns:
        pred_class: Predicted class (0: Normal, 1: Alzheimer's)
        probability: Probability of the predicted class
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0, pred_class].item()
    
    return pred_class, probability

def visualize_prediction(image_path, pred_class, probability):
    """
    Visualize the image with prediction
    
    Args:
        image_path: Path to the image file
        pred_class: Predicted class (0: Normal, 1: Alzheimer's)
        probability: Probability of the predicted class
    """
    # Load image
    try:
        # Try loading as a 2D image first
        image = Image.open(image_path).convert('RGB')
    except:
        # If it's a NIfTI file, extract a middle slice
        try:
            nifti_img = nib.load(image_path)
            img_data = nifti_img.get_fdata()
            
            # Extract middle slice from sagittal view
            slice_idx = img_data.shape[0] // 2
            slice_data = img_data[slice_idx, :, :]
            
            # Normalize to 0-255 and convert to PIL
            slice_data = ((slice_data - slice_data.min()) / 
                         (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            image = Image.fromarray(slice_data).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading {image_path}: {e}")
    
    # Plot image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    
    # Set title with prediction and probability
    class_name = 'Alzheimer\'s' if pred_class == 1 else 'Normal'
    title = f"Prediction: {class_name}\n"
    title += f"Probability: {probability:.2f}"
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def batch_inference(model, image_dir, output_dir=None, device=None):
    """
    Run inference on a batch of images
    
    Args:
        model: PyTorch model
        image_dir: Directory containing images
        output_dir: Directory to save results
        device: Device to run inference on
    
    Returns:
        results: Dictionary with results for each image
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.nii', '.nii.gz')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images")
    
    results = {}
    
    for image_path in image_files:
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path)
            
            # Make prediction
            pred_class, probability = predict(model, image_tensor, device)
            
            # Store result
            results[image_path] = {
                'class': pred_class,
                'class_name': 'Alzheimer\'s' if pred_class == 1 else 'Normal',
                'probability': probability
            }
            
            print(f"Processed {image_path}: {results[image_path]['class_name']} ({probability:.2f})")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        import csv
        with open(os.path.join(output_dir, 'results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Prediction', 'Probability'])
            
            for image_path, result in results.items():
                writer.writerow([
                    os.path.basename(image_path),
                    result['class_name'],
                    result['probability']
                ])
        
        print(f"Results saved to {os.path.join(output_dir, 'results.csv')}")
    
    return results

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Inference for Alzheimer's disease prediction")
    parser.add_argument('--model_path', type=str, default='./models/alzheimer_resnet.pth',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='resnet',
                        choices=['cnn', 'resnet'],
                        help='Type of model architecture')
    parser.add_argument('--image_path', type=str,
                        help='Path to a single image for inference')
    parser.add_argument('--image_dir', type=str,
                        help='Directory containing images for batch inference')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(args.model_type)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model file {args.model_path} not found. Using untrained model.")
    
    # Run inference
    if args.image_path:
        # Single image inference
        try:
            image_tensor = preprocess_image(args.image_path)
            pred_class, probability = predict(model, image_tensor, device)
            
            print(f"Prediction: {'Alzheimer\'s' if pred_class == 1 else 'Normal'}")
            print(f"Probability: {probability:.2f}")
            
            visualize_prediction(args.image_path, pred_class, probability)
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.image_dir:
        # Batch inference
        results = batch_inference(model, args.image_dir, args.output_dir, device)
        
        # Print summary
        num_alzheimers = sum(1 for r in results.values() if r['class'] == 1)
        num_normal = sum(1 for r in results.values() if r['class'] == 0)
        
        print("\nSummary:")
        print(f"Total images: {len(results)}")
        print(f"Predicted as Alzheimer's: {num_alzheimers}")
        print(f"Predicted as Normal: {num_normal}")
    
    else:
        print("Error: Please provide either --image_path or --image_dir")
