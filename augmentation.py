import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def get_augmentation_transforms():
    """
    Get data augmentation transforms for training
    
    Returns:
        transform: Composed transforms for training data augmentation
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform

def visualize_augmentations(image, num_augmentations=5):
    """
    Visualize augmentations applied to an image
    
    Args:
        image: PIL Image
        num_augmentations: Number of augmented images to generate
    """
    transform = get_augmentation_transforms()
    
    # Create a grid of augmented images
    plt.figure(figsize=(15, 3))
    
    # Original image
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')
    
    # Augmented images
    for i in range(num_augmentations):
        augmented = transform(image)
        
        # Convert tensor to numpy for visualization
        augmented = augmented.numpy().transpose((1, 2, 0))
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented = std * augmented + mean
        augmented = np.clip(augmented, 0, 1)
        
        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(augmented)
        plt.title(f"Aug {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_augmented_dataset(original_dataset, num_augmentations=3):
    """
    Create an augmented dataset by applying augmentations to the original dataset
    
    Args:
        original_dataset: Original dataset
        num_augmentations: Number of augmented samples to generate per original sample
    
    Returns:
        augmented_dataset: Dataset with augmented samples
    """
    # This is a conceptual function - in practice, you would implement this
    # based on your specific dataset class
    print(f"Creating augmented dataset with {num_augmentations} augmentations per sample")
    print("This would multiply your dataset size by", num_augmentations + 1)
    
    # In a real implementation, you would:
    # 1. Create a new dataset with original + augmented samples
    # 2. Apply different augmentations to each copy
    # 3. Return the combined dataset
    
    return "Augmented dataset would be created here"

if __name__ == "__main__":
    # Create a sample image for demonstration
    sample_image = Image.new('RGB', (224, 224), color='white')
    
    # Draw some shapes on the image to make it more interesting
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sample_image)
    draw.rectangle([(50, 50), (150, 150)], fill='black')
    draw.ellipse([(100, 100), (200, 200)], fill='gray')
    
    # Visualize augmentations
    print("Visualizing augmentations on a sample image:")
    visualize_augmentations(sample_image)
    
    # Demonstrate augmented dataset creation
    result = create_augmented_dataset("original_dataset", num_augmentations=3)
    print(result)
