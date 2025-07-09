import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import nibabel as nib

class AlzheimerDataset(Dataset):
    """Dataset class for Alzheimer's MRI scans"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # For this example, we'll assume we're working with 2D slices
        # In a real implementation, you might load 3D volumes with nibabel
        try:
            # Try loading as a 2D image first
            image = Image.open(img_path).convert('RGB')
        except:
            # If it's a NIfTI file, extract a middle slice
            try:
                nifti_img = nib.load(img_path)
                img_data = nifti_img.get_fdata()
                
                # Extract middle slice from sagittal view
                slice_idx = img_data.shape[0] // 2
                slice_data = img_data[slice_idx, :, :]
                
                # Normalize to 0-255 and convert to PIL
                slice_data = ((slice_data - slice_data.min()) / 
                             (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
                image = Image.fromarray(slice_data).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Return a blank image as fallback
                image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

def preprocess_mri_data(data_dir, val_size=0.2, batch_size=32, num_workers=4, excel_path=None):
    """
    Preprocess MRI data for Alzheimer's disease classification (flat folder structure)
    Args:
        data_dir: Directory containing class subfolders with images directly
        excel_path: Path to Excel file with patient metadata (optional)
        val_size: Proportion of training data to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
    Returns:
        train_loader, val_loader, test_loader: DataLoaders
        class_names: List of class names
        num_classes: Number of classes
    """
    print("Starting MRI data preprocessing (flat structure)...")
    
    # Get class names from folder names
    class_names = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    class_names.sort()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    num_classes = len(class_names)
    
    # Load metadata from Excel if provided
    metadata = {}
    if excel_path and os.path.exists(excel_path):
        try:
            import pandas as pd
            metadata_df = pd.read_excel(excel_path)
            # Create mapping from filename to metadata
            metadata = {}
            for _, row in metadata_df.iterrows():
                # Extract patient ID from filename pattern: OAS1_XXXX_MR1
                patient_id = f"OAS1_{str(row['ID']).zfill(4)}_MR1"
                metadata[patient_id] = row.to_dict()
            print(f"Loaded metadata for {len(metadata)} patients")
        except Exception as e:
            print(f"Error loading Excel metadata: {e}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load all image paths and labels
    all_paths = []
    all_labels = []
    all_metadata = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".nii", ".nii.gz")):
                file_path = os.path.join(class_dir, file_name)
                all_paths.append(file_path)
                all_labels.append(class_to_idx[class_name])
                
                # Extract patient ID from filename (OAS1_XXXX_MR1)
                patient_id = '_'.join(file_name.split('_')[:3])
                all_metadata.append(metadata.get(patient_id, {}))
    
    if not all_paths:
        print("No image files found in the directory! Please check your dataset.")
        return None, None, None, None, None
    
    # Split into train/val/test (80/10/10)
    train_indices, test_indices = train_test_split(
        np.arange(len(all_paths)), test_size=0.2, stratify=all_labels, random_state=42
    )
    val_indices, test_indices = train_test_split(
        test_indices, test_size=0.5, stratify=[all_labels[i] for i in test_indices], random_state=42
    )
    
    train_paths = [all_paths[i] for i in train_indices]
    val_paths = [all_paths[i] for i in val_indices]
    test_paths = [all_paths[i] for i in test_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    test_labels = [all_labels[i] for i in test_indices]
    
    print(f"Dataset statistics:")
    print(f"Train set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    print(f"Classes: {class_names}")
    print(f"Class distribution:")
    
    # Print class distribution
    for i, class_name in enumerate(class_names):
        train_count = sum(1 for label in train_labels if label == i)
        val_count = sum(1 for label in val_labels if label == i)
        test_count = sum(1 for label in test_labels if label == i)
        print(f"  {class_name}: Train={train_count}, Val={val_count}, Test={test_count}")
    
    # Create datasets with metadata
    train_dataset = AlzheimerDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = AlzheimerDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = AlzheimerDataset(test_paths, test_labels, transform=val_test_transform)
    
    # Add metadata to datasets
    train_dataset.metadata = [all_metadata[i] for i in train_indices]
    val_dataset.metadata = [all_metadata[i] for i in val_indices]
    test_dataset.metadata = [all_metadata[i] for i in test_indices]
    
    # Create data loaders optimized for GPU training
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print("Data preprocessing complete!")
    
    return train_loader, val_loader, test_loader, class_names, num_classes

def visualize_batch(dataloader, class_names, num_images=5):
    """Visualize a batch of images from the dataloader"""
    # Get a batch of training data
    images, labels = next(iter(dataloader))
    
    # Make a grid from batch
    images = images[:num_images]
    labels = labels[:num_images]
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        img = img.numpy().transpose((1, 2, 0))
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {class_names[label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage for flat directory structure
    train_loader, val_loader, test_loader, class_names, num_classes = preprocess_mri_data("./data")
    print("Data loaders created successfully")
    
    # Visualize a batch
    print("Visualizing a batch of training data:")
    visualize_batch(train_loader, class_names)
