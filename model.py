import torch
import torch.nn as nn
import torch.nn.functional as F

class AlzheimerCNN(nn.Module):
    """
    Convolutional Neural Network for Alzheimer's disease classification
    """
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # Input size calculation: 224/2/2/2/2 = 14
        # So after 4 pooling layers, we have 256 channels of 14x14 feature maps
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224x224 -> 112x112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112x112 -> 56x56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56x56 -> 28x28
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 28x28 -> 14x14
        
        # Flatten
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AlzheimerResNet(nn.Module):
    """
    Residual Network for Alzheimer's disease classification
    """
    def __init__(self, num_classes=4):
        super(AlzheimerResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Average pooling and classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def get_model(model_name="cnn", num_classes=4, pretrained=False):
    """
    Factory function to get the specified model
    
    Args:
        model_name: Name of the model to use ('cnn' or 'resnet')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: PyTorch model
    """
    if model_name == "cnn":
        model = AlzheimerCNN(num_classes=num_classes)
    elif model_name == "resnet":
        model = AlzheimerResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if pretrained:
        # In a real implementation, you would load pretrained weights here
        print("Pretrained weights would be loaded here")
    
    return model

if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test CNN model
    cnn_model = get_model("cnn")
    cnn_model.to(device)
    print(cnn_model)
    
    # Test with a random input
    x = torch.randn(1, 3, 224, 224).to(device)
    output = cnn_model(x)
    print(f"Output shape: {output.shape}")
    
    # Test ResNet model
    resnet_model = get_model("resnet")
    resnet_model.to(device)
    print(resnet_model)
    
    # Test with a random input
    output = resnet_model(x)
    print(f"Output shape: {output.shape}")
