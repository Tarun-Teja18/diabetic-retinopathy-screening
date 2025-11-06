# [Code based on sources: 131-137, 269]
import torch.nn as nn
import torchvision.models as models

def get_resnet(num_classes=5, pretrained=True):
    """
    Loads a pretrained ResNet50 model and replaces the 
    final fully connected layer.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Get in_features for the new fc layer
    num_ftrs = model.fc.in_features
    
    # Replace the final layer
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_efficientnet(num_classes=5, pretrained=True):
    """
    Loads a pretrained EfficientNet-B0 model and replaces the
    final classifier layer.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Get in_features for the new classifier
    num_ftrs = model.classifier[1].in_features
    
    # Replace the final layer
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, num_classes),
    )
    
    return model


'''import torch.nn as nn
import torchvision.models as models

def get_resnet(num_classes=5):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model'''
