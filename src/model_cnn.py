import torch.nn as nn
import torchvision.models as models


def get_resnet(num_classes: int = 5, pretrained: bool = True):
    """
    Returns a ResNet-50 model pretrained on ImageNet, with the final
    fully connected layer replaced to predict `num_classes` outputs.
    """
    # Choose weights if pretrained=True, else no weights
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None

    # Load backbone
    model = models.resnet50(weights=weights)

    # Replace classification head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_efficientnet(num_classes: int = 5, pretrained: bool = True):
    """
    Returns an EfficientNet-B0 model pretrained on ImageNet,
    with the final classifier layer replaced to predict `num_classes`.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None

    model = models.efficientnet_b0(weights=weights)

    # EfficientNet-B0 classifier is usually: Dropout -> Linear
    num_ftrs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, num_classes),
    )

    return model
