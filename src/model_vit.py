# model_vit.py
import torch.nn as nn
import timm

def get_vit(num_classes=2, img_size=224, pretrained=True):
    """
    Creates a Vision Transformer (ViT-B/16) model using timm,
    with a custom classification head for `num_classes`.
    """
    # This loads a ViT-Base model pretrained on ImageNet-21k/1k (depending on timm version)
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)

    # timm ViT models usually expose the classifier as `model.head`
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model
