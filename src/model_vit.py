# [Code based on sources: 138-152]
from vit_pytorch import ViT

def get_vit(num_classes=5, img_size=224):
    """
    Creates a Vision Transformer (ViT-B/16) model.
    """
    model = ViT(
        image_size=img_size, # <-- Use the argument here
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )
    return model


'''from vit_pytorch import ViT

def get_vit(num_classes=5):
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )
    return model'''