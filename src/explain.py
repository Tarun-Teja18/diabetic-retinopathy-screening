import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import our models and validation transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from dataset import get_validation_transforms

def get_model_and_layers(model_type, checkpoint_path, device):
    """
    Loads a model, its checkpoint, and identifies the correct target 
    layers for Grad-CAM.
    """
    # 1. Load the model architecture
    if model_type == "resnet":
        model = get_resnet(num_classes=5, pretrained=False)
    elif model_type == "efficientnet":
        model = get_efficientnet(num_classes=5, pretrained=False)
    elif model_type == "vit":
        model = get_vit(num_classes=5, img_size=224)
    else:
        raise ValueError("Invalid model_type")
        
    # 2. Load the trained weights
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None, None, None
        
    model.to(device).eval()

    # 3. Define target layers for Grad-CAM (CRITICAL)
    if model_type == "resnet":
        target_layers = [model.layer4[-1]]
        reshape_transform = None
    elif model_type == "efficientnet":
        target_layers = [model.features[-1]]
        reshape_transform = None
    elif model_type == "vit":
        target_layers = [model.transformer.layers[-1][0].norm]
        
        def vit_reshape_transform(tensor):
            patches = tensor[:, 1:, :]
            height = width = int(patches.shape[1]**0.5)
            patches = patches.reshape(-1, height, width, patches.shape[-1])
            patches = patches.permute(0, 3, 1, 2)
            return patches
        
        reshape_transform = vit_reshape_transform

    return model, target_layers, reshape_transform

def generate_heatmap(args):
    """
    Main function to load model, image, and generate/save a heatmap.
    """
    # --- 1. Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model and Target Layers ---
    model, target_layers, reshape_transform = get_model_and_layers(
        args.model_type, args.checkpoint, device
    )
    if model is None:
        return

    # --- 3. Load and Preprocess Image ---
    try:
        original_img_pil = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        return
        
    preprocess = get_validation_transforms(img_size=224)
    input_tensor = preprocess(original_img_pil).unsqueeze(0).to(device)
    
    original_img_cv = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
    original_img_cv = cv2.resize(original_img_cv, (224, 224))
    rgb_img = np.float32(original_img_cv) / 255

    # --- 4. Run Grad-CAM ---
    #
    # --- THIS BLOCK IS NOW FIXED ---
    # The `use_cuda` argument is removed.
    #
    with GradCAM(model=model,
                 target_layers=target_layers,
                 reshape_transform=reshape_transform) as cam:
        
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        pred_class = model(input_tensor).argmax(1).item()
        title = f"Model: {args.model_type} | Pred: Class {pred_class}"
        visualization = cv2.putText(
            visualization, title, (5, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # --- 5. Save Result ---
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, visualization)
    print(f"Heatmap saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps.")
    
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=["resnet", "efficientnet", "vit"],
                        help="Model architecture.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--image_path', type=str, required=True,
                        help="Path to the input fundus image.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save the output heatmap image.")
    
    args = parser.parse_args()
    generate_heatmap(args)


'''import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import our models and validation transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from dataset import get_validation_transforms

def get_model_and_layers(model_type, checkpoint_path, device):
    """
    Loads a model, its checkpoint, and identifies the correct target 
    layers for Grad-CAM.
    """
    # 1. Load the model architecture
    if model_type == "resnet":
        model = get_resnet(num_classes=5, pretrained=False)
    elif model_type == "efficientnet":
        model = get_efficientnet(num_classes=5, pretrained=False)
    elif model_type == "vit":
        model = get_vit(num_classes=5, img_size=224)
    else:
        raise ValueError("Invalid model_type")
        
    # 2. Load the trained weights
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None, None, None
        
    model.to(device).eval()

    # 3. Define target layers for Grad-CAM (CRITICAL)
    # This is different for each architecture
    if model_type == "resnet":
        target_layers = [model.layer4[-1]]
        reshape_transform = None
    elif model_type == "efficientnet":
        target_layers = [model.features[-1]]
        reshape_transform = None
    elif model_type == "vit":
        # For ViT, we target the last block's norm layer
        target_layers = [model.transformer.blocks[-1].norm_1]
        
        # ViT's output is not a 2D map, so we must tell Grad-CAM how
        # to reshape the (Batch, 197, 768) tensor into a (Batch, C, H, W) map
        def vit_reshape_transform(tensor):
            # tensor shape: (batch_size, 197, 768)
            # 197 = 1 (CLS token) + 196 (14x14 patches)
            # We skip the CLS token [:, 1:, :]
            patches = tensor[:, 1:, :]
            
            # Reshape to (batch_size, 14, 14, 768)
            height = width = int(patches.shape[1]**0.5)
            patches = patches.reshape(-1, height, width, patches.shape[-1])
            
            # Permute to (batch_size, 768, 14, 14)
            patches = patches.permute(0, 3, 1, 2)
            return patches
        
        reshape_transform = vit_reshape_transform

    return model, target_layers, reshape_transform

def generate_heatmap(args):
    """
    Main function to load model, image, and generate/save a heatmap.
    """
    # --- 1. Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model and Target Layers ---
    model, target_layers, reshape_transform = get_model_and_layers(
        args.model_type, args.checkpoint, device
    )
    if model is None:
        return

    # --- 3. Load and Preprocess Image ---
    try:
        original_img_pil = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        return
        
    preprocess = get_validation_transforms(img_size=224)
    input_tensor = preprocess(original_img_pil).unsqueeze(0).to(device)
    
    # Convert PIL image to OpenCV format for visualization
    # (H, W, C) and BGR
    original_img_cv = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
    original_img_cv = cv2.resize(original_img_cv, (224, 224))
    # Normalize for show_cam_on_image
    rgb_img = np.float32(original_img_cv) / 255

    # --- 4. Run Grad-CAM ---
    # use_cuda=True works for MPS (Apple) as well as CUDA (Nvidia)
    with GradCAM(model=model,
                 target_layers=target_layers,
                 reshape_transform=reshape_transform,
                 use_cuda=torch.backends.mps.is_available() or torch.cuda.is_available()) as cam:
        
        # We don't specify targets, so it uses the top-predicted class
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        
        # Overlay heatmap on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Add a title to the image
        pred_class = model(input_tensor).argmax(1).item()
        title = f"Model: {args.model_type} | Pred: Class {pred_class}"
        visualization = cv2.putText(
            visualization, title, (5, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # --- 5. Save Result ---
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, visualization)
    print(f"Heatmap saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps.")
    
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=["resnet", "efficientnet", "vit"],
                        help="Model architecture.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--image_path', type=str, required=True,
                        help="Path to the input fundus image.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save the output heatmap image.")
    
    args = parser.parse_args()
    generate_heatmap(args)'''



'''import torch
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def explain(model, img_path):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).cuda()

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    rgb_img = input_tensor[0].permute(1,2,0).cpu().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization'''
