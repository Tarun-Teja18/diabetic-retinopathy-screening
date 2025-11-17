import os
import cv2
import numpy as np
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import get_validation_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit

# Fixed dataset paths
TEST_CSV = "../data/test_split.csv"
IMG_DIR = "../data/train_images"
OUTPUT_DIR = "../outputs/gradcam"
IMG_SIZE = 224  # must match training size


def get_device():
    """Select device: XPU, CUDA, MPS, or CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def get_model_and_layers(model_type, checkpoint_path, device):
    """
    Load the chosen model (binary num_classes=2), its checkpoint,
    and choose the correct target layer(s) for Grad-CAM.
    """
    # 1) Build model (same architecture as in train.py: num_classes=2)
    if model_type == "resnet":
        model = get_resnet(num_classes=2, pretrained=False)
    elif model_type == "efficientnet":
        model = get_efficientnet(num_classes=2, pretrained=False)
    elif model_type == "vit":
        # timm-based ViT, we can set pretrained=False since we load our own weights
        model = get_vit(num_classes=2, img_size=224, pretrained=False)
    else:
        raise ValueError("Invalid model_type (use: resnet / efficientnet / vit)")

    # 2) Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at: {checkpoint_path}")
        return None, None, None

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # 3) Target layers and optional reshape for ViT
    reshape_transform = None

    if model_type == "resnet":
        # last conv block
        target_layers = [model.layer4[-1]]

    elif model_type == "efficientnet":
        # last features block
        target_layers = [model.features[-1]]

    elif model_type == "vit":
        # for timm vit_base_patch16_224
        # Use last transformer block
        target_layers = [model.blocks[-1].norm1]

        # ViT outputs [B, tokens, C] -> reshape to feature map
        def vit_reshape_transform(tensor):
            # tensor: [B, tokens, C], first token is CLS, ignore it
            b, n, c = tensor.shape
            h = w = int((n - 1) ** 0.5)  # remaining tokens are (h*w)
            tensor = tensor[:, 1:, :]    # drop CLS token
            tensor = tensor.reshape(b, h, w, c)
            tensor = tensor.permute(0, 3, 1, 2)  # [B, C, H, W]
            return tensor

        reshape_transform = vit_reshape_transform

    return model, target_layers, reshape_transform


def pick_sample_from_val():
    """
    Let the user:
      1) Enter an image ID directly (without .png), OR
      2) If left blank, fall back to choosing by row index from val_split.csv.

    Returns (img_id, label).
    """
    if not os.path.exists(TEST_CSV):
        print(f"[Error] Validation CSV not found at: {TEST_CSV}")
        return None, None

    df = pd.read_csv(TEST_CSV)

    # Assume first column = image id, 'diagnosis' = label
    id_col = df.columns[0]
    if "diagnosis" not in df.columns:
        print("[Error] 'diagnosis' column not found in val_split.csv")
        return None, None

    print(f"Loaded validation CSV: {TEST_CSV}")
    print(f"Total validation samples: {len(df)}")

    # === NEW: Ask for image ID first ===
    img_id_in = input(
        "\nEnter image ID (without .png) to visualize "
        "or press Enter to choose by row index: "
    ).strip()

    if img_id_in != "":
        # Try to find this image ID in the CSV
        matches = df[df[id_col].astype(str) == img_id_in]
        if len(matches) == 0:
            print(f"[Error] Image ID '{img_id_in}' not found in {TEST_CSV}.")
            print("Falling back to index selection...")
        else:
            row = matches.iloc[0]
            img_id = str(row[id_col])
            label = int(row["diagnosis"])
            print(f"\nSelected sample by image ID:")
            print(f"  Image ID: {img_id}")
            print(f"  Label:    {label}")
            return img_id, label

    # === Old behavior: show first rows and ask for index ===
    print("\nFirst 5 validation samples:")
    print(df[[id_col, "diagnosis"]].head())

    idx_in = input("\nEnter row index from validation CSV to visualize (default: 0): ").strip()
    if idx_in == "":
        idx = 0
    else:
        try:
            idx = int(idx_in)
        except ValueError:
            print("[Error] Invalid index. Using 0.")
            idx = 0

    if not (0 <= idx < len(df)):
        print(f"[Error] Index out of range (0 to {len(df)-1}). Using 0.")
        idx = 0

    row = df.iloc[idx]
    img_id = str(row[id_col])
    label = int(row["diagnosis"])

    print(f"\nSelected sample by index:")
    print(f"  Index:    {idx}")
    print(f"  Image ID: {img_id}")
    print(f"  Label:    {label}")

    return img_id, label


def run_grad_cam():
    print("=== Grad-CAM Visualization ===")

    # 1) Choose model type
    model_type = input("Enter model type (resnet / efficientnet / vit): ").strip().lower()
    if model_type not in ["resnet", "efficientnet", "vit"]:
        print("[Error] Invalid model type. Use: resnet / efficientnet / vit")
        return

    # 2) Choose checkpoint (dynamic but with a default)
    default_ckpt = f"../outputs/checkpoints/{model_type}_best.pth"
    ckpt_in = input(f"Path to checkpoint [.pth] [default: {default_ckpt}]: ").strip()
    checkpoint_path = ckpt_in if ckpt_in != "" else default_ckpt

    # 3) Device
    device = get_device()

    # 4) Load model + target layers
    model, target_layers, reshape_transform = get_model_and_layers(
        model_type, checkpoint_path, device
    )
    if model is None:
        return

    # 5) Choose a validation sample (by image ID or index)
    img_id, true_label = pick_sample_from_val()
    if img_id is None:
        return

    # 6) Load image from train_images
    img_path = os.path.join(IMG_DIR, img_id + ".png")
    if not os.path.exists(img_path):
        print(f"[Error] Image file not found at: {img_path}")
        return

    try:
        original_img_pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[Error] Could not open image: {e}")
        return

    # 7) Preprocess (same as validation transforms)
    preprocess = get_validation_transforms(img_size=IMG_SIZE)
    input_tensor = preprocess(original_img_pil).unsqueeze(0).to(device)

    # For visualization: BGR -> float RGB [0,1]
    original_img_cv = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
    original_img_cv = cv2.resize(original_img_cv, (IMG_SIZE, IMG_SIZE))
    rgb_img = np.float32(original_img_cv) / 255.0

    # 8) Run Grad-CAM
    print("\nComputing Grad-CAM heatmap...")
    with GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    ) as cam:

        outputs = model(input_tensor)
        pred_class = int(outputs.argmax(1).item())

        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        text = f"{model_type.upper()} | True: {true_label} Pred: {pred_class}"
        cv2.putText(
            visualization,
            text,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    # 9) Save heatmap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{img_id}_{model_type}_gradcam.png")
    cv2.imwrite(out_path, visualization)
    print(f"\nGrad-CAM heatmap saved to: {out_path}")


if __name__ == "__main__":
    run_grad_cam()
