# evaluate.py
# Evaluate a trained Diabetic Retinopathy model (binary: 0 vs 1)

import os
import json
import torch
from torch.utils.data import DataLoader
from dataset import RetinopathyDataset, get_validation_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(model_type,
                   checkpoint_path,
                   csv_file="../data/test_split.csv",
                   img_dir="../data/train_images",
                   img_size=224,
                   batch_size=32):
    # ----- Device selection (same style as train.py) -----
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- Dataset and DataLoader -----
    print(f"\nLoading validation dataset from: {csv_file}")
    val_transforms = get_validation_transforms(img_size=img_size)
    dataset = RetinopathyDataset(csv_file, img_dir, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # ----- Model selection (binary: num_classes = 2) -----
    print(f"\nLoading model: {model_type}")
    if model_type == "resnet":
        model = get_resnet(num_classes=2)
    elif model_type == "efficientnet":
        model = get_efficientnet(num_classes=2)
    elif model_type == "vit":
        model = get_vit(num_classes=2, img_size=img_size)
    else:
        raise ValueError("Invalid model type (use: resnet / efficientnet / vit)")

    # ----- Load checkpoint -----
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found at: {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    # ----- Evaluation loop -----
    all_preds = []
    all_labels = []
    all_probs = []

    print("\nEvaluating on validation set...\n")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of class 1
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ----- Metrics -----
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        print(f"Warning: Could not compute AUC: {e}")
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n--- Evaluation Results ({model_type}) ---")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"CSV file:     {csv_file}")
    print(f"Image folder: {img_dir}\n")

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"F1 (binary):  {f1:.4f}")
    print(f"AUC (ROC):    {auc:.4f}")

    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(cm)
    print("\nDone.\n")


if __name__ == "__main__":
    # ----- Simple console inputs with defaults -----
    model_type = input(
        "Enter model type to evaluate (resnet / efficientnet / vit): "
    ).strip().lower()

    # Default checkpoint name based on model_type
    default_ckpt = f"../outputs/checkpoints/{model_type}_best.pth"
    ckpt_in = input(
        f"Path to checkpoint [.pth] "
        f"[default: {default_ckpt}]: "
    ).strip()
    checkpoint_path = ckpt_in if ckpt_in else default_ckpt

    csv_in = input(
        "Path to Testing CSV "
        "[default: ../data/test_split.csv]: "
    ).strip()
    csv_file = csv_in if csv_in else "../data/val_split.csv"

    img_in = input(
        "Path to Testing images folder "
        "[default: ../data/train_images]: "
    ).strip()
    img_dir = img_in if img_in else "../data/train_images"

    evaluate_model(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        csv_file=csv_file,
        img_dir=img_dir,
        img_size=224,
        batch_size=32,
    )
