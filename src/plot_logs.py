import os
import json
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

from dataset import RetinopathyDataset, get_validation_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit

# Folders
LOG_DIR = "../outputs/logs"
PLOT_DIR = "../outputs/plots"
CHECKPOINT_DIR = "../outputs/checkpoints"

# Validation data (binary task)
VAL_CSV = "../data/val_split.csv"
VAL_IMG_DIR = "../data/train_images"

# Models you trained
MODEL_TYPES = ["resnet", "efficientnet", "vit"]


# ------------------ DEVICE --------------------------------- #

def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ------------------ HISTORY LOADING ------------------------ #

def load_history(model_type):
    """Load JSON training history for one model (if it exists)."""
    path = os.path.join(LOG_DIR, f"{model_type}_history.json")
    if not os.path.exists(path):
        print(f"[WARN] No history file found for {model_type}: {path}")
        return None

    with open(path, "r") as f:
        history = json.load(f)

    # Basic metrics (from train.py)
    hist = {
        "epochs": [h["epoch"] for h in history],
        "train_loss": [h["train_loss"] for h in history],
        "val_loss": [h["val_loss"] for h in history],
        "train_acc": [h["train_acc"] for h in history],
        "val_acc": [h["val_acc"] for h in history],
    }

    # Optional F1 and AUC (if you log them in train.py)
    if "val_f1" in history[0]:
        hist["val_f1"] = [h["val_f1"] for h in history]
    else:
        hist["val_f1"] = None
        print(f"[INFO] val_f1 not found in history for {model_type} (F1 curve may be empty).")

    if "val_auc" in history[0]:
        hist["val_auc"] = [h["val_auc"] for h in history]
    else:
        hist["val_auc"] = None
        print(f"[INFO] val_auc not found in history for {model_type} (AUC curve per epoch will be skipped).")

    return hist


# ------------------ PER-MODEL CURVES ----------------------- #

def plot_for_model(model_type, hist):
    """
    Make one figure for a single model and save it.

    - Subplot 1: Train / Val Loss
    - Subplot 2: Train / Val Accuracy
    - Subplot 3: Val F1 (and per-epoch AUC if available)
    """
    epochs = hist["epochs"]

    plt.figure(figsize=(15, 4))

    # ---- Left: Loss ----
    plt.subplot(1, 3, 1)
    plt.plot(epochs, hist["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, hist["val_loss"], marker="x", linestyle="--", label="Val Loss")
    plt.title(f"{model_type} - Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # ---- Middle: Accuracy ----
    plt.subplot(1, 3, 2)
    plt.plot(epochs, hist["train_acc"], marker="o", label="Train Acc")
    plt.plot(epochs, hist["val_acc"], marker="x", linestyle="--", label="Val Acc")
    plt.title(f"{model_type} - Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # ---- Right: F1 (per-epoch values) ----
    plt.subplot(1, 3, 3)
    have_any = False

    if hist.get("val_f1") is not None:
        plt.plot(epochs, hist["val_f1"], marker="o", label="Val F1")
        have_any = True

    if hist.get("val_auc") is not None:
        plt.plot(epochs, hist["val_auc"], marker="x", linestyle="--", label="Val AUC")
        have_any = True

    if have_any:
        plt.title(f"{model_type} - F1-score per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid(True)
        plt.legend()
    else:
        plt.title(f"{model_type} - No F1 logged")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid(True)

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_DIR, f"{model_type}_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"[OK] Saved curves for {model_type} to: {out_path}")


# ------------------ COMBINED F1 PLOT ----------------------- #

def plot_f1_score(hist_list, model_names):
    """One figure: validation F1 for all models on same plot."""
    plt.figure(figsize=(6, 4))

    for hist, name in zip(hist_list, model_names):
        if hist is None or hist.get("val_f1") is None:
            continue
        epochs = hist["epochs"]
        plt.plot(epochs, hist["val_f1"], marker="o", linestyle="--", label=name)

    plt.title("Validation F1-score for all models")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_DIR, "f1_curves_all_models.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"[OK] Saved combined F1 curves to: {out_path}")


# ------------------ ROC + AUC PER MODEL -------------------- #

def compute_roc_for_model(model_type):
    """
    Compute ROC curve (FPR, TPR) and AUC on the validation set
    using the best checkpoint for the given model.
    Assumes binary classification (classes 0 and 1).
    """
    device = get_device()
    print(f"[ROC] Using device: {device} for {model_type}")

    # 1) Validation dataset & loader
    val_transforms = get_validation_transforms()
    dataset = RetinopathyDataset(VAL_CSV, VAL_IMG_DIR, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 2) Load model architecture (num_classes=2 for binary)
    if model_type == "resnet":
        model = get_resnet(num_classes=2)
    elif model_type == "efficientnet":
        model = get_efficientnet(num_classes=2)
    elif model_type == "vit":
        model = get_vit(num_classes=2, img_size=224)
    else:
        print(f"[ROC] Unknown model_type: {model_type}")
        return None, None, None

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ROC] Checkpoint not found for {model_type}: {ckpt_path}")
        return None, None, None

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 3) Collect scores and true labels
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)                # shape [N, 2]
            probs = torch.softmax(outputs, dim=1)  # shape [N, 2]
            pos_scores = probs[:, 1]              # probability for class "1"

            all_scores.extend(pos_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 4) Compute ROC and AUC
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc_val = roc_auc_score(all_labels, all_scores)
        return fpr, tpr, auc_val
    except ValueError as e:
        print(f"[ROC] Could not compute ROC for {model_type}: {e}")
        return None, None, None


def plot_roc_for_model(model_type, fpr, tpr, auc_val):
    """Save ROC curve for a single model."""
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")

    plt.title(f"ROC Curve - {model_type}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_DIR, f"roc_{model_type}.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"[OK] Saved ROC curve for {model_type} to: {out_path}")


def plot_roc_all_models(roc_data):
    """Combined ROC curve for all models in one figure."""
    plt.figure(figsize=(6, 5))

    for model_type, (fpr, tpr, auc_val) in roc_data.items():
        if fpr is None:
            continue
        plt.plot(fpr, tpr, linestyle="--", label=f"{model_type} (AUC={auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")

    plt.title("ROC Curves - All Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_DIR, "roc_all_models.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"[OK] Saved combined ROC curves to: {out_path}")


# ------------------ MAIN ------------------------------------ #

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    hist_list = []
    names_for_f1 = []

    # 1) Per-model training curves
    for model_type in MODEL_TYPES:
        hist = load_history(model_type)
        if hist is None:
            continue
        hist_list.append(hist)
        names_for_f1.append(model_type)
        plot_for_model(model_type, hist)

    # 2) Combined F1 curve for all models
    if hist_list:
        plot_f1_score(hist_list, names_for_f1)

    # 3) ROC & AUC per model + combined plot
    roc_results = {}
    for model_type in MODEL_TYPES:
        fpr, tpr, auc_val = compute_roc_for_model(model_type)
        roc_results[model_type] = (fpr, tpr, auc_val)
        if fpr is not None:
            plot_roc_for_model(model_type, fpr, tpr, auc_val)

    # 4) Combined ROC curve
    plot_roc_all_models(roc_results)

    print("\nDone. Check the plots folder:", PLOT_DIR)


if __name__ == "__main__":
    main()
