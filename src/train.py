import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dataset import RetinopathyDataset, get_transforms, get_validation_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json


# compute class weights to handle imbalance
def calculate_class_weights(csv_file):
    df = pd.read_csv(csv_file)
    labels = df["diagnosis"].to_numpy()
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


# validation loop
@torch.no_grad()
def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, val_acc, val_f1


# training entry
def train_model(model_type):
    # choose device: Intel XPU, NVIDIA CUDA, Apple MPS, or CPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # default paths
    train_csv = "../data/train_split.csv"
    val_csv = "../data/val_split.csv"
    train_img_dir = "../data/train_images"
    val_img_dir = "../data/train_images"

    # output folders
    save_dir = "../outputs/checkpoints"
    log_dir = "../outputs/logs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # class weights
    class_weights = calculate_class_weights(train_csv).to(device)

    # datasets and loaders
    train_transforms = get_transforms()
    val_transforms = get_validation_transforms()

    train_dataset = RetinopathyDataset(train_csv, train_img_dir, transform=train_transforms)
    val_dataset = RetinopathyDataset(val_csv, val_img_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # model selection
    if model_type == "resnet":
        # Use DRStageNet-pretrained ResNet50 (path must match your .pth file)
        model = get_resnet(num_classes=2)
    elif model_type == "efficientnet":
        model = get_efficientnet(num_classes=2)
    elif model_type == "vit":
        model = get_vit(num_classes=2, img_size=224)
    else:
        raise ValueError("Invalid model name")
    model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # lr scheduler
    num_epochs = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_f1 = 0.0
    history = []

    # training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # validation
        val_loss, val_acc, val_f1 = validate_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )

        # save best by macro f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(save_dir, f"{model_type}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model updated and saved to {save_path}")

        # step lr scheduler
        scheduler.step()

    # save logs
    log_path = os.path.join(log_dir, f"{model_type}_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training completed. Logs saved to {log_path}")


if __name__ == "__main__":
    model_name = input("Enter model you want to train (resnet / efficientnet / vit): ").strip().lower()
    train_model(model_name)
