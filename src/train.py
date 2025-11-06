import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import RetinopathyDataset, get_transforms, get_validation_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json # <-- For logging

# --- Helper function to calculate class weights ---
def calculate_weights(csv_file):
    print(f"Calculating class weights from: {csv_file}")
    df = pd.read_csv(csv_file)
    classes = np.unique(df['diagnosis'])
    labels = df['diagnosis'].to_numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    print(f"Calculated weights for classes {classes}: {weights}")
    return torch.tensor(weights, dtype=torch.float)

# --- NEW: Validation loop function ---
def validate_model(model, dataloader, criterion, device):
    """
    Runs a validation loop.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    return avg_loss, f1

# --- Main Training Function (Heavily Upgraded) ---
def train_model(args):
    
    # --- 1. Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # --- 2. Data ---
    class_weights = calculate_weights(args.csv_file).to(device)
    
    # Training data
    train_transforms = get_transforms(img_size=args.img_size)
    train_dataset = RetinopathyDataset(args.csv_file, args.img_dir, transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Validation data
    val_transforms = get_validation_transforms(img_size=args.img_size)
    val_dataset = RetinopathyDataset(args.val_csv, args.val_img_dir, transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 3. Model ---
    print(f"Loading model: {args.model_type}")
    if args.model_type == "resnet":
        model = get_resnet(num_classes=5)
    elif args.model_type == "efficientnet":
        model = get_efficientnet(num_classes=5)
    elif args.model_type == "vit":
        model = get_vit(num_classes=5, img_size=args.img_size)
    else:
        raise ValueError("Invalid model type specified")
    model.to(device)

    # --- 4. Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 5. Training & Validation Loop ---
    best_val_f1 = 0.0
    history = [] # To store logs

    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train() # Set model to training mode
        total_train_loss = 0
        
        # Training loop
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation loop
        avg_val_loss, val_f1 = validate_model(model, val_dataloader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1 (Macro): {val_f1:.4f}")

        # Log results
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_f1': val_f1
        })

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(args.save_dir, f"{args.model_type}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved (Val F1: {val_f1:.4f}) to {save_path}")

    # --- 6. Save Logs ---
    log_path = os.path.join(args.log_dir, f"{args.model_type}_history.json")
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {log_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Model")
    
    # Data paths
    parser.add_argument('--csv_file', type=str, default="data/train_split.csv", help="Path to training CSV file")
    parser.add_argument('--img_dir', type=str, default="data/train_images", help="Path to training image directory")
    parser.add_argument('--val_csv', type=str, default="data/val_split.csv", help="Path to validation CSV file")
    
    # --- THIS LINE IS NOW FIXED ---
    parser.add_argument('--val_img_dir', type=str, default="data/train_images", help="Path to validation image directory")
    
    # Output paths
    parser.add_argument('--save_dir', type=str, default="outputs/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument('--log_dir', type=str, default="outputs/logs", help="Directory to save training logs")
    
    # Model params
    parser.add_argument('--model_type', type=str, required=True, choices=["resnet", "efficientnet", "vit"], help="Model architecture to train")
    parser.add_argument('--img_size', type=int, default=224, help="Image size for training")
    
    # Training params
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(args)



'''# [Code based on sources: 153-194, 257-258, 371-375]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RetinopathyDataset, get_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_weights(csv_file):
    print(f"Calculating class weights from: {csv_file}")
    df = pd.read_csv(csv_file)
    classes = np.unique(df['diagnosis'])
    labels = df['diagnosis'].to_numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    print(f"Calculated weights for classes {classes}: {weights}")
    return torch.tensor(weights, dtype=torch.float)

def train_model(args):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    class_weights = calculate_weights(args.csv_file).to(device)

    print("Loading dataset...")
    transforms = get_transforms(img_size=args.img_size)
    dataset = RetinopathyDataset(args.csv_file, args.img_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4) 
    
    print(f"Loading model: {args.model_type}")
    if args.model_type == "resnet":
        model = get_resnet(num_classes=5)
    elif args.model_type == "efficientnet":
        model = get_efficientnet(num_classes=5)
    elif args.model_type == "vit":
        model = get_vit(num_classes=5, img_size=args.img_size)
    else:
        raise ValueError("Invalid model type specified")
        
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using weighted CrossEntropyLoss.")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss, total_preds, total_labels = 0, [], []
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(total_labels, total_preds, average="macro")
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Loss: {avg_loss:.4f} | F1 (Macro): {f1:.4f}")

    save_path = os.path.join(args.save_dir, f"{args.model_type}_weighted_epoch{args.num_epochs}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Model")
    
    # --- UPDATED DEFAULTS ---
    parser.add_argument('--csv_file', type=str, default="data/train_split.csv", 
                        help="Path to training CSV file")
    parser.add_argument('--img_dir', type=str, default="data/train_images", 
                        help="Path to training image directory")
    # ------------------------

    parser.add_argument('--save_dir', type=str, default="outputs/checkpoints", 
                        help="Directory to save model checkpoints")
    parser.add_argument('--model_type', type=str, default="resnet", 
                        choices=["resnet", "efficientnet", "vit"], 
                        help="Model architecture to train")
    parser.add_argument('--img_size', type=int, default=224, 
                        help="Image size for training")
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(args)'''



'''# [Code based on sources: 153-194, 257-258, 371-375]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RetinopathyDataset, get_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_weights(csv_file):
    print("Calculating class weights...")
    df = pd.read_csv(csv_file)
    classes = np.unique(df['diagnosis'])
    labels = df['diagnosis'].to_numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    print(f"Calculated weights for classes {classes}: {weights}")
    return torch.tensor(weights, dtype=torch.float)

def train_model(args):
    
    # --- UPDATED: Device Logic for Apple Silicon (M3) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # Use Apple's Metal Performance Shaders
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    # ---------------------------------------------------

    # Calculate weights
    class_weights = calculate_weights(args.csv_file).to(device)

    # Dataset and Dataloader
    print("Loading dataset...")
    transforms = get_transforms(img_size=args.img_size)
    dataset = RetinopathyDataset(args.csv_file, args.img_dir, transform=transforms)
    # Increase num_workers for faster data loading
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4) 
    
    # Model
    print(f"Loading model: {args.model_type}")
    if args.model_type == "resnet":
        model = get_resnet(num_classes=5)
    elif args.model_type == "efficientnet":
        model = get_efficientnet(num_classes=5)
    elif args.model_type == "vit":
        model = get_vit(num_classes=5, img_size=args.img_size)
    else:
        raise ValueError("Invalid model type specified")
        
    model.to(device) # <-- This sends the model to the GPU

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using weighted CrossEntropyLoss.")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss, total_preds, total_labels = 0, [], []
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            # <-- This sends your data to the GPU
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(total_labels, total_preds, average="macro")
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Loss: {avg_loss:.4f} | F1 (Macro): {f1:.4f}")

    # Save the model
    save_path = os.path.join(args.save_dir, f"{args.model_type}_weighted_epoch{args.num_epochs}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Model")
    
    parser.add_argument('--csv_file', type=str, default="data/train.csv", 
                        help="Path to training CSV file")
    parser.add_argument('--img_dir', type=str, default="data/train_images", 
                        help="Path to training image directory")
    parser.add_argument('--save_dir', type=str, default="outputs/checkpoints", 
                        help="Directory to save model checkpoints")
    
    parser.add_argument('--model_type', type=str, default="resnet", 
                        choices=["resnet", "efficientnet", "vit"], 
                        help="Model architecture to train")
    
    parser.add_argument('--img_size', type=int, default=224, 
                        help="Image size for training")
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(args)'''



'''# [Code based on sources: 153-194, 257-258, 371-375]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RetinopathyDataset, get_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import argparse
import pandas as pd # <-- NEW: For reading CSV to calculate weights
import numpy as np  # <-- NEW: For calculating weights
from sklearn.utils.class_weight import compute_class_weight # <-- NEW: The main tool

# --- NEW: Function to calculate class weights ---
def calculate_weights(csv_file):
    """
    Calculates class weights for a severely imbalanced dataset.
    """
    print("Calculating class weights...")
    df = pd.read_csv(csv_file)
    
    # Get all unique classes (0, 1, 2, 3, 4)
    classes = np.unique(df['diagnosis'])
    
    # Get the labels from the dataframe
    labels = df['diagnosis'].to_numpy()
    
    # Calculate weights using 'balanced' mode
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    print(f"Calculated weights for classes {classes}: {weights}")
    return torch.tensor(weights, dtype=torch.float)
# ------------------------------------------------

def train_model(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- NEW: Calculate weights before doing anything else ---
    class_weights = calculate_weights(args.csv_file).to(device)
    # --------------------------------------------------------

    # Dataset and Dataloader
    print("Loading dataset...")
    transforms = get_transforms(img_size=args.img_size)
    dataset = RetinopathyDataset(args.csv_file, args.img_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    print(f"Loading model: {args.model_type}")
    if args.model_type == "resnet":
        model = get_resnet(num_classes=5)
    elif args.model_type == "efficientnet":
        model = get_efficientnet(num_classes=5)
    elif args.model_type == "vit":
        model = get_vit(num_classes=5, img_size=args.img_size)
    else:
        raise ValueError("Invalid model type specified")
        
    model.to(device)

    # --- UPDATED: Pass the weights to the loss function ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using weighted CrossEntropyLoss.")
    # -----------------------------------------------------
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss, total_preds, total_labels = 0, [], []
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(total_labels, total_preds, average="macro")
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Loss: {avg_loss:.4f} | F1 (Macro): {f1:.4f}")

    # Save the model
    save_path = os.path.join(args.save_dir, f"{args.model_type}_weighted_epoch{args.num_epochs}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Model")
    
    parser.add_argument('--csv_file', type=str, default="data/train.csv", 
                        help="Path to training CSV file")
    parser.add_argument('--img_dir', type=str, default="data/train_images", 
                        help="Path to training image directory")
    parser.add_argument('--save_dir', type=str, default="outputs/checkpoints", 
                        help="Directory to save model checkpoints")
    
    parser.add_argument('--model_type', type=str, default="resnet", 
                        choices=["resnet", "efficientnet", "vit"], 
                        help="Model architecture to train")
    
    parser.add_argument('--img_size', type=int, default=224, 
                        help="Image size for training")
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(args)'''


'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RetinopathyDataset, get_transforms
from model_cnn import get_resnet
from sklearn.metrics import f1_score

def train_model(csv_file, img_dir, num_epochs=5, batch_size=32, lr=1e-4, model_type="resnet"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = RetinopathyDataset(csv_file, img_dir, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    if model_type == "resnet":
        model = get_resnet()
    else:
        from model_vit import get_vit
        model = get_vit()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_preds, total_labels = 0, [], []

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

        f1 = f1_score(total_labels, total_preds, average="macro")
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f} F1: {f1:.4f}")

    return model

if __name__ == "__main__":
    model = train_model("data/train.csv", "data/train_images", num_epochs=10, model_type="resnet")'''
