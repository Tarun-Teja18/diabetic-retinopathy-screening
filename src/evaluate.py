# [Code based on sources: 86, 271, 376-382]
import torch
from torch.utils.data import DataLoader
from dataset import RetinopathyDataset, get_validation_transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def evaluate_model(args):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading validation dataset from {args.csv_file}...")
    val_transforms = get_validation_transforms(img_size=args.img_size)
    # Pass the correct img_dir
    dataset = RetinopathyDataset(args.csv_file, args.img_dir, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Loading model: {args.model_type}")
    if args.model_type == "resnet":
        model = get_resnet(num_classes=5, pretrained=False)
    elif args.model_type == "efficientnet":
        model = get_efficientnet(num_classes=5, pretrained=False)
    elif args.model_type == "vit":
        model = get_vit(num_classes=5, img_size=args.img_size)
    else:
        raise ValueError("Invalid model type specified")
        
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
        
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())

    # --- Calculate Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    # Binarize labels for AUC
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])
    all_probs = np.array(all_probs)
    
    try:
        auc = roc_auc_score(all_labels_bin, all_probs, average="macro", multi_class="ovr")
    except ValueError as e:
        print(f"Warning: Could not compute AUC. Error: {e}")
        auc = 0.0 # Set AUC to 0 if calculation fails

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n--- Evaluation Results for {args.model_type} ---")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"F1 (Macro): {f1:.4f}")
    print(f"AUC (Macro): {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    print("\n(Rows = True Labels, Columns = Predicted Labels)")
    print("-" * (30 + len(args.model_type)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diabetic Retinopathy Model")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint (.pth) file")
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=["resnet", "efficientnet", "vit"], 
                        help="Model architecture that was trained")
    
    # --- UPDATED DEFAULTS ---
    parser.add_argument('--csv_file', type=str, default="data/val_split.csv", 
                        help="Path to validation CSV file")
    parser.add_argument('--img_dir', type=str, default="data/train_images", 
                        help="Path to validation image directory")
    # ------------------------
    
    parser.add_argument('--img_size', type=int, default=224, 
                        help="Image size used during training")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Evaluation batch size")
    
    args = parser.parse_args()
    
    evaluate_model(args)


'''# [Code based on sources: 86, 271, 376-382]
import torch
from torch.utils.data import DataLoader
from dataset import RetinopathyDataset, get_validation_transforms # <-- Import validation transforms
from model_cnn import get_resnet, get_efficientnet
from model_vit import get_vit
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def evaluate_model(args):
    
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Dataset and Dataloader ---
    print("Loading test dataset...")
    # Use the validation transforms (no augmentation)
    val_transforms = get_validation_transforms(img_size=args.img_size)
    dataset = RetinopathyDataset(args.csv_file, args.img_dir, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # --- Model ---
    print(f"Loading model: {args.model_type}")
    if args.model_type == "resnet":
        model = get_resnet(num_classes=5, pretrained=False) # No need for pretrained weights
    elif args.model_type == "efficientnet":
        model = get_efficientnet(num_classes=5, pretrained=False)
    elif args.model_type == "vit":
        model = get_vit(num_classes=5, img_size=args.img_size)
    else:
        raise ValueError("Invalid model type specified")
        
    # --- Load Checkpoint ---
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
        
    model.to(device)
    model.eval() # <-- Set model to evaluation mode

    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []
    all_probs = [] # For AUC

    with torch.no_grad(): # Disable gradient calculation
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # Get probabilities (for AUC)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Get predictions (for F1, Accuracy)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())

    # --- Calculate Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    # Binarize labels for AUC calculation (One-vs-Rest)
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])
    all_probs = np.array(all_probs)
    
    # Handle cases where not all classes are present in the test set
    if all_labels_bin.shape[1] == 1:
        # Binary case (unlikely, but safe)
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        # Multi-class One-vs-Rest
        auc = roc_auc_score(all_labels_bin, all_probs, average="macro", multi_class="ovr")
    
    cm = confusion_matrix(all_labels, all_preds)

    # --- Print Results ---
    print(f"\n--- Evaluation Results for {args.model_type} ---")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"F1 (Macro): {f1:.4f}")
    print(f"AUC (Macro): {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    print("\n(Rows = True Labels, Columns = Predicted Labels)")
    print("-" * (30 + len(args.model_type)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diabetic Retinopathy Model")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint (.pth) file")
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=["resnet", "efficientnet", "vit"], 
                        help="Model architecture that was trained")
    
    parser.add_argument('--csv_file', type=str, default="data/test.csv", 
                        help="Path to test CSV file")
    parser.add_argument('--img_dir', type=str, default="data/test_images", 
                        help="Path to test image directory")
    
    parser.add_argument('--img_size', type=int, default=224, 
                        help="Image size used during training")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Evaluation batch size")
    
    args = parser.parse_args()
    
    evaluate_model(args)'''