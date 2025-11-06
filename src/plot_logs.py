import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# Configuration
LOG_DIR = "outputs/logs"
PLOT_DIR = "outputs/plots"
MODEL_NAMES = ["resnet", "efficientnet", "vit"]

def plot_metrics():
    # Ensure plot directory exists
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Set a nice style
    sns.set_style("whitegrid")
    
    all_logs = []
    
    # --- 1. Load all log files ---
    for model_name in MODEL_NAMES:
        log_file = os.path.join(LOG_DIR, f"{model_name}_history.json")
        try:
            with open(log_file, 'r') as f:
                history = json.load(f)
                
                # Convert to DataFrame and add model name
                df = pd.DataFrame(history)
                df['model'] = model_name
                all_logs.append(df)
                
                print(f"Loaded log for {model_name}")
        except FileNotFoundError:
            print(f"Warning: Log file not found for {model_name}. Skipping.")
    
    if not all_logs:
        print("Error: No log files found. Please run training first.")
        return
        
    # Combine all logs into one big DataFrame
    full_df = pd.concat(all_logs)

    # --- 2. Create and Save Validation Loss Plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=full_df, x='epoch', y='val_loss', hue='model')
    plt.title('Validation Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend(title='Model')
    plt.xticks(range(1, full_df['epoch'].max() + 1)) # Ensure integer x-axis
    
    save_path = os.path.join(PLOT_DIR, "validation_loss_comparison.png")
    plt.savefig(save_path)
    print(f"Validation loss plot saved to {save_path}")

    # --- 3. Create and Save Validation F1 Plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=full_df, x='epoch', y='val_f1', hue='model')
    plt.title('Validation F1 (Macro) vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1 (Macro)')
    plt.legend(title='Model')
    plt.xticks(range(1, full_df['epoch'].max() + 1))
    
    save_path = os.path.join(PLOT_DIR, "validation_f1_comparison.png")
    plt.savefig(save_path)
    print(f"Validation F1 plot saved to {save_path}")

    # --- 4. Create and Save Training Loss Plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=full_df, x='epoch', y='train_loss', hue='model')
    plt.title('Training Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend(title='Model')
    plt.xticks(range(1, full_df['epoch'].max() + 1))
    
    save_path = os.path.join(PLOT_DIR, "training_loss_comparison.png")
    plt.savefig(save_path)
    print(f"Training loss plot saved to {save_path}")

if __name__ == "__main__":
    plot_metrics()