import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Configuration
SOURCE_CSV = "data/train.csv"
TRAIN_SPLIT_CSV = "data/train_split.csv"
VAL_SPLIT_CSV = "data/val_split.csv"
VAL_SPLIT_SIZE = 0.20 # 20% for validation
RANDOM_STATE = 42    # For reproducible results

def split_data():
    print(f"Loading data from {SOURCE_CSV}...")
    try:
        df = pd.read_csv(SOURCE_CSV)
    except FileNotFoundError:
        print(f"Error: {SOURCE_CSV} not found.")
        print("Please make sure you have the original train.csv in the data/ directory.")
        return

    if 'diagnosis' not in df.columns:
        print("Error: 'diagnosis' column not found in CSV.")
        return

    print(f"Splitting data: {100-VAL_SPLIT_SIZE*100}% train, {VAL_SPLIT_SIZE*100}% validation.")
    
    # Use stratified split to maintain class distribution
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['diagnosis'] # <-- Ensures both sets have all classes
    )
    
    # Save the new CSVs
    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(VAL_SPLIT_CSV, index=False)
    
    print(f"Successfully created:")
    print(f"  - {TRAIN_SPLIT_CSV} ({len(train_df)} samples)")
    print(f"  - {VAL_SPLIT_CSV} ({len(val_df)} samples)")
    print("\nOriginal vs. New Class Distributions:")
    
    # Print distributions to confirm
    print("\n--- Original ---")
    print(df['diagnosis'].value_counts(normalize=True).sort_index())
    print("\n--- Train Split ---")
    print(train_df['diagnosis'].value_counts(normalize=True).sort_index())
    print("\n--- Validation Split ---")
    print(val_df['diagnosis'].value_counts(normalize=True).sort_index())

if __name__ == "__main__":
    split_data()