import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
SOURCE_CSV      = "../data/train.csv"
TRAIN_SPLIT_CSV = "../data/train_split.csv"
VAL_SPLIT_CSV   = "../data/val_split.csv"
TEST_SPLIT_CSV  = "../data/test_split.csv"

# We want 488 images per binary class (0 and 1) -> 500 total
PER_CLASS    = 488
RANDOM_STATE = 42

# Desired final ratios
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20   # (VAL + TEST = 0.40)


def describe_counts(df, title):
    print(f"\n{title}")
    print("Total samples:", len(df))
    print(df["diagnosis"].value_counts().sort_index())
    print(df["diagnosis"].value_counts(normalize=True).sort_index())


def main():
    print(f"Loading data from {SOURCE_CSV}...")
    try:
        df = pd.read_csv(SOURCE_CSV)
    except FileNotFoundError:
        print(f"Error: {SOURCE_CSV} not found.")
        print("Please make sure train.csv is in the ../data/ directory.")
        return

    if "diagnosis" not in df.columns:
        print("Error: 'diagnosis' column not found in CSV.")
        return

    # 1) Keep only classes 0, 3, 4
    df = df[df["diagnosis"].isin([0, 3, 4])].copy()
    describe_counts(df, "Original filtered distribution (classes 0,3,4)")

    # 2) Map to binary labels:
    #    0 -> 0
    #    3,4 -> 1
    df["diagnosis_bin"] = df["diagnosis"].map({0: 0, 3: 1, 4: 1})

    # Overwrite 'diagnosis' with binary labels for the rest of the pipeline
    df["diagnosis"] = df["diagnosis_bin"]
    df = df.drop(columns=["diagnosis_bin"])

    describe_counts(df, "After mapping to binary labels (0 vs 1)")

    # 3) Balance to PER_CLASS per class (250 each => 500 total)
    balanced_parts = []
    for cls in [0, 1]:
        cls_df = df[df["diagnosis"] == cls]
        if len(cls_df) < PER_CLASS:
            raise ValueError(
                f"Not enough samples for class {cls}. "
                f"Have {len(cls_df)}, need {PER_CLASS}."
            )
        balanced_parts.append(
            cls_df.sample(PER_CLASS, random_state=RANDOM_STATE)
        )

    balanced_df = pd.concat(balanced_parts, axis=0)
    balanced_df = (
        balanced_df.sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    describe_counts(balanced_df, f"Balanced subset (target {PER_CLASS} per class)")

    # 4) Split 60 / 20 / 20 with stratification
    # First: split off 60% train, remaining 40% -> temp
    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=(1.0 - TRAIN_RATIO),  # 0.40
        random_state=RANDOM_STATE,
        stratify=balanced_df["diagnosis"],
    )

    # Then: split temp into 50/50 -> 20% val, 20% test overall
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),  # 0.20 / 0.40 = 0.5
        random_state=RANDOM_STATE,
        stratify=temp_df["diagnosis"],
    )

    # Save CSVs
    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(VAL_SPLIT_CSV, index=False)
    test_df.to_csv(TEST_SPLIT_CSV, index=False)

    describe_counts(train_df, "Train split (60%)")
    describe_counts(val_df, "Validation split (20%)")
    describe_counts(test_df, "Test split (20%)")

    print("\nFiles created:")
    print(f"  Train CSV: {TRAIN_SPLIT_CSV}")
    print(f"  Val   CSV: {VAL_SPLIT_CSV}")
    print(f"  Test  CSV: {TEST_SPLIT_CSV}")


if __name__ == "__main__":
    main()
























# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # Paths (same as before)
# SOURCE_CSV = "../data/train.csv"
# TRAIN_SPLIT_CSV = "../data/train_split.csv"
# VAL_SPLIT_CSV = "../data/val_split.csv"
#
# # We want 250 images per binary class (0 and 1)
# PER_CLASS = 250
# RANDOM_STATE = 42
# VAL_SIZE = 0.20  # 20% validation, 80% training
#
#
# def describe_counts(df, title):
#     print(f"\n{title}")
#     print("Total samples:", len(df))
#     print(df["diagnosis"].value_counts().sort_index())
#     print(df["diagnosis"].value_counts(normalize=True).sort_index())
#
#
# def main():
#     print(f"Loading data from {SOURCE_CSV}...")
#     try:
#         df = pd.read_csv(SOURCE_CSV)
#     except FileNotFoundError:
#         print(f"Error: {SOURCE_CSV} not found.")
#         print("Please make sure train.csv is in the ../data/ directory.")
#         return
#
#     if "diagnosis" not in df.columns:
#         print("Error: 'diagnosis' column not found in CSV.")
#         return
#
#     # 1) Keep only classes 0, 3, 4
#     df = df[df["diagnosis"].isin([0, 3, 4])].copy()
#     print("After filtering to classes 0, 3, 4:")
#     describe_counts(df, "Original filtered distribution (0,3,4)")
#
#     # 2) Map to binary labels:
#     #    0 -> 0
#     #    3,4 -> 1
#     df["diagnosis_bin"] = df["diagnosis"].map({0: 0, 3: 1, 4: 1})
#
#     # Overwrite 'diagnosis' with binary labels for the rest of the pipeline
#     df["diagnosis"] = df["diagnosis_bin"]
#     df = df.drop(columns=["diagnosis_bin"])
#
#     describe_counts(df, "After mapping to binary labels (0 vs 1)")
#
#     # 3) Balance to PER_CLASS per class (250 each => 500 total)
#     balanced_parts = []
#     for cls in [0, 1]:
#         cls_df = df[df["diagnosis"] == cls]
#         if len(cls_df) < PER_CLASS:
#             raise ValueError(
#                 f"Not enough samples for class {cls}. "
#                 f"Have {len(cls_df)}, need {PER_CLASS}."
#             )
#         balanced_parts.append(
#             cls_df.sample(PER_CLASS, random_state=RANDOM_STATE)
#         )
#
#     balanced_df = pd.concat(balanced_parts, axis=0)
#     # Shuffle rows
#     balanced_df = balanced_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
#
#     describe_counts(balanced_df, f"Balanced subset (target {PER_CLASS} per class)")
#
#     # 4) Train/validation split (stratified)
#     train_df, val_df = train_test_split(
#         balanced_df,
#         test_size=VAL_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=balanced_df["diagnosis"]
#     )
#
#     # Save CSVs
#     train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
#     val_df.to_csv(VAL_SPLIT_CSV, index=False)
#
#     describe_counts(train_df, "Train split")
#     describe_counts(val_df, "Validation split")
#
#     print("\nFiles created:")
#     print(f"  Train CSV: {TRAIN_SPLIT_CSV}")
#     print(f"  Val   CSV: {VAL_SPLIT_CSV}")
#
#
# if __name__ == "__main__":
#     main()
