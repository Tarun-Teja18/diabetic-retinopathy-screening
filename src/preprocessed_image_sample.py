import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2

# --- Configuration ---
# Set these paths to match your project
IMAGE_FOLDER = "../data/train_images"
OUTPUT_FOLDER = "../outputs/unnormalized_images"


# --- End Configuration ---

# -----------------------------
# Same CLAHE class from your dataset.py
# -----------------------------
class ApplyCLAHE(object):
    """Apply CLAHE to enhance contrast."""

    def _init_(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def _call_(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid_size)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_eq)


# -----------------------------
# Transform (NOW includes augmentations)
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Full TRAINING preprocessing pipeline
transform_with_augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # <--- ADDED
    transforms.RandomRotation(10),  # <--- ADDED
    ApplyCLAHE(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Un-normalization transform
unnormalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1 / s for s in IMAGENET_STD]
)


# -----------------------------
# FUNCTION: preprocess and save
# -----------------------------
def preprocess_unnormalize_and_save(img_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    # Load image
    image = Image.open(img_path).convert("RGB")

    print("Applying random training augmentations...")
    # Apply full preprocessing
    preprocessed_tensor = transform_with_augmentations(image)

    # Convert tensor back to image for saving (un-normalize)
    img_unnorm = unnormalize(preprocessed_tensor)

    # Clip values to [0, 1] range after un-normalization
    img_unnorm = torch.clamp(img_unnorm, 0, 1)

    # Convert to NumPy array, scale to [0, 255]
    img_np = (img_unnorm.permute(1, 2, 0).numpy() * 255).astype("uint8")

    # Save output image
    img_name = os.path.basename(img_path)
    # Add a suffix to avoid overwriting
    file_name, file_ext = os.path.splitext(img_name)
    save_name = f"{file_name}_augmented_unnormalized{file_ext}"

    save_path = os.path.join(output_folder, save_name)
    Image.fromarray(img_np).save(save_path)

    print(f"✅ Preprocessed & un-normalized image saved at: {save_path}")


# -----------------------------
# MAIN (run from terminal)
# -----------------------------
if __name__ == "__main__":
    print(f"Loading images from: {IMAGE_FOLDER}")
    print(f"Saving un-normalized images to: {OUTPUT_FOLDER}")

    try:
        img_name = input("Enter image name (e.g., '10_left.png'): ")
        full_path = os.path.join(IMAGE_FOLDER, img_name)
        preprocess_unnormalize_and_save(full_path, OUTPUT_FOLDER)

    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"An error occurred: {e}")