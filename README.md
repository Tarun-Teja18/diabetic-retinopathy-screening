# 👁️ AI-assisted Diabetic Retinopathy Screening from Fundus Images

This repository contains the source code, training, and evaluation scripts for an AI-assisted Diabetic Retinopathy (DR) screening system. The project benchmarks three state-of-the-art deep learning architectures—**ResNet-50**, **EfficientNet-B0**, and a **Vision Transformer (ViT-B/16)**—on a high-stakes, binary classification task: distinguishing **'No DR' (Healthy)** from **'Severe-Stage DR'**.

This work was developed as a final project for a Machine Vision course (CS7367) offered by Kennesaw State University.

## 🚀 Project Goal

The objective was to develop a fast and reliable deep learning system for early, targeted DR screening. By simplifying the standard 5-class DR grading into a binary classification task (Class 0: Healthy vs. Class 1: Severe/Proliferative), we focused on identifying high-risk cases that require immediate clinical attention.

## 💻 Repository Structure

| File/Folder | Description |
| :--- | :--- |
| `split_data.py` | Utility script used to filter the original dataset, map it to a balanced binary task, and perform the 60/20/20 train/validation/test split. |
| `dataset.py` | Defines the `RetinopathyDataset` class and image transformation pipelines, including custom **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement. |
| `model_cnn.py` | Contains functions to load and adapt ResNet-50 and EfficientNet-B0 models for binary classification. |
| `model_vit.py` | Contains the function to load and adapt the ViT-B/16 model from the `timm` library. |
| `train.py` | Main script for training models (ResNet, EfficientNet, ViT). Handles data loading, optimization, and checkpoint saving. |
| `evaluate.py` | Script for final evaluation on the held-out test set, calculating Accuracy, F1-score, AUC-ROC, and the Confusion Matrix. |
| `plot_logs.py` | Script to load JSON history logs and generate training/validation curves (Loss, Accuracy, F1-score) and final ROC curves for comparative analysis. |
| `explain.py` | Script implementing **Grad-CAM** (Gradient-weighted Class Activation Mapping) for model interpretability and visualization. |
| `../outputs/` | (Expected folder) Contains saved model checkpoints (`.pth`), training history logs (`.json`), and generated plots/heatmaps (`.png`). |

## ⚙️ Methodology & Pipeline

The project follows a rigorous four-stage pipeline:

1.  **Data Preparation:** Filtering the APTOS 2019 dataset to include only Class 0 (Healthy), Class 3 (Severe DR), and Class 4 (Proliferative DR). These were then mapped to a balanced binary set (0 vs. 1).
2.  **Preprocessing & Augmentation:** All images are resized to **224x224** pixels. Training images undergo **RandomHorizontalFlip**, **RandomRotation**, and **CLAHE** for contrast enhancement. All images are normalized using ImageNet statistics.
3.  **Model Training:** Benchmarking ResNet-50, EfficientNet-B0, and ViT-B/16, all initialized with ImageNet weights and adapted for 2-class output. Models were trained for 10 epochs using **AdamW** and **CosineAnnealingLR**.
4.  **Evaluation & Interpretability:** Quantitative assessment using **Accuracy, F1-score, and AUC-ROC**, followed by qualitative assessment using **Grad-CAM** to visualize model attention.


---

## 📊 Key Results

Two experiments were conducted: one on a smaller dataset ($N=500$) and one on a larger dataset ($N=976$). All models achieved near-perfect performance on this balanced, binary task, demonstrating the high separability of the simplified problem.

### Experiment 1: Small Dataset ($N=250$/class)

This experiment used a small, balanced dataset of **500 total images** (300 train, 100 val, 100 test). All three models achieved near-perfect performance.

| Model | Accuracy | F1-score | AUC |
| :--- | :--- | :--- | :--- |
| **ResNet-50** | **1.0000** | **1.0000** | **1.0000** |
| **EfficientNet-B0** | **1.0000** | **1.0000** | **1.0000** |
| **ViT-B/16** | 0.9900 | 0.9899 | 0.9988 |

#### Training Dynamics (Validation F1-score)
On this smaller dataset, the **ResNet-50** model had a strong start, and the **ViT** model learned extremely quickly, matching the ResNet by epoch 4. The **EfficientNet** model showed the steadiest, most gradual improvement.


---

### Experiment 2: Larger Dataset ($N=488$/class)

This experiment repeated the process with a more robust, balanced dataset of **976 total images** (585 train, 195 val, 196 test). Both ResNet-50 and ViT achieved perfect scores on the test set.

| Model | Accuracy | F1-score | AUC |
| :--- | :--- | :--- | :--- |
| **ResNet-50** | **1.0000** | **1.0000** | **1.0000** |
| **EfficientNet-B0** | 0.9897 | 0.9896 | 0.9999 |
| **ViT-B/16** | **1.0000** | **1.0000** | **1.0000** |

* *Note: EfficientNet's two misclassifications were False Positives (Healthy classified as Severe-Stage DR).*

#### Training Dynamics (Validation F1-score)
With the larger dataset, the **ViT** model was the fastest to converge, achieving a perfect F1-score of **1.0** by only the second epoch.

---

## 🔍 Explainability (Grad-CAM Findings)

The most critical finding was revealed by **Grad-CAM**, which showed that quantitative metrics alone do not guarantee clinical trustworthiness.

* **CNNs (ResNet/EfficientNet):** The attention heatmaps for the CNN models consistently focused on the actual **pathological regions (lesions)** in the severe images, indicating they learned the correct clinical features. For healthy images, the focus was appropriately diffuse.
* **Vision Transformer (ViT):** Despite achieving a perfect score, the ViT's attention maps for severe images were clearly drawn to the **black borders/artifacts** at the edge of the image, rather than the lesions themselves. This suggests the ViT model may have relied on a confounding artifact for its prediction, making the CNNs the more trustworthy choice for clinical adoption in this scenario.

### Grad-CAM Visualization Summary

| | Healthy (Class 0) | Severe-Stage DR (Class 1) |
| :---: | :---: | :---: |
| **ResNet/EfficientNet Attention** | Broad, diffuse (Correct focus) | Focused on **lesion areas** (Correct focus) |
| **ViT-B/16 Attention** | Arbitrary spots (Non-clinical focus) | Focused on **image border/artifact** (Confounding shortcut) |

---

## 🚀 How to Run the Code

### Prerequisites

1.  **Dataset:** Download the APTOS 2019 Blindness Detection dataset and place the `train.csv` and image files in a `../data/` directory relative to the scripts.
2.  **Environment:** Ensure you have PyTorch, NumPy, scikit-learn, `tqdm`, `timm`, `pandas`, and `pytorch-grad-cam` installed.

### 1. Prepare Data Splits

Run the `split_data.py` script to generate the balanced binary training, validation, and testing CSV files.

```bash
python split_data.py
```

### 2\. Train a Model

Run `train.py` and enter the model type you wish to train (`resnet`, `efficientnet`, or `vit`).

```bash
python train.py
# Example: Enter model you want to train (resnet / efficientnet / vit): resnet
```

### 3\. Evaluate the Model

Once training is complete, run `evaluate.py` to get the final test set metrics (Accuracy, F1-score, AUC, Confusion Matrix).

```bash
python evaluate.py
# Example: Enter model type to evaluate (resnet / efficientnet / vit): resnet
```

### 4\. Generate Plots (Optional)

Run `plot_logs.py` to generate and save the training/validation curves and the combined ROC plot for all models in the `../outputs/plots` folder.

```bash
python plot_logs.py
```

### 5\. Visualize with Grad-CAM

Run `explain.py` to generate and save a Grad-CAM heatmap for a chosen image and model.

```bash
python explain.py
# Follow console prompts to select model and image ID.
```
