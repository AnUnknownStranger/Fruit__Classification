## Fruit Classification (Group 23)

Computer Vision project exploring **handcrafted features** (HSV histogram, edges/HOG, texture/GLCM) and a **CNN baseline (ResNet-18)** for fruit image classification.

## Team (Group 23)

- **Wei Chen** — `wc2917`
- **Lamees Alwasil** — `laa2203`
- **Arth Singh** — `as7389`


## Datasets

- **Fruits Classification** (used for classical feature experiments): [Kaggle – Fruits Classification](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification)
- **Fruit Recognition** (additional dataset used in the project): [Kaggle – Fruit Recognition](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition)

## Project structure

- `Fruits Classification/`
  - `train/<class_name>/*`
  - `valid/<class_name>/*`
  - `test/<class_name>/*`
- `preprocess.py`: dataset loader used by `hsv_hist.py` and `CNN.py`
- `hsv_hist.py`: HSV histogram + SVM (GridSearchCV)
- `edge_detection.py`: edge-based features (Canny stats) + HOG features + Random Forest
- `texture_features.py`: GLCM/Haralick texture features + Random Forest
- `compare_models.py`: combined features (**HSV+HOG** vs **HSV+Canny**) + Random Forest
- `CNN.py`: ResNet-18 training using PyTorch

## Setup

This repo assumes you have Python installed and the common CV/ML packages available (e.g., `numpy`, `opencv-python`, `scikit-learn`, `scikit-image`, `Pillow`, and for `CNN.py`: `torch` + `torchvision`).

Make sure the dataset folder is located at:

- `Fruits Classification/` (relative to the project root)

If your dataset is in a different location, update `BASE_DIR` in the scripts that define it (e.g., `edge_detection.py`, `texture_features.py`, `compare_models.py`, `preprocess.py`).

## How to run

From the project root:

- **HSV Color Histogram **:

```bash
python hsv_hist.py
```

- **Edge Detection**:

```bash
python edge_detection.py
```

- **Texture (GLCM)**:

```bash
python texture_features.py
```

- **Combined Features**:

```bash
python compare_models.py
```

- **CNN baseline (ResNet-18)**:

```bash
python CNN.py
```

## Notes

- Classical feature scripts (`hsv_hist.py`, `edge_detection.py`, `texture_features.py`, `compare_models.py`) use a **single stratified train/test split** for evaluation unless otherwise specified in the script.
- `Fruits Classification/data_splitting.py` is included as a helper script to create `train/valid/test` folders (use with care, as it moves files).
