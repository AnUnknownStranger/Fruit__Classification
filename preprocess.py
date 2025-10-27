import os
from pathlib import Path
import numpy as np
from PIL import Image


BASE_DIR = Path("Fruits Classification")

def load_images_from_folder(folder_path,normalized):
    images = []
    labels = []
    scale = 255 if normalized else 1
    for class_name in sorted(os.listdir(folder_path)):
        class_dir = folder_path / class_name
        if not class_dir.is_dir():
            continue

        for img_file in class_dir.glob("*.*"):
            try:
                img = Image.open(img_file).convert("RGB")  
                img = img.resize((224,224))
                img_array = np.array(img, dtype=np.float32)/scale   
                images.append(img_array)
                labels.append(class_name)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")

    return np.array(images), np.array(labels)


def load_dataset(normalized):
    datasets = {}
    for split in ["train", "valid", "test"]:
        split_path = BASE_DIR / split
        if not split_path.exists():
            print(f"Warning: {split_path} not found.")
            continue
        
        print(f"Loading {split} dataset...")
        X, y = load_images_from_folder(split_path,normalized)
        print(f"{split}: Loaded {X.shape[0]} images.")
        datasets[split] = (X, y)

    return datasets

