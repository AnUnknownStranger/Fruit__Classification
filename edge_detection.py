"""
Edge Detection Features for Fruit Classification
==========================================================
- Edge detection features for fruit classification using canny and hog features.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

BASE_DIR = Path("Fruits Classification")

def load_images(folder_path, max_per_class=200):
    """Load images with limit for faster processing"""
    images = []
    labels = []
    
    for class_name in sorted(os.listdir(folder_path)):
        class_dir = folder_path / class_name
        if not class_dir.is_dir():
            continue
        
        count = 0
        for img_file in class_dir.glob("*.*"):
            if count >= max_per_class:
                break
                
            try:
                img = Image.open(img_file).convert("RGB")  
                img = img.resize((224, 224))
                images.append(np.array(img))
                labels.append(class_name)
                count += 1
            except:
                pass
    
    return images, labels

def extract_canny_features(image):
    """Canny edge detection features"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Just the essential features
    edge_density = np.sum(edges > 0) / edges.size
    
    # Gradient info
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    features = [
        edge_density,
        magnitude.mean() / 255.0,
        magnitude.std() / 255.0,
        np.abs(sobelx).mean() / 255.0,
        np.abs(sobely).mean() / 255.0,
    ]
    
    return np.array(features)

def extract_hog_features(image):
    """HOG features for fruit classification"""
    from skimage.feature import hog
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(32, 32),  # Larger cells = fewer features
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    return features

def test_method(name, extract_func, images_train, images_test, y_train, y_test):
    """Test a single method"""
    print(f"Testing {name}...")
    
    # Extract features
    X_train = np.array([extract_func(img) for img in images_train])
    X_test = np.array([extract_func(img) for img in images_test])
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    return {
        'name': name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'features': X_train.shape[1]
    }

def main():

    
    # Load data
    train_path = BASE_DIR / "train"
    if not train_path.exists():
        print(f"❌ Error: {train_path} not found!")
        return
    
    images, labels = load_images(train_path, max_per_class=200)
    
    # Split data
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(labels_train)
    y_test = le.transform(labels_test)
    

    methods = {
        'Canny': extract_canny_features,
        'HOG': extract_hog_features,
    }
    
    results = []
    for name, func in methods.items():
        result = test_method(name, func, images_train, images_test, y_train, y_test)
        results.append(result)
    
    # Show results
    print(f"\n{'Method':<10} {'Test Acc':<12} {'Features':<10}")
    print("-" * 35)
    
    for result in sorted(results, key=lambda x: x['test_acc'], reverse=True):
        print(f"{result['name']:<10} {result['test_acc']:.4f}      {result['features']:<10}")
    

if __name__ == "__main__":
    main()
