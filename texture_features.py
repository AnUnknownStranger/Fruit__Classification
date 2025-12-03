"""
Texture Features for Fruit Classification
==========================================================
Texture feature extraction for fruit classification using:
- Gray-Level Co-occurrence Matrix (GLCM) - Haralick features
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from skimage.feature import graycomatrix, graycoprops

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

def extract_glcm_features(image):
    """
    Gray-Level Co-occurrence Matrix (GLCM) - Haralick features
    Captures texture properties like contrast, correlation, energy, homogeneity
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Quantize to reduce computation (0-255 -> 0-15)
    gray_quantized = (gray // 16).astype(np.uint8)
    
    # Calculate GLCM for different distances and angles
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    features = []
    
    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(gray_quantized, 
                               distances=[distance], 
                               angles=[angle],
                               levels=16,
                               symmetric=True,
                               normed=True)
            
            # Extract Haralick features
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            
            features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
    
    return np.array(features)

def report(name, X, y, model, le):
    """Detailed reporting function for each dataset split"""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    print(f"[{name}] accuracy={acc:.4f} macroF1={f1:.4f}")
    if name == "test":
        print(classification_report(y, y_pred, target_names=le.classes_))
    return acc, f1

def test_method(name, extract_func, images_train, images_test, y_train, y_test, le):
    """Test a single method with detailed reporting"""
    print(f"\n{'='*60}")
    print(f"Testing {name}...")
    print(f"{'='*60}")
    
    # Extract features
    X_train = np.array([extract_func(img) for img in images_train])
    X_test = np.array([extract_func(img) for img in images_test])
    
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Train model with regularization to prevent overfitting
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,  # Limit tree depth to prevent overfitting
        min_samples_split=5,  
        min_samples_leaf=2,  
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Detailed reporting
    train_acc, train_f1 = report("train", X_train, y_train, clf, le)
    test_acc, test_f1 = report("test", X_test, y_test, clf, le)
       
    return {
        'name': name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'features': X_train.shape[1],
        'model': clf
    }

def main():
    """Main function to test texture feature methods"""
    
    # Load data
    train_path = BASE_DIR / "train"
    if not train_path.exists():
        print(f"Error: {train_path} not found!")
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
    
    # Test GLCM method
    result = test_method('GLCM', extract_glcm_features, images_train, images_test, y_train, y_test, le)
    
   
if __name__ == "__main__":
    main()
