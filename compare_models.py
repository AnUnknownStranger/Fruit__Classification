"""
Combined Fruit Classification
==========================================================
Combines features from hsv_hist.py and edge_detection.py
for improved fruit classification performance.
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import functions from the other files
from hsv_hist import convert_hist
from edge_detection import extract_canny_features, extract_hog_features, load_images

BASE_DIR = Path("Fruits Classification")

# ============================================================================
# Combined Feature Extraction
# ============================================================================

def extract_combined_features(image):
    """
    Combine all features using functions from both files:
    - HSV histogram from hsv_hist.py
    - Canny features from edge_detection.py
    - HOG features from edge_detection.py
    """
    # Convert RGB to HSV for histogram (hsv_hist.py expects HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Extract HSV histogram using function from hsv_hist.py
    hsv_feat = convert_hist(hsv)
    
    # Extract Canny features using function from edge_detection.py
    canny_feat = extract_canny_features(image)
    
    # Extract HOG features using function from edge_detection.py
    hog_feat = extract_hog_features(image)
    
    # Concatenate features (compare HSV+Canny vs HSV+HOG; do NOT combine Canny+HOG)
    combined_hsv_hog = np.concatenate([hsv_feat, hog_feat])
    combined_hsv_canny = np.concatenate([hsv_feat, canny_feat])
    
    return combined_hsv_hog, combined_hsv_canny 

# ============================================================================
# Model Training and Evaluation
# ============================================================================
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model using the combined features
    """
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    return acc, f1


def main():
    """Main function to compare models"""

    # Load data
    train_path = BASE_DIR / "train"
    if not train_path.exists():
        print(f"Error: Train directory not found at {train_path}")
        return
    
    # Load images using the correct function signature
    images, labels = load_images(train_path, max_per_class=200)
    
    # Split data
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Encode labels (convert strings to numbers)
    le = LabelEncoder()
    y_train = le.fit_transform(labels_train)
    y_test = le.transform(labels_test)
        
    # Extract combined features
    print("\nExtracting combined features...")
    
    # Extract features for training set
    train_features = [extract_combined_features(img) for img in images_train]
    X_train_hsv_hog = np.array([feat[0] for feat in train_features])
    X_train_hsv_canny = np.array([feat[1] for feat in train_features])
    
    # Extract features for test set
    test_features = [extract_combined_features(img) for img in images_test]
    X_test_hsv_hog = np.array([feat[0] for feat in test_features])
    X_test_hsv_canny = np.array([feat[1] for feat in test_features]) 
    
    print(f"Feature dimension (HSV + HOG): {X_train_hsv_hog.shape[1]}")
    print(f"Feature dimension (HSV + Canny): {X_train_hsv_canny.shape[1]}")
    
    # Train and evaluate models
    acc_hsv_hog, f1_hsv_hog = train_and_evaluate_model(X_train_hsv_hog, y_train, X_test_hsv_hog, y_test)
    acc_hsv_canny, f1_hsv_canny = train_and_evaluate_model(X_train_hsv_canny, y_train, X_test_hsv_canny, y_test)
    print(f"\nResults:")
    print(f"Accuracy (HSV + HOG): {acc_hsv_hog:.4f}")
    print(f"F1 Score (HSV + HOG): {f1_hsv_hog:.4f}")
    print(f"Accuracy (HSV + Canny): {acc_hsv_canny:.4f}")
    print(f"F1 Score (HSV + Canny): {f1_hsv_canny:.4f}")
    
if __name__ == "__main__":

    main()