import preprocess
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def convert_hist(img):
    #Convert img pixels to hsv histogram
    h_hist = cv2.calcHist([img], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([img], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([img], [2], None, [32], [0, 256])
    combined = np.concatenate([h_hist.ravel(), s_hist.ravel(), v_hist.ravel()]).astype(np.float32)
    total = combined.sum()
    #Normalize
    return combined/total


def convert(img,label):
    imgs = []
    for i in range(len(img)):
        imgs.append(convert_hist(img[i]))
    X = np.vstack(imgs)
    return X,label

def cvhis():
    #Load the data
    dataset = preprocess.load_dataset(False)
    #load and convert the data into HSV histogram
    X_train_img, y_t_l = dataset['train']
    X_val,y_val = dataset['valid']
    X_test, y_test = dataset['test']
    X_train, y_train = convert(X_train_img,y_t_l)
    X_val, y_val = convert(X_val,y_val)
    X_test,y_test = convert(X_test,y_test)
    #Encode the label
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    #Apply SVM and grid search to find the best parameter
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svm", SVC(kernel='rbf', probability=True, class_weight="balanced")),
    ])
    grid = {
        "svm__C":    [1, 3, 10, 30, 100],
        "svm__gamma":[ "scale", "auto", 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 ],
    }
    best_mod = GridSearchCV(model,grid, cv=5, n_jobs=-1, verbose=1)
    best_mod.fit(X_train, y_train)
    
    #Initiate a report function
    def report(name, X, y):
        y_pred = best_mod.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        print(f"[{name}] accuracy={acc:.4f} macroF1={f1:.4f}")
        if name == "test":
            print(classification_report(y, y_pred, target_names=le.classes_))

    report("train", X_train, y_train)
    report("valid", X_val, y_val)
    report("test", X_test, y_test)
    return



if __name__ == "__main__":
    cvhis()








