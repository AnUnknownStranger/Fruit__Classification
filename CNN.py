import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from preprocess import load_dataset

MODEL_PATH = "model.pth"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, label2idx: dict):
        self.X = X.astype(np.float32)
        self.y = np.array([label2idx[c] for c in y])
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)

    def __len__(self):
        return len(self.X)
    
    #Load the image as torch tensor and normalize the image
    def __getitem__(self, idx):
        img = self.X[idx]
        img = (img - self.mean) / self.std
        img_t = torch.from_numpy(img).permute(2,0,1)
        label = int(self.y[idx])
        return img_t, label

def build_model(num_classes):
    #Load the restnet18 model
    model = models.resnet18(pretrained=True)
    #Convert the final layer to linear layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def train(normalized=True):
    #Load the dataset
    datasets = load_dataset(normalized,0.1)

    X_train, y_train = datasets["train"]
    X_valid, y_valid = datasets["valid"]
    X_test,  y_test  = datasets["test"]

    #Sort the classes based on name
    classes = sorted(set(y_train.tolist()))
    #Convert each label to numerate id 
    label2idx = {c:i for i,c in enumerate(classes)}
    
    #Convert dataset to numpy dataset
    ds_train = NumpyDataset(X_train, y_train, label2idx)
    ds_val   = NumpyDataset(X_valid, y_valid, label2idx)
    ds_test  = NumpyDataset(X_test,  y_test,  label2idx)

    #Load the data with bacthsize
    loader_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=32, shuffle=False)
    loader_test  = DataLoader(ds_test,  batch_size=32, shuffle=False)
    #load the model
    model = build_model(num_classes=len(classes))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    #Perform training while keeping the best accuracy model
    best_val_acc = 0.0
    for epoch in range(0, 10):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for xb, yb in loader_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            preds = out.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            seen += xb.size(0)
        train_loss = total_loss / seen
        train_acc = correct / seen

        #Compute validation accuracy of the current trained model
        model.eval()
        correct = 0
        seen = 0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                seen += xb.size(0)
        val_acc = correct / seen

        print(f"Epoch {epoch+1}/{10}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  time={(time.time()-t0):.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> saved best model:", MODEL_PATH)

    #Compute testing accuracy
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    correct = 0
    seen = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader_test:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(yb.cpu().numpy().tolist())
            correct += int((preds == yb).sum().item())
            seen += xb.size(0)
    test_acc = correct / seen
    print(f"Test accuracy: {test_acc:.4f}  (classes: {classes})")
    return model


if __name__ == "__main__":
    model = train(normalized=True)