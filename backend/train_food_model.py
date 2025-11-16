"""
Food Safety Model Training - No External Downloads
Trains a simple CNN model from scratch
"""

import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import Dataset, DataLoader, random_split  # type: ignore
from torchvision import transforms  # type: ignore
from PIL import Image  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # type: ignore
import sys

BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cpu")
MODEL_SAVE_PATH = "models/food_safety_model.pth"

print(f"Using Device: {DEVICE}")
os.makedirs("models", exist_ok=True)

class FoodDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(labels_csv)
        self.transform = transform
        self.valid_indices = []
        
        for idx in range(len(self.img_labels)):
            img_name = self.img_labels.iloc[idx, 0]
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_name = self.img_labels.iloc[actual_idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = int(self.img_labels.iloc[actual_idx, 1])
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except:
            image = torch.randn(3, 224, 224)
            label = 0
            return image, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train():
    print("\n" + "="*60)
    print("FOOD SAFETY MODEL TRAINING (LOCAL)")
    print("="*60 + "\n")
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FoodDataset("dataset/train", "dataset/labels_(train).csv", transform=transform)
    print(f"Found {len(dataset)} valid images")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    # Create model
    print("Creating CNN model (no external downloads)...")
    model = SimpleCNN().to(DEVICE)
    print("Model created\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    best_preds = None
    best_labels = None
    
    print(f"Training for {EPOCHS} epochs...\n")
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_preds = all_preds
            best_labels = all_labels
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Model saved (Best: {best_acc:.2f}%)")
    
    # Calculate final metrics
    print(f"\n{'='*60}")
    print("FINAL METRICS")
    print(f"{'='*60}\n")
    
    accuracy = accuracy_score(best_labels, best_preds)
    precision = precision_score(best_labels, best_preds, average='weighted', zero_division=0)
    recall = recall_score(best_labels, best_preds, average='weighted', zero_division=0)
    f1 = f1_score(best_labels, best_preds, average='weighted', zero_division=0)
    
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Precision:                {precision*100:.2f}%")
    print(f"Recall:                   {recall*100:.2f}%")
    print(f"F1 Score:                 {f1*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(best_labels, best_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Safe  Unsafe")
    print(f"Actual Safe     {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Actual Unsafe   {cm[1,0]:6d}  {cm[1,1]:6d}\n")
    
    # Performance metrics
    tpr = 100 * cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    tnr = 100 * cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    fpr = 100 * cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    fnr = 100 * cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    print(f"Performance Analysis:")
    print(f"   True Positive Rate (Sensitivity):  {tpr:.2f}%")
    print(f"   True Negative Rate (Specificity):  {tnr:.2f}%")
    print(f"   False Positive Rate:               {fpr:.2f}%")
    print(f"   False Negative Rate:               {fnr:.2f}%")
    
    # Save metrics to file
    with open("training_metrics.txt", "w") as f:
        f.write("FOOD SAFETY MODEL - TRAINING METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Type:               Simple CNN (3 Conv Layers)\n")
        f.write(f"Training Samples:         {len(train_dataset)}\n")
        f.write(f"Validation Samples:       {len(val_dataset)}\n")
        f.write(f"Epochs:                   {EPOCHS}\n")
        f.write(f"Batch Size:               {BATCH_SIZE}\n\n")
        f.write(f"ACCURACY METRICS:\n")
        f.write(f"-" * 60 + "\n")
        f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"Precision:                {precision*100:.2f}%\n")
        f.write(f"Recall:                   {recall*100:.2f}%\n")
        f.write(f"F1 Score:                 {f1*100:.2f}%\n\n")
        f.write(f"CONFUSION MATRIX:\n")
        f.write(f"-" * 60 + "\n")
        f.write(f"                Predicted\n")
        f.write(f"                Safe  Unsafe\n")
        f.write(f"Actual Safe     {cm[0,0]:6d}  {cm[0,1]:6d}\n")
        f.write(f"Actual Unsafe   {cm[1,0]:6d}  {cm[1,1]:6d}\n\n")
        f.write(f"DETAILED PERFORMANCE:\n")
        f.write(f"-" * 60 + "\n")
        f.write(f"True Positive Rate (Sensitivity):  {tpr:.2f}%\n")
        f.write(f"True Negative Rate (Specificity):  {tnr:.2f}%\n")
        f.write(f"False Positive Rate:               {fpr:.2f}%\n")
        f.write(f"False Negative Rate:               {fnr:.2f}%\n")
    
    print(f"\nModel saved to:     {MODEL_SAVE_PATH}")
    print(f"Metrics saved to:   training_metrics.txt")
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}\n")
    
    return best_acc, accuracy, precision, recall, f1

if __name__ == "__main__":
    try:
        best_acc, accuracy, precision, recall, f1 = train()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
