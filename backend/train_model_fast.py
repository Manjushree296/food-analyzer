"""
Fast Food Safety Model Training Script
Uses a lightweight MobileNetV2 model for faster training
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")  # Force CPU for faster training on Windows
MODEL_SAVE_PATH = "models/food_safety_model.pth"

print(f"Device: {DEVICE}")
os.makedirs("models", exist_ok=True)

class FoodDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(labels_csv)
        self.transform = transform
        self.valid_indices = []
        
        # Check which images exist
        for idx in range(len(self.img_labels)):
            img_name = self.img_labels.iloc[idx, 0]
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
        
        print(f"✓ Found {len(self.valid_indices)} valid images")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_name = self.img_labels.iloc[actual_idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels.iloc[actual_idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FoodDataset("dataset/train", "dataset/labels_(train).csv", transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def get_model():
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    return model.to(DEVICE)

def train():
    print("\n" + "="*60)
    print("🚀 FOOD SAFETY MODEL TRAINING (FAST VERSION)")
    print("="*60 + "\n")
    
    train_loader, val_loader = get_data_loaders()
    model = get_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss, train_acc, train_total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_acc += (predicted == labels).sum().item()
        
        train_acc = 100 * train_acc / train_total
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss, val_acc, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_acc += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100 * val_acc / val_total
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Model saved (Best: {best_acc:.2f}%)")
    
    # Final metrics
    print(f"\n{'='*60}")
    print("📊 FINAL METRICS")
    print(f"{'='*60}")
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\n✓ Best Validation Accuracy: {best_acc:.2f}%")
    print(f"✓ Final Test Accuracy:      {accuracy*100:.2f}%")
    print(f"✓ Precision:                {precision*100:.2f}%")
    print(f"✓ Recall:                   {recall*100:.2f}%")
    print(f"✓ F1 Score:                 {f1*100:.2f}%")
    
    # Save metrics
    with open("training_metrics.txt", "w") as f:
        f.write("FOOD SAFETY MODEL - TRAINING METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"Final Test Accuracy:      {accuracy*100:.2f}%\n")
        f.write(f"Precision:                {precision*100:.2f}%\n")
        f.write(f"Recall:                   {recall*100:.2f}%\n")
        f.write(f"F1 Score:                 {f1*100:.2f}%\n")
    
    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Metrics saved to: training_metrics.txt")
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
