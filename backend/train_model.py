"""
Food Safety Model Training Script
Trains a ResNet50 model on food safety dataset and evaluates accuracy
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# Configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/food_safety_model.pth"
ACCURACY_LOG_PATH = "training_metrics.txt"

print(f"Using device: {DEVICE}")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

class FoodDataset(Dataset):
    """Custom Dataset for food safety images"""
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Handle missing images
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            # Return a dummy image to avoid crashes
            image = Image.new('RGB', (224, 224))
        else:
            image = Image.open(img_path).convert('RGB')
        
        label = int(self.img_labels.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders():
    """Create train and test data loaders"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = FoodDataset(
        "dataset/train",
        "dataset/labels_(train).csv",
        transform=transform
    )
    
    test_dataset = FoodDataset(
        "dataset/test",
        "dataset/labels_(train).csv",  # Use same labels as fallback
        transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader

def get_model():
    """Load pre-trained ResNet50 and modify for binary classification"""
    model = models.resnet50(pretrained=True)
    
    # Replace the final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)  # Binary classification: Safe (0) or Unsafe (1)
    )
    
    model = model.to(DEVICE)
    return model

def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'Loss': f'{total_loss/(total//BATCH_SIZE):.4f}'})
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion):
    """Validate model on test set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Validating")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = total_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model():
    """Main training loop"""
    print("\n" + "="*60)
    print("🚀 FOOD SAFETY MODEL TRAINING")
    print("="*60 + "\n")
    
    # Load data
    train_loader, test_loader = get_data_loaders()
    
    # Initialize model
    model = get_model()
    print("\n✓ Model initialized: ResNet50 with binary classification head")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_acc = 0.0
    
    print(f"\n📊 Training Configuration:")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Device: {DEVICE}\n")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(model, test_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"\n📈 Metrics:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   ✓ Model saved! (Best accuracy: {best_acc:.2f}%)")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("🏆 FINAL EVALUATION")
    print(f"{'='*60}\n")
    
    _, _, final_preds, final_labels = validate(model, test_loader, criterion)
    
    accuracy = accuracy_score(final_labels, final_preds)
    precision = precision_score(final_labels, final_preds, average='weighted')
    recall = recall_score(final_labels, final_preds, average='weighted')
    f1 = f1_score(final_labels, final_preds, average='weighted')
    
    print(f"✓ Test Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Precision:      {precision*100:.2f}%")
    print(f"✓ Recall:         {recall*100:.2f}%")
    print(f"✓ F1 Score:       {f1*100:.2f}%")
    
    print(f"\n📋 Classification Report:\n")
    print(classification_report(final_labels, final_preds, target_names=['Safe (0)', 'Unsafe (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(final_labels, final_preds)
    print(f"\n🔢 Confusion Matrix:")
    print(f"   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Save metrics to file
    with open(ACCURACY_LOG_PATH, 'w') as f:
        f.write("FOOD SAFETY MODEL - TRAINING METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"Final Test Accuracy:      {accuracy*100:.2f}%\n")
        f.write(f"Precision:                {precision*100:.2f}%\n")
        f.write(f"Recall:                   {recall*100:.2f}%\n")
        f.write(f"F1 Score:                 {f1*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(final_labels, final_preds, target_names=['Safe (0)', 'Unsafe (1)']))
        f.write("\n\nConfusion Matrix:\n")
        f.write(f"              Predicted\n")
        f.write(f"              Safe  Unsafe\n")
        f.write(f"Actual Safe   {cm[0,0]:6d}  {cm[0,1]:6d}\n")
        f.write(f"Actual Unsafe {cm[1,0]:6d}  {cm[1,1]:6d}\n")
    
    print(f"\n✓ Metrics saved to: {ACCURACY_LOG_PATH}")
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📁 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📊 Metrics saved to: {ACCURACY_LOG_PATH}")

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("✓ Training history plot saved to: training_history.png")
    plt.close()

if __name__ == "__main__":
    train_model()
