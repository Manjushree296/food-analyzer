import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

data_dir = "dataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    print("Epoch", epoch + 1)
    for images, labels in train_loader:
        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model, "models/food_safety_model.pth")
print("Model Saved!")
