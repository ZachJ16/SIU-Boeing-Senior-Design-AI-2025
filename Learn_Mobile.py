import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
import os

# 1. Transforms (Including Data Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

# 2. Load Dataset
train_dataset = ImageFolder('train', transform=transform)
val_dataset = ImageFolder('val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 3. Calculate class weights based on the frequency of each class in the training set
class_counts = np.array([len([x for x in train_dataset.samples if x[1] == i]) for i in range(len(train_dataset.classes))])
class_weights = 1. / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 4. Load MobileNetV2
model = mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))  # Adjust for number of classes

# Load previous model weights if available
model_path = "mobilenetv2_maze.pth"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        print("Loaded pretrained weights from mobilenetv2_maze.pth")
    except Exception as e:
        print(f"Warning: Could not load weights. Error: {e}")
else:
    print("No pretrained weights found, training from scratch.")

# 5. Loss & Optimizer with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 6. Training loop (Extended Epochs)
for epoch in range(50):  
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
    
    accuracy = (correct_preds / total_preds) * 100
    print(f"Epoch {epoch+1}/30, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    # Validation
    if (epoch+1) % 5 == 0:
        model.eval()
        val_correct = 0
        val_total = 0
        all_true_labels = []
        all_pred_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(predicted.cpu().numpy())

        val_accuracy = (val_correct / val_total) * 100
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(classification_report(all_true_labels, all_pred_labels, target_names=train_dataset.classes))

# Save the updated model
torch.save(model.state_dict(), "mobilenetv2_maze.pth")
print("Model saved to mobilenetv2_maze.pth")
