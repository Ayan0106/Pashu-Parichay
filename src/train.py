import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import copy

# =============================
# Configuration
# =============================

DATA_PATH = '/kaggle/input/datasets/lukex9442/indian-bovine-breeds/Indian_bovine_breeds/Indian_bovine_breeds'
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 0.05
MODEL_SAVE_PATH = 'bovine_model.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Data Transforms
# =============================

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# =============================
# Dataset Loading
# =============================

class DatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to apply transforms to a Subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_dataloaders(data_path, batch_size=32):
    full_dataset = datasets.ImageFolder(data_path)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_raw, val_raw = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = DatasetWrapper(train_raw, data_transforms['train'])
    val_dataset = DatasetWrapper(val_raw, data_transforms['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    return dataloaders, class_names, num_classes


# =============================
# Model Initialization
# =============================

def initialize_model(num_classes):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


# =============================
# Training Function
# =============================

def train_model(model, dataloaders, num_epochs=30):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, history


# =============================
# Plot Results + Confusion Matrix
# =============================

def plot_results(history, model, dataloaders, class_names):
    # Loss & Accuracy curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # Confusion Matrix
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


# =============================
# Main
# =============================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    dataloaders, class_names, num_classes = get_dataloaders(DATA_PATH, BATCH_SIZE)
    print(f"Classes ({num_classes}): {class_names}")

    model = initialize_model(num_classes)

    model, history = train_model(model, dataloaders, NUM_EPOCHS)

    plot_results(history, model, dataloaders, class_names)

    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names
    }, MODEL_SAVE_PATH)

    print(f"\nModel saved to {MODEL_SAVE_PATH}")
