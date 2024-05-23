import math
import time

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2, Lambda
import torchvision
from torch.nn import functional as F

### TEST INFORMATION ###

epochs = 1
batch_size = 32
lr = 0.0001
patience = 10
path = 'model1e_500_32_0001'

print("Model: 1e, Epochs: 1, Batch size: 32, Optimiser: Adam, lr=0.0001")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if (torch.cuda.is_available()):
  print(torch.cuda.get_device_name(device))
torch.set_default_device(device)

### DATA AUGMENTATION ###

# We perform random transformations to better generalise the training dataset
img_size = (224, 224)
train_transform = transforms.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=img_size, antialias=True),
    v2.AutoAugment(),
    # v2.Resize((100, 100)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.2),
    v2.RandomErasing(0.4),
    # v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.2),
    v2.RandomRotation(20),
    v2.ToDtype(torch.float32, scale=True),
    # These are the values I have calculated
    v2.Normalize(mean=[0.433, 0.382, 0.296], std=[0.259, 0.209, 0.221])
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    v2.ToImage(),
    v2.Resize(img_size, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.433, 0.382, 0.296], std=[0.259, 0.209, 0.221])
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    v2.ToImage(),
    v2.Resize(img_size, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.433, 0.382, 0.296], std=[0.259, 0.209, 0.221])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

### DATASET ###

# I will download the data from PyTorch's website and use the appropriate data loader
train_dataset = datasets.Flowers102(
    root='',
    split="train",
    download=True,
    transform=train_transform
    # target_transform=Lambda(lambda y: F.one_hot(torch.FloatTensor(y), num_classes=102))
)

valid_dataset = datasets.Flowers102(
    root='',
    split="val",
    download=True,
    transform=valid_transform
    # target_transform=Lambda(lambda y: torch.zeros(102, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_dataset = datasets.Flowers102(
    root='',
    split="test",
    download=True,
    transform=test_transform
    # target_transform=Lambda(lambda y: torch.zeros(102, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

### MODEL ###

# 100 epochs, `autoaugment`, no random resize crop scale, Train 51.7%, Validation 46.9%, Test 41.24%
class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*7*7, 4096),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(4096, 102)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.classifier(x)
        return x

# Choosing model
model = MyNN()
model.to(device)

### TRAINING ###

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser, patience=patience)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, generator=torch.Generator(device=device))
valid_loader = DataLoader(valid_dataset, batch_size, generator=torch.Generator(device=device))

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(dataloader.dataset)
    correct /= size
    print(f"Training Set: Loss: {train_loss:>8f}, Accuracy: {(100*correct):>0.1f}%")
    return train_loss, correct


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * X.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Validation Set: Loss: {test_loss:>8f}, Accuracy: {(100*correct):>0.1f}% \n")
    return test_loss, correct

## MODEL LOOP ##

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_acc = 0.
best_loss = 100
early_stop_counter = 0
for e in range(0, epochs):
    print(f"Epoch [{e+1}/{epochs}]")
    train_loss, train_acc = train_loop(train_loader, model, criterion, optimiser)
    val_loss, val_acc = test_loop(valid_loader, model, criterion)

    lr_scheduler.step(val_loss)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{path}.pth")
        best_loss = val_loss
        early_stop_counter = 0
    # Early Stopping
    # if val_loss > best_loss:
    #     early_stop_counter += 1
    #     if early_stop_counter >= patience:
    #         print(f"Stopping early")
    #         break

# Check to compare the last two losses (if more than 5% difference, save end model)
if (val_acc - best_acc) >= 0.05:
    torch.save(model.state_dict(), f"{path}.pth")
    
### VISUALISATION ###

fig, (axLoss, axAcc) = plt.subplots(nrows=1, ncols=2)

# Visualise the losses
axLoss.set_title("Loss")
axLoss.plot(train_losses, label='Training')
axLoss.plot(val_losses, label='Validation')
axLoss.set(xlabel='Epoch', ylabel='Loss')
axLoss.legend()

# Visualise the losses
axAcc.set_title("Accuracy (Represented as a decimal)")
axAcc.plot(train_losses, label='Training')
axAcc.plot(val_losses, label='Validation')
axAcc.set(xlabel='Epoch', ylabel='Accuracy')
axAcc.legend()

plt.savefig(f"{path}_stats")

### TESTING ###

test_loader = DataLoader(test_dataset, batch_size, generator=torch.Generator(device=device))

def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        acc = .0
        for i, data in enumerate(test_loader):
            X = data[0].to(device)
            y = data[1].to(device)

            predicted = model(X)

            # Check each image's prediction
            acc += (predicted.argmax(dim=1) == y).sum().item()
    model.train()
    return acc/len(test_loader.dataset)

test_model = MyNN().to(device)
test_model.load_state_dict(torch.load(f"{path}.pth"))
test_acc = test(test_model, test_loader)

print(f"Test Accuracy: {test_acc*100:.2f}%")