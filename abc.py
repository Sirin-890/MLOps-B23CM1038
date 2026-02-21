import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from datasets import load_dataset
import wandb
from huggingface_hub import create_repo, upload_file
import shutil


HF_TOKEN = "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"
HF_USERNAME = "b23cm1038"
REPO_NAME = "mlops_resnet50_stl10"


WANDB_API_KEY = "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"
wandb.login(key=WANDB_API_KEY, relogin=True)


ds = load_dataset("Chiranjeev007/STL-10_Subset", streaming=False)  
print("Available splits:", ds.keys())

train_set = ds['train']
val_set   = ds['validation']
test_set  = ds['test']

print("Train:", len(train_set), "Val:", len(val_set), "Test:", len(test_set))


train_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


class CustomDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = CustomDataset(train_set, transform=train_transforms)
val_dataset   = CustomDataset(val_set, transform=test_transforms)
test_dataset  = CustomDataset(test_set, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.resnet50(pretrained=False)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

NUM_EPOCHS = 40


wandb.init(
    project="mlops-resnet50",
    name="minor-resnet50",
    config={
        "architecture": "ResNet18",
        "dataset": "STL-10",
        "epochs": NUM_EPOCHS,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "weight_decay": 5e-4,
        "lr_scheduler": "StepLR",
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
)
wandb.watch(model, criterion, log="all", log_freq=100)

print("Dashboard URL:", wandb.run.get_url())


for epoch in range(NUM_EPOCHS):
    # --- Training ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc  = 100. * correct / total

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= val_total
    val_acc  = 100. * val_correct / val_total

    # --- Test Accuracy ---
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total

    scheduler.step()

    # --- Log to WandB ---
    wandb.log({
        "epoch": epoch+1,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "test/accuracy": test_acc,
        "lr": optimizer.param_groups[0]['lr'],
    })

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
        f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% "
        f"Test Acc: {test_acc:.2f}%"
    )

wandb.finish()


torch.save(model.state_dict(), "resnet18_stl10.pth")
# Correct way to create repo under your own account
HF_TOKEN = "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"
REPO_NAME = "mlops_resnet50_stl10"

# 1️⃣ Create the repo (under your account)
create_repo(
    repo_id=REPO_NAME,   # just the repo name
    token=HF_TOKEN,
    exist_ok=True         # avoids error if repo already exists
)

# 2️⃣ Upload the model file
upload_file(
    path_or_fileobj="resnet18_stl10.pth",
    path_in_repo="resnet18_stl10.pth",  # name inside repo
    repo_id=REPO_NAME,
    token=HF_TOKEN
)

print(f"Model pushed to Hugging Face Hub: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")

