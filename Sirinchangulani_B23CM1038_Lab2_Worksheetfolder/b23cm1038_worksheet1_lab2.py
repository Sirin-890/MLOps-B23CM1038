
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from ptflops import get_model_complexity_info
import wandb
from loguru import logger
import copy

BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(
    project="cifar10-gradient-weight-flow",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "architecture": "ResNet18",
        "dataset": "CIFAR-10"
    }
)

class CIFAR10Custom(Dataset):
    def __init__(self, root, train=True):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=True
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label

train_dataset = CIFAR10Custom("./data", train=True)
test_dataset  = CIFAR10Custom("./data", train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

train_dataset = CIFAR10Custom("./data", train=True)
test_dataset  = CIFAR10Custom("./data", train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def get_cifar_resnet18():
    model = resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model

model = get_cifar_resnet18().to(DEVICE)

flops, params = get_model_complexity_info(
    model,
    (3, 32, 32),
    as_strings=True,
    print_per_layer_stat=False
)

logger.debug(f"FLOPs: {flops}")
logger.debug(f"Params: {params}")

wandb.log({
    "FLOPs": flops,
    "Parameters": params
})

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def log_gradient_flow(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log(
                {f"grad_norm/{name}": param.grad.norm().item()},
                step=step
            )

def log_weight_update(model, prev_weights, step):
    for name, param in model.named_parameters():
        if name in prev_weights:
            delta = (param.data - prev_weights[name]).norm().item()
            wandb.log(
                {f"weight_update/{name}": delta},
                step=step
            )

global_step = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        prev_weights = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
        }

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        log_gradient_flow(model, global_step)

        optimizer.step()

        log_weight_update(model, prev_weights, global_step)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        global_step += 1

    train_acc = 100.0 * correct / total
    train_loss = running_loss / len(train_loader)

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc
    })

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%"
    )
    print("#"*100)

wandb.finish()







