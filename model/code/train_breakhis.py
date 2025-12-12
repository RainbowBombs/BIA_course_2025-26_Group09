"""
Training script for BreaKHis 400X (binary classification with ResNet18).

This script:
  - Builds train / test dataloaders for BreaKHis 400X
  - Trains ResNet18
  - Saves the best checkpoint (by validation accuracy)
  - Logs per-epoch metrics and performance (time + peak GPU memory) to CSV

The evaluation script (eval_breakhis.py) will:
  - Read the CSV log to draw curves
  - Evaluate on internal test set and external IDC dataset
"""

import os
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

import pandas as pd


# ============================================================
# Global configuration (adjust paths if needed)
# ============================================================

# Root directory of BreaKHis 400X (must contain "train" and "test")
BREAKHIS_ROOT = r"C:/Users/17733/Desktop/bia4/ICA/BreaKHis 400X"

# Checkpoint and training log paths
CHECKPOINT_DIR = os.path.join(BREAKHIS_ROOT, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_best.pth")
TRAIN_LOG_CSV = os.path.join(BREAKHIS_ROOT, "training_log.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

CLASS_TO_IDX = {"benign": 0, "malignant": 1}

# Mean / std previously computed on the training set
MEAN = [0.755842387676239, 0.5889061689376831, 0.7419362664222717]
STD = [0.14278094470500946, 0.20091958343982697, 0.1162722110748291]


# ============================================================
# Dataset & Dataloaders
# ============================================================

class BreastHistDataset(Dataset):
    """
    Generic binary dataset: expects two subfolders under root_dir:
      - benign
      - malignant
    """

    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        for class_name, label in CLASS_TO_IDX.items():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    self.samples.append(
                        (os.path.join(class_path, fname), label)
                    )

        print(f"[INFO] Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((700, 460)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10, fill=0),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Validation / test transform
test_transform = transforms.Compose([
    transforms.Resize((700, 460)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def build_breakhis_dataloaders(root: str, batch_size: int = 16):
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    train_set = BreastHistDataset(train_dir, transform=train_transform)
    test_set = BreastHistDataset(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader


# ============================================================
# Model & Training
# ============================================================

def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_for_training(model, dataloader, criterion, device):
    """
    Validation evaluation during training. Returns loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device,
    checkpoint_path: str,
    log_csv_path: str,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1,
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    log_rows = []

    for epoch in range(1, num_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        since = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_for_training(
            model, val_loader, criterion, device
        )
        scheduler.step()

        time_elapsed = time.time() - since

        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_mem_mb = 0.0

        print(
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
            f"Time: {time_elapsed:.1f}s  Peak GPU: {peak_mem_mb:.1f} MB"
        )

        # Log metrics for this epoch
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time_sec": time_elapsed,
                "peak_gpu_mb": peak_mem_mb,
            }
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            print(f"[INFO] New best val acc: {best_acc:.4f}, saved to {checkpoint_path}")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")

    # Save training log to CSV (for plotting in eval_breakhis.py)
    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(log_csv_path, index=False)
    print(f"[INFO] Training log saved to: {log_csv_path}")

    model.load_state_dict(best_model_wts)
    return model


# ============================================================
# Main
# ============================================================

def main():
    train_loader, test_loader = build_breakhis_dataloaders(
        BREAKHIS_ROOT, batch_size=BATCH_SIZE
    )

    model = create_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

    _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # using test set as validation here
        num_epochs=NUM_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        checkpoint_path=CHECKPOINT_PATH,
        log_csv_path=TRAIN_LOG_CSV,
    )


if __name__ == "__main__":
    main()
