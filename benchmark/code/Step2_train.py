"""
Step2_train.py

Benchmark training script for six backbone models:

  - CNN family (torchvision): resnet18, resnet34, densenet121
  - Token-mixer family (timm): vit, tokenmixer, convmixer

All baselines are trained on the same BreaKHis 400X dataset with the same
data augmentation defined in Step1_dataloader.py (no dataset-specific
normalization), so that we can compare backbones consistently.

The script:
  1. Builds DataLoaders from: data/BreaKHis 400X/train and data/BreaKHis 400X/test
  2. Creates the selected model (optionally ImageNet-pretrained)
  3. Trains for N epochs
  4. Saves the best checkpoint (by test accuracy) into:
     benchmark/model_result/<model_name>_best.pth
"""

import os
import time
import copy
import argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import timm  # ViT / MLP-Mixer / ConvMixer

from Step1_dataloader import build_dataloaders


# -------------------------------------------------
# 1. Model factory
# -------------------------------------------------
def create_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create a backbone model for binary classification.

    Args:
        model_name: one of
            ["resnet18", "resnet34", "densenet121", "vit", "tokenmixer", "convmixer"]
        num_classes: output dimension (2 for benign / malignant)
        pretrained: whether to use ImageNet pretrained weights

    Returns:
        torch.nn.Module
    """
    name = model_name.lower()

    # --- CNN family (torchvision) ---
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # --- Token / patch family (timm) ---
    elif name == "vit":
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif name == "tokenmixer":
        model = timm.create_model(
            "mixer_b16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif name == "convmixer":
        # Use an available ConvMixer model name from timm
        model = timm.create_model(
            "convmixer_768_32",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


# -------------------------------------------------
# 2. One training / evaluation epoch
# -------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        (epoch_loss, epoch_accuracy)
    """
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
        preds = outputs.argmax(dim=1)

        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += (preds == labels).sum().item()
        total_samples += batch_size

    return running_loss / total_samples, running_corrects / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the test set.

    Returns:
        (epoch_loss, epoch_accuracy)
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
        preds = outputs.argmax(dim=1)

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += (preds == labels).sum().item()
        total_samples += batch_size

    return running_loss / total_samples, running_corrects / total_samples


# -------------------------------------------------
# 3. Main training loop
# -------------------------------------------------
def train_benchmark(
    model_name: str,
    data_root: str,
    output_dir: str,
    num_classes: int = 2,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    pretrained: bool = True,
) -> Dict[str, list]:
    """
    Train a baseline backbone and save the best checkpoint by test accuracy.

    Args:
        model_name: backbone name
        data_root: dataset root, e.g. "data/BreaKHis 400X"
        output_dir: checkpoint folder (default: benchmark/model_result)
        num_classes: number of output classes
        num_epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
        weight_decay: weight decay for Adam
        pretrained: whether to use ImageNet pretrained weights

    Returns:
        history dict with loss/accuracy curves
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training {model_name} on {device}")

    # Data
    train_loader, test_loader = build_dataloaders(
        root=data_root,
        batch_size=batch_size,
        num_workers=0,
    )

    # Model
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"{model_name}_best.pth")

    for epoch in range(1, num_epochs + 1):
        since = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}  ({model_name})")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - since
        print(
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # Save best checkpoint by test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, ckpt_path)
            print(f"[INFO] New best test acc: {best_acc:.4f}, saved to {ckpt_path}")

    print(f"\n[INFO] Training finished. Best test acc ({model_name}): {best_acc:.4f}")
    return history


# -------------------------------------------------
# 4. CLI entry point
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline backbones on BreaKHis 400X (shared augmentation, no custom normalization)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "densenet121", "vit", "tokenmixer", "convmixer"],
        help="Backbone model to train.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/BreaKHis 400X",
        help="Dataset root. Must contain train/ and test/ folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark/model_result",
        help="Directory to save best checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")

    args = parser.parse_args()

    train_benchmark(
        model_name=args.model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_classes=2,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
    )