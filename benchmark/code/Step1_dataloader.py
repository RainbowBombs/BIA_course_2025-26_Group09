"""
Step1_dataloader.py

Data loading utilities for the BreaKHis 400X dataset WITHOUT any dataset-specific
preprocessing (i.e., no custom normalization). This module provides:

  - ChestTumorDataset: a simple binary classification dataset
  - train_transform: augmentation used for training (resize + basic flips/rotation)
  - test_transform: deterministic transform for evaluation (resize only)
  - build_dataloaders: helper to create train / test DataLoaders

Expected dataset structure:
  data/BreaKHis 400X/
    train/
      benign/
      malignant/
    test/
      benign/
      malignant/
"""

import os
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class ChestTumorDataset(Dataset):
    """
    A minimal dataset for binary breast tumor classification.

    Directory structure:
      root_dir/
        benign/
          *.png / *.jpg / ...
        malignant/
          *.png / *.jpg / ...

    Class mapping (fixed):
      0 -> benign
      1 -> malignant
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Path to the folder containing class subfolders.
            transform: torchvision transforms applied to each image.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.class_names: List[str] = ["benign", "malignant"]
        self.samples: List[Tuple[str, int]] = []

        for label_idx, cls_name in enumerate(self.class_names):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(ALLOWED_EXTENSIONS):
                    continue
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in: {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]

        # Load image as RGB
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# -------------------------------------------------
# Transforms (NO dataset-specific normalization)
# -------------------------------------------------
train_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),
        T.ToTensor(),  # keep raw [0, 1] range; no normalization here
    ]
)

test_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
    ]
)


def build_dataloaders(
    root: str,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and test dataloaders.

    Args:
        root: Path to dataset root, e.g. "data/BreaKHis 400X".
              Must contain: root/train and root/test.
        batch_size: Mini-batch size.
        num_workers: DataLoader workers (0 is safest on macOS).

    Returns:
        (train_loader, test_loader)
    """
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    train_dataset = ChestTumorDataset(train_dir, transform=train_transform)
    test_dataset = ChestTumorDataset(test_dir, transform=test_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader