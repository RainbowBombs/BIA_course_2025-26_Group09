import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from torchvision import models, transforms


# =========================================================
# Repo-relative paths
# =========================================================
DATA_ROOT = os.path.join("data", "BreaKHis 400X")  # contains train/ and test/
WEIGHTS_DIR = os.path.join("benchmark", "model_result")
OUT_DIR = os.path.join("benchmark", "figure", "cm")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["benign", "malignant"]  # fixed label order: 0 -> benign, 1 -> malignant


# =========================================================
# Dataset (evaluation: resize + tensor only)
# =========================================================
class BreastDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for label, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

test_dir = os.path.join(DATA_ROOT, "test")
test_dataset = BreastDataset(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


# =========================================================
# Model builders (must match your training architectures)
# =========================================================
def build_model(arch: str, num_classes: int = 2) -> nn.Module:
    arch = arch.lower()

    if arch == "resnet18":
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch == "resnet34":
        m = models.resnet34(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch == "densenet121":
        m = models.densenet121(pretrained=False)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    if arch in ["vit", "tokenmixer", "convmixer"]:
        import timm

        if arch == "vit":
            model_name = "vit_small_patch16_224"
        elif arch == "tokenmixer":
            model_name = "mixer_b16_224"
        else:
            # If loading fails, try: "convmixer_768_32.in1k"
            model_name = "convmixer_768_32"

        m = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        return m

    raise ValueError(f"Unknown arch: {arch}")


# =========================================================
# Confusion matrix plotting
# =========================================================
def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="viridis")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# Baselines (exclude BAN)
# Expected weight files under benchmark/model_result/
# =========================================================
MODELS = [
    ("resnet18", "ResNet18", "resnet18_best.pth"),
    ("resnet34", "ResNet34", "resnet34_best.pth"),
    ("densenet121", "DenseNet121", "densenet121_best.pth"),
    ("vit", "ViT", "vit_best.pth"),
    ("tokenmixer", "TokenMixer", "tokenmixer_best.pth"),
    ("convmixer", "ConvMixer", "convmixer_best.pth"),
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    for arch, display_name, weight_file in MODELS:
        weight_path = os.path.join(WEIGHTS_DIR, weight_file)
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Missing weight: {weight_path}")

        print(f"[INFO] Evaluating {display_name} ...")
        model = build_model(arch, num_classes=2)
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        y_true: List[int] = []
        y_pred: List[int] = []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                y_true.extend(labels.numpy().tolist())
                y_pred.extend(preds)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        save_path = os.path.join(OUT_DIR, f"cm_{arch}.png")
        plot_confusion_matrix(cm, f"Confusion Matrix - {display_name} (BreaKHis)", save_path)
        print(f"[OK] Saved: {save_path}")

    print("[DONE] All confusion matrices saved to:", OUT_DIR)


if __name__ == "__main__":
    main()