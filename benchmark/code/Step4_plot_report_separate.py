import os
from typing import List, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =========================================================
# Repo-relative paths
# =========================================================
DATA_ROOT = os.path.join("data", "BreaKHis 400X")  # contains train/ and test/
WEIGHTS_DIR = os.path.join("benchmark", "model_result")
OUT_DIR = os.path.join("benchmark", "figure", "report")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["benign", "malignant"]  # fixed label order: 0 -> benign, 1 -> malignant


# =========================================================
# Dataset (resize + tensor only)
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
# Plot classification report as a table image
# =========================================================
def save_report_image(report_dict: dict, title: str, save_path: str) -> None:
    rows = ["benign", "malignant", "accuracy", "macro avg", "weighted avg"]
    cols = ["precision", "recall", "f1-score", "support"]

    table_data = []
    for r in rows:
        if r == "accuracy":
            acc = float(report_dict.get("accuracy", 0.0))
            support_total = float(report_dict.get("macro avg", {}).get("support", 0.0))
            table_data.append(["", "", f"{acc:.4f}", f"{support_total:.0f}"])
        else:
            item = report_dict.get(r, {})
            table_data.append(
                [
                    f"{float(item.get('precision', 0.0)):.4f}",
                    f"{float(item.get('recall', 0.0)):.4f}",
                    f"{float(item.get('f1-score', 0.0)):.4f}",
                    f"{float(item.get('support', 0.0)):.0f}",
                ]
            )

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=18)

    tbl = ax.table(
        cellText=table_data,
        rowLabels=rows,
        colLabels=cols,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.8)

    for (_, _), cell in tbl.get_celld().items():
        cell.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# Baselines (exclude BAN)
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

        report = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            digits=4,
            zero_division=0,
        )

        title = f"Classification Report - {display_name} (BreaKHis Test)"
        save_path = os.path.join(OUT_DIR, f"report_{arch}.png")
        save_report_image(report, title, save_path)
        print(f"[OK] Saved: {save_path}")

    print("[DONE] All report images saved to:", OUT_DIR)


if __name__ == "__main__":
    main()