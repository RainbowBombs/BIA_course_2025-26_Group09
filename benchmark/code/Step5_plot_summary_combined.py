"""
Step5_plot_summary_combined.py

Generate combined figures (ROC curve + bar charts) for:
  - BAN (BreaKHis Analysis Network): ResNet18 trained with BAN preprocessing
  - 6 baselines: ResNet18/34, DenseNet121, ViT, TokenMixer, ConvMixer

Evaluation policy:
  - BAN evaluation uses BAN test transform:
      Resize(700, 460) + ToTensor + Normalize(mean/std)
  - Baselines evaluation uses:
      Resize(224, 224) + ToTensor

Inputs (GitHub paths):
  - Dataset root: data/BreaKHis 400X (expects train/ and test/)
  - BAN weight: model/model result/BreaKHis Analysis Network_best.pth
  - Baseline weights: benchmark/model_result/*.pth

Outputs:
  benchmark/figure/summary/
    - roc_ban_vs_6_baselines.png
    - bar_accuracy_ban_vs_6_baselines.png
    - bar_auc_ban_vs_6_baselines.png
    - bar_f1_malignant_ban_vs_6_baselines.png
"""

import os
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from torchvision import models, transforms
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score


# -----------------------------
# GitHub paths
# -----------------------------
DATA_ROOT = os.path.join("data", "BreaKHis 400X")
TEST_DIR = os.path.join(DATA_ROOT, "test")

BAN_NAME = "BreaKHis Analysis Network (BAN)"
BAN_WEIGHT = os.path.join("model", "model result", "BreaKHis Analysis Network_best.pth")

BASELINE_DIR = os.path.join("benchmark", "model_result")
BASELINE_WEIGHTS = {
    "ResNet18": os.path.join(BASELINE_DIR, "resnet18_best.pth"),
    "ResNet34": os.path.join(BASELINE_DIR, "resnet34_best.pth"),
    "DenseNet121": os.path.join(BASELINE_DIR, "densenet121_best.pth"),
    "ViT": os.path.join(BASELINE_DIR, "vit_best.pth"),
    "TokenMixer": os.path.join(BASELINE_DIR, "tokenmixer_best.pth"),
    "ConvMixer": os.path.join(BASELINE_DIR, "convmixer_best.pth"),
}

OUT_DIR = os.path.join("benchmark", "figure", "summary")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Labels
# -----------------------------
CLASS_NAMES = ["benign", "malignant"]  # 0 -> benign, 1 -> malignant
POS_LABEL = 1  # malignant


# -----------------------------
# BAN normalization (must match BAN training)
# -----------------------------
BAN_MEAN = [0.755842387676239, 0.5889061689376831, 0.7419362664222717]
BAN_STD = [0.14278094470500946, 0.20091958343982697, 0.1162722110748291]


# -----------------------------
# Dataset
# -----------------------------
class BreakHisTestDataset(Dataset):
    """
    Test dataset loader:
      <root>/benign/*
      <root>/malignant/*
    """

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for label_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# -----------------------------
# Transforms
# -----------------------------
baseline_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

ban_test_transform = transforms.Compose(
    [
        transforms.Resize((700, 460)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BAN_MEAN, std=BAN_STD),
    ]
)


# -----------------------------
# Model builders (must match training)
# -----------------------------
def build_baseline_model(name: str, num_classes: int = 2) -> nn.Module:
    n = name.lower()

    if n == "resnet18":
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if n == "resnet34":
        m = models.resnet34(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if n == "densenet121":
        m = models.densenet121(pretrained=False)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    if n in ["vit", "tokenmixer", "convmixer"]:
        import timm

        if n == "vit":
            timm_name = "vit_small_patch16_224"
        elif n == "tokenmixer":
            timm_name = "mixer_b16_224"
        else:
            timm_name = "convmixer_768_32"

        m = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        return m

    raise ValueError(f"Unknown baseline name: {name}")


def build_ban_model(num_classes: int = 2) -> nn.Module:
    """
    BAN uses a ResNet18 backbone (team-defined method).
    """
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# -----------------------------
# Evaluation helpers
# -----------------------------
@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: (N,)
      y_prob: (N,) probability of malignant
    """
    model.eval()
    y_true_list: List[int] = []
    y_prob_list: List[float] = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels_np = labels.numpy()

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, POS_LABEL].detach().cpu().numpy()

        y_true_list.extend(labels_np.tolist())
        y_prob_list.extend(probs.tolist())

    return np.array(y_true_list), np.array(y_prob_list)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1_malig = f1_score(y_true, y_pred, pos_label=POS_LABEL)
    return {"acc": float(acc), "auc": float(auc), "f1_malignant": float(f1_malig)}


# -----------------------------
# Plotting (consistent color mapping)
# -----------------------------
def get_color_map() -> Dict[str, str]:
    """
    Fixed colors to keep consistent across ROC and bar charts.
    """
    return {
        BAN_NAME: "#d62728",       # red
        "ResNet18": "#1f77b4",      # blue
        "ResNet34": "#ff7f0e",      # orange
        "DenseNet121": "#2ca02c",   # green
        "ViT": "#9467bd",          # purple
        "TokenMixer": "#8c564b",   # brown
        "ConvMixer": "#17becf",    # cyan
    }


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: str,
    colors: Dict[str, str],
):
    plt.figure(figsize=(11, 8))

    for name, (fpr, tpr, auc) in roc_data.items():
        lw = 5 if name == BAN_NAME else 2.5
        plt.plot(
            fpr,
            tpr,
            linewidth=lw,
            color=colors.get(name, None),
            label=f"{name} (AUC={auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, color="gray")
    plt.title("ROC Curves - BAN vs 6 baselines (BreaKHis Test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_bar_metric(
    names: List[str],
    values: List[float],
    metric_title: str,
    ylabel: str,
    save_path: str,
    colors: Dict[str, str],
):
    plt.figure(figsize=(12, 6))

    bar_colors = [colors.get(n, None) for n in names]
    bars = plt.bar(names, values, color=bar_colors)

    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(metric_title)
    plt.ylim(0.0, 1.0)

    for i, n in enumerate(names):
        if n == BAN_NAME:
            bars[i].set_linewidth(3.0)
            bars[i].set_edgecolor("black")
        else:
            bars[i].set_linewidth(1.0)
            bars[i].set_edgecolor("black")

        v = values[i]
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=11)

    # Legend
    legend_handles = [Patch(facecolor=colors[n], edgecolor="black", label=n) for n in names]
    plt.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    # Sanity checks
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Missing test directory: {TEST_DIR}")

    if not os.path.isfile(BAN_WEIGHT):
        raise FileNotFoundError(f"Missing BAN weight: {BAN_WEIGHT}")

    for k, p in BASELINE_WEIGHTS.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing baseline weight ({k}): {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    colors = get_color_map()

    # 1) BAN evaluation with BAN transform
    ban_dataset = BreakHisTestDataset(TEST_DIR, transform=ban_test_transform)
    ban_loader = DataLoader(ban_dataset, batch_size=16, shuffle=False, num_workers=0)

    ban_model = build_ban_model(num_classes=2)
    ban_state = torch.load(BAN_WEIGHT, map_location="cpu")
    ban_model.load_state_dict(ban_state)
    ban_model.to(device)

    y_true_ban, y_prob_ban = eval_model(ban_model, ban_loader, device)
    ban_metrics = compute_metrics(y_true_ban, y_prob_ban)
    print("[INFO] BAN metrics:", ban_metrics)

    # 2) Baselines evaluation with baseline transform
    baseline_dataset = BreakHisTestDataset(TEST_DIR, transform=baseline_test_transform)
    baseline_loader = DataLoader(baseline_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_metrics: Dict[str, Dict[str, float]] = {BAN_NAME: ban_metrics}
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    fpr, tpr, _ = roc_curve(y_true_ban, y_prob_ban, pos_label=POS_LABEL)
    roc_data[BAN_NAME] = (fpr, tpr, ban_metrics["auc"])

    baseline_arch_map = {
        "ResNet18": "resnet18",
        "ResNet34": "resnet34",
        "DenseNet121": "densenet121",
        "ViT": "vit",
        "TokenMixer": "tokenmixer",
        "ConvMixer": "convmixer",
    }

    for display_name, arch in baseline_arch_map.items():
        weight_path = BASELINE_WEIGHTS[display_name]

        model = build_baseline_model(arch, num_classes=2)
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)

        y_true, y_prob = eval_model(model, baseline_loader, device)
        metrics = compute_metrics(y_true, y_prob)

        all_metrics[display_name] = metrics
        print(f"[INFO] {display_name} metrics:", metrics)

        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=POS_LABEL)
        roc_data[display_name] = (fpr, tpr, metrics["auc"])

    # 3) ROC curve (7 lines)
    roc_save = os.path.join(OUT_DIR, "roc_ban_vs_6_baselines.png")
    plot_roc_curves(roc_data, roc_save, colors)
    print("[OK] Saved ROC:", roc_save)

    # 4) Bar charts (3 figures), fixed order
    ordered_names = [BAN_NAME] + list(baseline_arch_map.keys())

    acc_vals = [all_metrics[n]["acc"] for n in ordered_names]
    auc_vals = [all_metrics[n]["auc"] for n in ordered_names]
    f1_vals = [all_metrics[n]["f1_malignant"] for n in ordered_names]

    save_acc = os.path.join(OUT_DIR, "bar_accuracy_ban_vs_6_baselines.png")
    plot_bar_metric(
        ordered_names,
        acc_vals,
        metric_title="Accuracy - BAN vs 6 baselines (BreaKHis Test)",
        ylabel="Accuracy",
        save_path=save_acc,
        colors=colors,
    )
    print("[OK] Saved bar (ACC):", save_acc)

    save_auc = os.path.join(OUT_DIR, "bar_auc_ban_vs_6_baselines.png")
    plot_bar_metric(
        ordered_names,
        auc_vals,
        metric_title="ROC AUC - BAN vs 6 baselines (BreaKHis Test)",
        ylabel="ROC AUC",
        save_path=save_auc,
        colors=colors,
    )
    print("[OK] Saved bar (AUC):", save_auc)

    save_f1 = os.path.join(OUT_DIR, "bar_f1_malignant_ban_vs_6_baselines.png")
    plot_bar_metric(
        ordered_names,
        f1_vals,
        metric_title="F1 (malignant) - BAN vs 6 baselines (BreaKHis Test)",
        ylabel="F1 (malignant)",
        save_path=save_f1,
        colors=colors,
    )
    print("[OK] Saved bar (F1 malignant):", save_f1)

    print("[DONE] All figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()