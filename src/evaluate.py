import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
)


def ensure_dirs() -> tuple[Path, Path]:
    figures_dir = Path("reports/figures")
    metrics_dir = Path("reports/metrics")
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, metrics_dir


def save_metrics(metrics: dict, filename: str) -> None:
    _, metrics_dir = ensure_dirs()
    out_path = metrics_dir / filename
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    figures_dir, _ = ensure_dirs()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200)
    plt.close()


def plot_roc_curve(y_true, probs, title: str, filename: str) -> float:
    figures_dir, _ = ensure_dirs()

    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200)
    plt.close()

    return float(auc)


def plot_pr_curve(y_true, probs, title: str, filename: str) -> float:
    figures_dir, _ = ensure_dirs()

    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200)
    plt.close()

    return float(ap)


def plot_accuracy_vs_threshold(y_true, probs, title: str, filename: str) -> None:
    figures_dir, _ = ensure_dirs()

    thresholds = np.linspace(0.0, 1.0, 101)
    acc = [accuracy_score(y_true, probs >= t) for t in thresholds]

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, acc)
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200)
    plt.close()


def compute_basic_metrics(y_true, probs, threshold: float = 0.5) -> dict:
    y_pred = (np.array(probs) >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }