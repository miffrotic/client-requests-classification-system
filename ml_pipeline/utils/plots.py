from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    matrix = multilabel_confusion_matrix(y_true, y_pred)
    num_labels = len(class_names)
    fig, axes = plt.subplots(
        nrows=max(1, (num_labels + 2) // 3),
        ncols=min(3, num_labels),
        figsize=(4 * min(3, num_labels), 3 * max(1, (num_labels + 2) // 3)),
    )
    axes = np.array(axes).reshape(-1)
    for idx, class_name in enumerate(class_names):
        sns.heatmap(
            matrix[idx],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["pred_0", "pred_1"],
            yticklabels=["true_0", "true_1"],
        )
        axes[idx].set_title(class_name)
    for idx in range(num_labels, len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    output_path: str | Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="train_loss")
    ax.plot(epochs, val_losses, marker="o", label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
