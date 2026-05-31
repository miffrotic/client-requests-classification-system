import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: float | None = None,
) -> dict[str, float]:
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def logits_to_predictions(logits: torch.Tensor, threshold: float) -> np.ndarray:
    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).int().cpu().numpy()


def compute_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.binary_cross_entropy_with_logits(logits, labels).item())


def compute_sample_losses(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    probabilities = torch.sigmoid(logits)
    eps = 1e-7
    losses = -(
        labels * torch.log(probabilities + eps) + (1 - labels) * torch.log(1 - probabilities + eps)
    )
    return losses.mean(dim=1).cpu().numpy()
