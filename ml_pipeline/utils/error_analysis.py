from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml_pipeline.utils.data import intents_to_string
from ml_pipeline.utils.metrics import compute_sample_losses, logits_to_predictions


def build_error_analysis(
    model,
    dataloader: DataLoader,
    texts: list[str],
    labels: list[list[str]],
    mlb,
    threshold: float,
    top_k: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    sample_losses = compute_sample_losses(logits_tensor, labels_tensor)
    probabilities = torch.sigmoid(logits_tensor).numpy()
    predictions = logits_to_predictions(logits_tensor, threshold)

    y_true = mlb.transform(labels)
    error_scores = []
    rows = []

    for idx, text in enumerate(texts):
        true_labels = [mlb.classes_[class_idx] for class_idx, value in enumerate(y_true[idx]) if value == 1]
        pred_labels = [
            mlb.classes_[class_idx] for class_idx, value in enumerate(predictions[idx]) if value == 1
        ]
        false_positives = len(set(pred_labels) - set(true_labels))
        false_negatives = len(set(true_labels) - set(pred_labels))
        error_score = sample_losses[idx] + false_positives + false_negatives
        error_scores.append(error_score)

        pred_indices = [class_idx for class_idx, value in enumerate(predictions[idx]) if value == 1]
        if pred_indices:
            prob = float(max(probabilities[idx][class_idx] for class_idx in pred_indices))
        else:
            prob = float(np.max(probabilities[idx]))

        rows.append(
            {
                "text": text,
                "true_intent": intents_to_string(true_labels) if true_labels else "unknown intent",
                "predicted_intent": intents_to_string(pred_labels) if pred_labels else "unknown intent",
                "prob": round(prob, 4),
                "error_score": error_score,
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values("error_score", ascending=False).head(top_k)
    return dataframe.drop(columns=["error_score"])


def save_error_analysis(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    dataframe.to_csv(output_path, index=False)
