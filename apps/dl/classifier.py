from pathlib import Path

import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from config import BASE_DIR


MODEL_DIR = Path(BASE_DIR / "safetensors_models")
THRESHOLD = 0.5
MAX_LENGTH = 512


tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = {int(idx): label for idx, label in model.config.id2label.items()}


def predict_intents(text: str, threshold: float = THRESHOLD) -> str:
    if not text.strip():
        return "unknown intent"

    encoded_text = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**encoded_text).logits

    probabilities = torch.sigmoid(logits)[0]
    predicted_labels = [
        id2label[class_idx] for class_idx, score in enumerate(probabilities) if score.item() >= threshold
    ]

    return ", ".join(predicted_labels) if predicted_labels else "unknown intent"
