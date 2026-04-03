from pathlib import Path

import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from config import BASE_DIR


MODEL_DIR = Path(BASE_DIR / "safetensors_models")
THRESHOLD = 0.25
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

    probabilities = torch.sigmoid(logits)[0].tolist()
     
    prob_dict = {id2label[i]: round(p, 3) for i, p in enumerate(probabilities)}
    top_guesses = {k: v for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True) if v > 0.05}
    print(f"Вероятности для '{text}': {top_guesses}")

    predicted_labels = [
        id2label[class_idx] for class_idx, score in enumerate(probabilities) if score >= threshold
    ]

    # если предсказание необходимо, но ни один класс не пробил порог
    # if not predicted_labels:
    #     max_prob = max(probabilities)
    #     predicted_labels = [
    #         id2label[idx] for idx, score in enumerate(probabilities)
    #         if score >= (max_prob - 0.1) and score > 0.1
    #     ]

    return ", ".join(predicted_labels) if predicted_labels else "unknown intent"