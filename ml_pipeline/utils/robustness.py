import random
from pathlib import Path

import torch

from ml_pipeline.utils.data import intents_to_string


def introduce_typo(text: str, rng: random.Random) -> str:
    if len(text) < 3:
        return text
    chars = list(text)
    operation = rng.choice(["swap", "delete"])
    index = rng.randint(0, len(chars) - 2)
    if operation == "swap" and index + 1 < len(chars):
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
    elif operation == "delete":
        del chars[index]
    return "".join(chars)


def predict_text(
    model,
    tokenizer,
    text: str,
    threshold: float,
    max_length: int,
    id2label: dict[int, str],
    device: torch.device,
) -> tuple[str, float]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
    probabilities = torch.sigmoid(logits)[0].tolist()
    predicted_labels = [
        id2label[class_idx] for class_idx, score in enumerate(probabilities) if score >= threshold
    ]
    intent_string = intents_to_string(predicted_labels) if predicted_labels else "unknown intent"
    confidence = max(probabilities) if probabilities else 0.0
    return intent_string, round(confidence, 4)


def build_robustness_report(
    model,
    tokenizer,
    texts: list[str],
    threshold: float,
    max_length: int,
    id2label: dict[int, str],
    device: torch.device,
    num_samples: int,
    seed: int,
) -> str:
    rng = random.Random(seed)
    sample_texts = texts[:]
    rng.shuffle(sample_texts)
    sample_texts = sample_texts[:num_samples]

    lines = ["Robustness report", "=" * 60, ""]
    for original_text in sample_texts:
        typo_text = introduce_typo(original_text, rng)
        original_pred, original_prob = predict_text(
            model, tokenizer, original_text, threshold, max_length, id2label, device
        )
        typo_pred, typo_prob = predict_text(
            model, tokenizer, typo_text, threshold, max_length, id2label, device
        )
        lines.extend(
            [
                f'ORIGINAL: "{original_text}" -> {original_pred} ({original_prob})',
                f'TYPO:     "{typo_text}" -> {typo_pred} ({typo_prob})',
                "",
            ]
        )
    return "\n".join(lines)


def save_robustness_report(report: str, output_path: str | Path) -> None:
    Path(output_path).write_text(report, encoding="utf-8")
