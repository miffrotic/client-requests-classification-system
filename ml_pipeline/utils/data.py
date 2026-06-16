import ast
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast


def parse_intent_value(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            pass
    if ", " in text:
        return [part.strip() for part in text.split(", ") if part.strip()]
    return [text]


def intents_to_string(intents: list[str]) -> str:
    return ", ".join(sorted(set(intents)))


def load_split_csv(path: str | Path, text_column: str, label_column: str) -> tuple[list[str], list[list[str]]]:
    df = pd.read_csv(path)
    texts = df[text_column].fillna("").astype(str).tolist()
    labels = [parse_intent_value(value) for value in df[label_column].tolist()]
    return texts, labels


def load_xy_split_csvs(
    x_path: str | Path,
    y_path: str | Path,
    text_column: str,
    label_column: str,
) -> tuple[list[str], list[list[str]]]:
    """Load texts and labels from separate X and y CSV files.

    Supports the pre-existing ``ml_base/data_split/`` layout where features
    and labels are stored in separate files (e.g. ``X_train.csv`` /
    ``y_train.csv``).

    Args:
        x_path:       Path to the features CSV (must contain ``text_column``).
        y_path:       Path to the labels CSV (must contain ``label_column``).
        text_column:  Name of the text column in the X file.
        label_column: Name of the label column in the y file.

    Returns:
        Tuple of (texts, labels) ready for ``IntentDataset``.
    """
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if len(x_df) != len(y_df):
        msg = (
            f"Row count mismatch: {x_path} has {len(x_df)} rows "
            f"but {y_path} has {len(y_df)} rows."
        )
        raise ValueError(msg)

    texts = x_df[text_column].fillna("").astype(str).tolist()
    labels = [parse_intent_value(value) for value in y_df[label_column].tolist()]
    return texts, labels


def fit_multilabel_binarizer(train_labels: list[list[str]]) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)
    return mlb


class IntentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[list[str]],
        tokenizer: DistilBertTokenizerFast,
        mlb: MultiLabelBinarizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.targets = mlb.transform(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.targets[index], dtype=torch.float)
        return item


def create_dataloader(
    texts: list[str],
    labels: list[list[str]],
    tokenizer: DistilBertTokenizerFast,
    mlb: MultiLabelBinarizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = IntentDataset(texts, labels, tokenizer, mlb, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
