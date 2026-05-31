import sys
from pathlib import Path

ML_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(ML_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT.parent))

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from ml_pipeline.utils.data import intents_to_string, parse_intent_value


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (ML_PIPELINE_ROOT / path_obj).resolve()


def load_source_dataset(source_path: Path) -> pd.DataFrame:
    if source_path.suffix.lower() == ".parquet":
        dataframe = pd.read_parquet(source_path)
    else:
        dataframe = pd.read_csv(source_path)

    required_columns = {"instruction", "intent"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        msg = f"Dataset must contain columns: {sorted(required_columns)}. Missing: {sorted(missing)}"
        raise ValueError(msg)

    dataframe = dataframe.copy()
    dataframe["instruction"] = dataframe["instruction"].fillna("").astype(str)
    dataframe["intent"] = dataframe["intent"].apply(
        lambda value: intents_to_string(parse_intent_value(value))
    )
    return dataframe[["instruction", "intent"]]


def save_split(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    source_path = resolve_path(cfg.data.source_path)
    if not source_path.exists():
        msg = f"Source dataset not found: {source_path}"
        raise FileNotFoundError(msg)

    dataframe = load_source_dataset(source_path)
    train_data_path = resolve_path(cfg.data.train_data)
    val_data_path = resolve_path(cfg.data.val_data)
    test_data_path = resolve_path(cfg.data.test_data)

    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=cfg.data.split.test_size,
        random_state=cfg.data.split.random_state,
    )
    relative_val_size = cfg.data.split.val_size / (1 - cfg.data.split.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=cfg.data.split.random_state,
    )

    save_split(train_df, train_data_path)
    save_split(val_df, val_data_path)
    save_split(test_df, test_data_path)

    print(f"Source dataset: {source_path}")
    print(f"Total rows: {len(dataframe)}")
    print(f"Train rows: {len(train_df)} -> {train_data_path}")
    print(f"Val rows: {len(val_df)} -> {val_data_path}")
    print(f"Test rows: {len(test_df)} -> {test_data_path}")


if __name__ == "__main__":
    main()
