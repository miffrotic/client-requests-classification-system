import sys
import tempfile
from pathlib import Path

ML_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(ML_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT.parent))

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from ml_pipeline.utils.baseline import evaluate_baseline, train_baseline
from ml_pipeline.utils.data import (
    create_dataloader,
    fit_multilabel_binarizer,
    load_xy_split_csvs,
)
from ml_pipeline.utils.error_analysis import build_error_analysis, save_error_analysis
from ml_pipeline.utils.metrics import compute_bce_loss, compute_multilabel_metrics, logits_to_predictions
from ml_pipeline.utils.mlflow_utils import configure_mlflow_s3, log_config_params, tag_best_run_as_prd
from ml_pipeline.utils.plots import save_confusion_matrix, save_learning_curves
from ml_pipeline.utils.robustness import build_robustness_report, save_robustness_report
from ml_pipeline.utils.seed import set_seed


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (ML_PIPELINE_ROOT / path_obj).resolve()


def build_compute_metrics(threshold: float):
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits_to_predictions(torch.tensor(logits), threshold)
        return compute_multilabel_metrics(labels, predictions)

    return compute_metrics


def evaluate_model_on_loader(model, dataloader, device, threshold: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(batch["labels"].cpu())
            total_loss += compute_bce_loss(outputs.logits, batch["labels"])
            num_batches += 1

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    predictions = logits_to_predictions(logits_tensor, threshold)
    metrics = compute_multilabel_metrics(
        labels_tensor.numpy(),
        predictions,
        loss=total_loss / max(num_batches, 1),
    )
    return metrics, labels_tensor.numpy(), predictions


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    configure_mlflow_s3(cfg)

    # Resolve and validate all six X/y paths before doing any heavy work.
    split_paths = {
        "x_train": resolve_path(cfg.data.x_train),
        "y_train": resolve_path(cfg.data.y_train),
        "x_val":   resolve_path(cfg.data.x_val),
        "y_val":   resolve_path(cfg.data.y_val),
        "x_test":  resolve_path(cfg.data.x_test),
        "y_test":  resolve_path(cfg.data.y_test),
    }
    for key, path in split_paths.items():
        if not path.exists():
            msg = f"Split file not found ({key}): {path}"
            raise FileNotFoundError(msg)

    texts_train, labels_train = load_xy_split_csvs(
        split_paths["x_train"], split_paths["y_train"],
        cfg.data.text_column, cfg.data.label_column,
    )
    texts_val, labels_val = load_xy_split_csvs(
        split_paths["x_val"], split_paths["y_val"],
        cfg.data.text_column, cfg.data.label_column,
    )
    texts_test, labels_test = load_xy_split_csvs(
        split_paths["x_test"], split_paths["y_test"],
        cfg.data.text_column, cfg.data.label_column,
    )

    mlb = fit_multilabel_binarizer(labels_train)
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.model.model_name)

    train_loader = create_dataloader(
        texts_train,
        labels_train,
        tokenizer,
        mlb,
        cfg.model.max_length,
        cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        texts_val,
        labels_val,
        tokenizer,
        mlb,
        cfg.model.max_length,
        cfg.training.batch_size,
        shuffle=False,
    )
    test_loader = create_dataloader(
        texts_test,
        labels_test,
        tokenizer,
        mlb,
        cfg.model.max_length,
        cfg.training.batch_size,
        shuffle=False,
    )

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        log_config_params(cfg)

        vectorizer, baseline_model = train_baseline(texts_train, labels_train, mlb)
        baseline_metrics = evaluate_baseline(
            vectorizer,
            baseline_model,
            texts_test,
            labels_test,
            mlb,
        )
        mlflow.log_metrics({f"baseline_{key}": value for key, value in baseline_metrics.items()})

        id2label = {index: label for index, label in enumerate(mlb.classes_)}
        label2id = {label: index for index, label in id2label.items()}
        model = DistilBertForSequenceClassification.from_pretrained(
            cfg.model.model_name,
            num_labels=len(mlb.classes_),
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id,
        )

        output_dir = resolve_path("outputs/training")
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=cfg.training.epochs,
            per_device_train_batch_size=cfg.training.batch_size,
            per_device_eval_batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_ratio=cfg.training.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            report_to="none",
            load_best_model_at_end=False,
            # GPU / mixed-precision settings
            fp16=cfg.training.fp16,
            bf16=cfg.training.bf16,
            dataloader_num_workers=cfg.training.dataloader_num_workers,
            dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            compute_metrics=build_compute_metrics(cfg.model.threshold),
        )
        trainer.train()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_metrics, y_true, y_pred = evaluate_model_on_loader(
            model,
            test_loader,
            device,
            cfg.model.threshold,
        )
        mlflow.log_metrics(test_metrics)

        train_losses = [
            entry["loss"]
            for entry in trainer.state.log_history
            if "loss" in entry and "eval_loss" not in entry
        ]
        val_losses = [entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            confusion_path = tmp_path / "confusion_matrix.png"
            curves_path = tmp_path / "learning_curves.png"
            error_path = tmp_path / "error_analysis.csv"
            robustness_path = tmp_path / "robustness_report.txt"

            save_confusion_matrix(y_true, y_pred, list(mlb.classes_), confusion_path)
            save_learning_curves(train_losses, val_losses, curves_path)

            error_df = build_error_analysis(
                model,
                test_loader,
                texts_test,
                labels_test,
                mlb,
                cfg.model.threshold,
                cfg.error_analysis.top_k,
                device,
            )
            save_error_analysis(error_df, error_path)

            robustness_report = build_robustness_report(
                model,
                tokenizer,
                texts_test,
                cfg.model.threshold,
                cfg.model.inference_max_length,
                id2label,
                device,
                cfg.robustness.num_samples,
                cfg.seed,
            )
            save_robustness_report(robustness_report, robustness_path)

            mlflow.log_artifact(str(confusion_path))
            mlflow.log_artifact(str(curves_path))
            mlflow.log_artifact(str(error_path))
            mlflow.log_artifact(str(robustness_path))

            # Save model + tokenizer in HuggingFace format so that
            # DistilBertForSequenceClassification.from_pretrained() and
            # DistilBertTokenizerFast.from_pretrained() work directly on
            # the downloaded artifact path in dl_demonstration.py.
            hf_model_dir = tmp_path / "hf_model"
            hf_model_dir.mkdir()
            model.save_pretrained(hf_model_dir)
            tokenizer.save_pretrained(hf_model_dir)
            mlflow.log_artifacts(str(hf_model_dir), artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        tag_best_run_as_prd(cfg.mlflow.experiment_name, run_id)

        print(f"MLflow run id: {run_id}")
        print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
