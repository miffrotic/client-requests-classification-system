import sys
import tempfile
from pathlib import Path

ML_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(ML_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT.parent))

import hydra
import mlflow
import torch
from omegaconf import DictConfig
from mlflow.tracking import MlflowClient
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from ml_pipeline.utils.mlflow_utils import configure_mlflow_s3, find_prd_run_id
from ml_pipeline.utils.robustness import predict_text


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_mlflow_s3(cfg)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    run_id = find_prd_run_id(cfg.mlflow.experiment_name)
    if run_id is None:
        msg = (
            f"No run with tag PRD=Production found in experiment "
            f"'{cfg.mlflow.experiment_name}'. Run dl_experiments.py first."
        )
        raise RuntimeError(msg)

    client = MlflowClient()
    run = client.get_run(run_id)
    version = run.data.tags.get("version", "unknown")
    weighted_f1 = run.data.metrics.get("weighted_f1", "n/a")

    text = cfg.demo.text
    print(f"Using PRD model: run_id={run_id}, version={version}, weighted_f1={weighted_f1}")
    print(f"Input text: {text}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=tmp_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = {int(index): label for index, label in model.config.id2label.items()}
    predicted_intent, confidence = predict_text(
        model,
        tokenizer,
        text,
        cfg.model.threshold,
        cfg.model.inference_max_length,
        id2label,
        device,
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.model.inference_max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    probabilities = torch.sigmoid(logits)[0].tolist()
    top_predictions = sorted(
        ((id2label[index], round(score, 4)) for index, score in enumerate(probabilities)),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    print(f"Predicted intents: {predicted_intent}")
    print(f"Confidence (max prob): {confidence}")
    print("Top-5 probabilities:")
    for label, score in top_predictions:
        print(f"  {label}: {score}")


if __name__ == "__main__":
    main()
