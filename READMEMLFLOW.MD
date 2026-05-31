# ML Pipeline: DistilBERT + MLflow + Hydra

Пошаговая инструкция по запуску ML-пайплайна для чекпойнта.

## 1. Установка зависимостей

Из корня репозитория:

```bash
poetry install
```

Все зависимости ML-пайплайна (Hydra, MLflow, boto3, matplotlib, seaborn, pyarrow) уже указаны в [`pyproject.toml`](pyproject.toml) вместе с `torch`, `transformers`, `scikit-learn` и `pandas`.

## 2. Подготовка данных

Положите полный датасет локально (файл не коммитится в git):

- `augmented_customer_support_dataset.parquet`, или
- `customer_support_dataset_generated.csv`

Путь по умолчанию указан в `ml_pipeline/conf/config.yaml` (`data.source_path`).

Сгенерируйте train/val/test split (70/15/15, `random_state=42`):

```bash
python ml_pipeline/prepare_data.py
```

При необходимости переопределите путь к исходному файлу:

```bash
python ml_pipeline/prepare_data.py data.source_path=../customer_support_dataset_generated.csv
```

Результат сохраняется в `ml_pipeline/data/`:

- `train.csv`
- `val.csv`
- `test.csv`

## 3. Поднятие инфраструктуры MLflow + MinIO + PostgreSQL

```bash
docker compose -f docker-compose.mlflow.yml up -d
```

Проверка статуса:

```bash
docker compose -f docker-compose.mlflow.yml ps
```

### Порты и доступы

| Сервис | URL | Логин | Пароль |
|--------|-----|-------|--------|
| MLflow UI | http://localhost:5000 | — | — |
| MinIO API (S3) | http://localhost:9000 | `minioadmin` | `minioadmin` |
| MinIO Console | http://localhost:9001 | `minioadmin` | `minioadmin` |
| PostgreSQL | localhost:5433 | `mlflow` | `mlflow` |

Bucket для артефактов создаётся автоматически: `mlflow`.

## 4. Переменные окружения для клиента (хост)

Перед запуском экспериментов задайте переменные (PowerShell):

```powershell
$env:MLFLOW_S3_ENDPOINT_URL = "http://127.0.0.1:9000"
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin"
```

Linux/macOS:

```bash
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

Эти значения также прописаны в `ml_pipeline/conf/config.yaml` и применяются скриптами автоматически.

## 5. Запуск обучения и логирования в MLflow

```bash
python ml_pipeline/dl_experiments.py
```

Быстрый smoke-test (1 эпоха):

```bash
python ml_pipeline/dl_experiments.py training.epochs=1
```

Скрипт выполняет:

1. Фиксацию seed.
2. Загрузку split-файлов и токенизацию.
3. Baseline (TF-IDF + DummyClassifier) с метриками `baseline_*`.
4. Обучение DistilBERT (`distilbert-base-uncased`).
5. Логирование в MLflow:
   - все параметры из `config.yaml`;
   - метрики: `micro_f1`, `macro_f1`, `weighted_f1`, `loss`;
   - артефакты в S3 (MinIO): `confusion_matrix.png`, `learning_curves.png`, `error_analysis.csv`, `robustness_report.txt`;
   - модель (`model/`).
6. Автоматическую установку тега `PRD=Production` и `version=vN` для лучшего run по `weighted_f1`.

## 6. Просмотр результатов в MLflow

1. Откройте http://localhost:5000
2. Выберите эксперимент `distilbert-intent-classification`
3. Откройте нужный run:
   - **Parameters** — гиперпараметры и конфиг
   - **Metrics** — F1 и loss
   - **Artifacts** — графики, CSV, модель
   - **Tags** — `PRD=Production`, `version=v1` для лучшего прогона

Артефакты также доступны в MinIO Console (http://localhost:9001) в bucket `mlflow`.

## 7. Демонстрация inference по PRD-модели

```bash
python ml_pipeline/dl_demonstration.py demo.text="I need a refund for my order"
```

Или с дефолтным текстом из config:

```bash
python ml_pipeline/dl_demonstration.py
```

Скрипт:

1. Не обучает модель.
2. Подключается к MLflow.
3. Находит run с тегом `PRD=Production`.
4. Загружает модель через `mlflow.transformers.load_model`.
5. Выводит предсказанные интенты и top-5 вероятностей.

## 8. Остановка инфраструктуры

```bash
docker compose -f docker-compose.mlflow.yml down
```

Для полного удаления данных добавьте `-v`:

```bash
docker compose -f docker-compose.mlflow.yml down -v
```

## Структура ML-пайплайна

```
ml_pipeline/
  conf/config.yaml
  prepare_data.py
  dl_experiments.py
  dl_demonstration.py
  utils/
  data/          # split-файлы (gitignored)
  outputs/       # временные артефакты обучения (gitignored)
```

## Примечания

- Существующий код API, бота и inference (`apps/`, `bot.py`) **не изменяется**.
- Обученная модель сохраняется в MLflow/MinIO; копирование в `safetensors_models/` для production API — отдельный deploy-шаг.
- Порог классификации: `0.25` (как в `apps/dl/classifier.py`).
- Обучение использует `max_length=64`; demo/inference — `max_length=512`.
