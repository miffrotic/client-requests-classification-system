from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

from ml_pipeline.utils.metrics import compute_multilabel_metrics


def train_baseline(texts_train: list[str], labels_train: list[list[str]], mlb) -> tuple[object, object]:
    vectorizer = TfidfVectorizer(max_features=10000)
    x_train = vectorizer.fit_transform(texts_train)
    y_train = mlb.transform(labels_train)
    classifier = MultiOutputClassifier(DummyClassifier(strategy="most_frequent"))
    classifier.fit(x_train, y_train)
    return vectorizer, classifier


def evaluate_baseline(
    vectorizer,
    classifier,
    texts: list[str],
    labels: list[list[str]],
    mlb,
) -> dict[str, float]:
    x_data = vectorizer.transform(texts)
    y_true = mlb.transform(labels)
    y_pred = classifier.predict(x_data)
    return compute_multilabel_metrics(y_true, y_pred, loss=None)
