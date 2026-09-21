from typing import Any

from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(models: dict[str, Any], x_test, y_test) -> dict[str, dict[str, float]]:
    """Calcula métricas básicas para cada modelo treinado."""
    metrics = {}

    for name, model in models.items():
        predictions = model.predict(x_test)
        model_metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        metrics[name] = model_metrics
        logger.info(f"{name}: {model_metrics}")

    return metrics