from loguru import logger
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)

SCORING = ["roc_auc", "f1", "precision", "recall"]


def _build_models() -> dict[str, Pipeline]:
    return {
        "gradient_boosting": create_gradient_boosting_pipeline(),
        "xgboost": create_xgboost_pipeline(),
        "lightgbm": create_lightgbm_pipeline(),
    }


def train_models(x_train, y_train) -> dict[str, Pipeline]:
    """Treina os modelos configurados e retorna os pipelines ajustados."""
    models = {}

    for name, model in _build_models().items():
        logger.info(f"Treinando modelo {name}...")
        model.fit(x_train, y_train)
        models[name] = model

    return models


def cross_validate_models(
    x: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """Avalia os modelos com folds estratificados e retorna médias e desvios."""
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = {}

    for name, model in _build_models().items():
        scores = cross_validate(model, x, y, cv=splitter, scoring=SCORING)
        summary = {
            metric: {
                "mean": scores[f"test_{metric}"].mean(),
                "std": scores[f"test_{metric}"].std(),
            }
            for metric in SCORING
        }
        results[name] = summary
        logger.info(
            f"[CV] {name}: "
            + ", ".join(
                f"{metric}={values['mean']:.3f}+/-{values['std']:.3f}"
                for metric, values in summary.items()
            )
        )

    return results
