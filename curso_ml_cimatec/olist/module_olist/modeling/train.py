from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import typer

from module_olist.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)
from module_olist.modeling.split import FEATURES, split_data

app = typer.Typer()

CATEGORICAL_FEATURES = ["customer_state"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]

SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
}

CV_METRICS = ["roc_auc", "f1", "precision", "recall"]


def build_model(random_state: int = 42) -> Pipeline:
    """Pré-processa as features (one-hot para UF, mediana para numéricas) e treina uma Random Forest."""
    # sparse_threshold=0 força saída densa: com as 27 UFs o one-hot vira matriz esparsa,
    # e a explicação SHAP (explain.py) monta um DataFrame a partir dela.
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ],
        sparse_threshold=0,
    )
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


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
        scores = cross_validate(model, x, y, cv=splitter, scoring=CV_METRICS)
        # cross_validate devolve um array de `cv` valores por métrica (um por fold).
        # Logamos fold a fold para ver a variação entre eles: um fold muito pior
        # que os outros pode indicar um subconjunto de dados atípico.
        for fold_idx in range(cv):
            fold_metrics = ", ".join(
                f"{metric}={scores[f'test_{metric}'][fold_idx]:.3f}" for metric in CV_METRICS
            )
            logger.debug(f"[CV] {name} | fold {fold_idx + 1}/{cv}: {fold_metrics}")

        summary = {
            metric: {
                "mean": scores[f"test_{metric}"].mean(),
                "std": scores[f"test_{metric}"].std(),
            }
            for metric in CV_METRICS
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


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Validação cruzada estratificada, devolvendo a média das métricas de treino e de teste."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        build_model(random_state),
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
    )
    result = {}
    for metric in SCORING:
        result[f"{metric}_treino"] = scores[f"train_{metric}"].mean()
        result[f"{metric}_teste"] = scores[f"test_{metric}"].mean()
    return result


def _holdout_metrics(model, X_train, y_train, X_test, y_test) -> dict[str, float]:
    """Métricas de treino e de teste de um modelo ajustado em um único split."""
    result = {}
    for name, X, y in [("treino", X_train, y_train), ("teste", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        result[f"roc_auc_{name}"] = roc_auc_score(y, proba)
        result[f"average_precision_{name}"] = average_precision_score(y, proba)
        result[f"f1_{name}"] = f1_score(y, proba >= 0.5)
    return result


def compare_validation(data: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Compara a avaliação em um único split 80/20 com a validação cruzada de 5 folds.

    A diferença entre treino e teste (`gap_roc_auc`) indica overfitting: quanto maior, mais
    o modelo decora o treino em vez de generalizar.
    """
    X_train, X_test, y_train, y_test = split_data(data, random_state=random_state)

    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]

    holdout_model = build_model(random_state).fit(X_train, y_train)
    holdout = _holdout_metrics(holdout_model, X_train, y_train, X_test, y_test)
    cv = cross_validate_model(X_train, y_train, random_state=random_state)

    rows = [
        {
            "validacao": "baseline (classe majoritária)",
            "roc_auc_teste": roc_auc_score(y_test, baseline_proba),
        },
        {"validacao": "holdout 80/20", **holdout},
        {
            "validacao": "cross-validation 5 folds (média)",
            "roc_auc_treino": cv["roc_auc_treino"],
            "roc_auc_teste": cv["roc_auc_teste"],
            "average_precision_teste": cv["average_precision_teste"],
            "f1_teste": cv["f1_teste"],
        },
    ]
    comparison = pd.DataFrame(rows)
    comparison["gap_roc_auc"] = comparison["roc_auc_treino"] - comparison["roc_auc_teste"]
    return comparison


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    report_path: Path = REPORTS_DIR / "validation_comparison.csv",
):
    """Compara holdout e cross-validation, treina o modelo final e salva em disco."""
    data = pd.read_csv(features_path)
    logger.info(f"Dataset carregado: {len(data):,} pedidos")

    comparison = compare_validation(data)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_path, index=False)
    logger.info(f"Comparação de validação salva em {report_path}")

    X_train, _, y_train, _ = split_data(data)
    model = build_model().fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.success(f"Modelo treinado salvo em {model_path}")


if __name__ == "__main__":
    app()
