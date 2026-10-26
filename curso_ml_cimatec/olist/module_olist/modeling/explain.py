from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import shap
import typer

from module_olist.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from module_olist.modeling.split import split_data

app = typer.Typer()


def explain_predictions(model, X: pd.DataFrame, max_samples: int = 500, random_state: int = 42):
    """Calcula os valores SHAP da classe "atrasado" para uma amostra de pedidos.

    O pré-processamento é aplicado antes do explainer para que os valores SHAP sejam
    indexados pelas colunas transformadas (ex.: `categorical__customer_state_BA`).
    """
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    X_transformed = pd.DataFrame(
        preprocessor.transform(X),
        columns=preprocessor.get_feature_names_out(),
    )
    sample = X_transformed.sample(
        min(max_samples, len(X_transformed)),
        random_state=random_state,
    )

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(sample)
    # Para classificação binária, o shap devolve (amostras, features, classes).
    return shap_values[:, :, 1], sample


def importance_table(shap_values) -> pd.DataFrame:
    """Importância média absoluta de cada feature, em ordem decrescente."""
    importance = pd.DataFrame(
        {
            "feature": shap_values.feature_names,
            "mean_abs_shap": abs(shap_values.values).mean(axis=0),
        }
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    figure_path: Path = FIGURES_DIR / "shap_summary.png",
    report_path: Path = REPORTS_DIR / "shap_importance.csv",
):
    """Gera o gráfico-resumo SHAP e a tabela de importância no conjunto de teste."""
    data = pd.read_csv(features_path)
    _, X_test, _, _ = split_data(data)
    model = joblib.load(model_path)

    shap_values, sample = explain_predictions(model, X_test)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()
    logger.success(f"Gráfico SHAP salvo em {figure_path}")

    importance = importance_table(shap_values)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(report_path, index=False)
    logger.info(f"Importância SHAP salva em {report_path}")


if __name__ == "__main__":
    app()
