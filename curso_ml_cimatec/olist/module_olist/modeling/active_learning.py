from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import typer

from module_olist.config import PROCESSED_DATA_DIR, REPORTS_DIR
from module_olist.modeling.split import split_data
from module_olist.modeling.train import build_model

app = typer.Typer()


def select_queries(
    model,
    X_pool: pd.DataFrame,
    n_queries: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Escolhe as posições do pool que serão rotuladas na próxima rodada.

    - `uncertainty`: pedidos cuja probabilidade está mais próxima de 0.5 (o modelo está em dúvida).
    - `random`: amostragem aleatória, usada como linha de base.
    """
    if strategy == "random":
        return rng.choice(len(X_pool), size=n_queries, replace=False)

    proba = model.predict_proba(X_pool)[:, 1]
    uncertainty = 1 - np.abs(2 * proba - 1)
    return np.argsort(-uncertainty)[:n_queries]


def run_active_learning(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    n_initial: int = 100,
    n_queries: int = 20,
    n_rounds: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simula o active learning sobre um pool em que os rótulos são conhecidos.

    A cada rodada o modelo é treinado com os rótulos disponíveis, avaliado em um conjunto de
    teste fixo, e `n_queries` novos pedidos são rotulados segundo a estratégia escolhida.
    """
    rng = np.random.default_rng(random_state)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    labeled = list(rng.choice(len(X_pool), size=n_initial, replace=False))
    rows = []
    for round_number in range(n_rounds + 1):
        unlabeled = np.setdiff1d(np.arange(len(X_pool)), labeled)
        model = build_model(random_state).fit(X_pool.iloc[labeled], y_pool.iloc[labeled])
        proba = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "strategy": strategy,
                "round": round_number,
                "n_labeled": len(labeled),
                "roc_auc_teste": roc_auc_score(y_test, proba),
            }
        )
        logger.info(f"[{strategy}] rodada {round_number}: {len(labeled)} rótulos")

        if round_number == n_rounds:
            break

        picks = select_queries(
            model,
            X_pool.iloc[unlabeled],
            n_queries,
            strategy,
            rng,
        )
        labeled.extend(unlabeled[picks])

    return pd.DataFrame(rows)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    report_path: Path = REPORTS_DIR / "active_learning_curve.csv",
):
    """Compara active learning por incerteza com amostragem aleatória."""
    data = pd.read_csv(features_path)
    X, _, y, _ = split_data(data)

    curves = pd.concat(
        [run_active_learning(X, y, strategy) for strategy in ["uncertainty", "random"]],
        ignore_index=True,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(report_path, index=False)
    logger.success(f"Curvas de active learning salvas em {report_path}")


if __name__ == "__main__":
    app()
