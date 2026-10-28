import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_validate

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)

# Métricas coletadas tanto na cross-validation quanto (futuramente) na
# avaliação final. roc_auc mede a separação geral entre as classes;
# f1/precision/recall focam no desempenho especificamente na classe
# minoritária (pedidos atrasados).
SCORING = ["roc_auc", "f1", "precision", "recall"]


def _build_models() -> dict:
    # Cria uma instância nova de cada pipeline a cada chamada, para que
    # train_models e cross_validate_models nunca compartilhem o mesmo
    # objeto de modelo (evita que o fit de um contamine o outro).
    return {
        "gradient_boosting": create_gradient_boosting_pipeline(),
        "xgboost": create_xgboost_pipeline(),
        "lightgbm": create_lightgbm_pipeline(),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Treina os modelos de machine learning e retorna um dicionário com os modelos treinados.

    Args:
        X_train (pd.DataFrame): Os dados de treinamento.
        y_train (pd.Series): As labels de treinamento.

    Returns:
        dict: Um dicionário com os modelos treinados.
    """
    # Inicializado fora do try para garantir que a função sempre retorne
    # um dict (mesmo vazio) e nunca lance UnboundLocalError.
    trained_models = {}
    try:
        # Ajusta cada pipeline (pré-processador + modelo) nos dados de
        # treino. Cada modelo é treinado de forma independente, então a
        # falha de um não deveria impedir os demais — mas como estão no
        # mesmo laço, uma exceção aqui interrompe o restante do loop.
        for model_name, model in _build_models().items():
            model.fit(X_train, y_train)
            trained_models[model_name] = model
    except Exception as e:
        logger.exception(f"Erro ao treinar modelos: {e}")

    return trained_models


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Avalia cada pipeline via k-fold estratificado, sem ajustar o modelo final.

    Args:
        X (pd.DataFrame): Features de treino.
        y (pd.Series): Alvo de treino.
        cv (int): Número de folds.
        random_state (int): Semente para reprodutibilidade do KFold.

    Returns:
        dict: Para cada modelo, média e desvio padrão de cada métrica em SCORING.
    """
    # StratifiedKFold mantém a proporção de pedidos atrasados/no prazo em
    # cada fold, já que a classe positiva (is_late=1) é minoritária.
    # shuffle=True evita que a ordem original dos dados vire viés entre
    # os folds; random_state fixa a semente para o resultado ser
    # reprodutível entre execuções.
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    results = {}
    for model_name, model in _build_models().items():
        try:
            # cross_validate treina e avalia o pipeline em cada um dos
            # `cv` folds, sem deixar o modelo final ajustado — serve só
            # para comparar candidatos antes do treino definitivo.
            scores = cross_validate(model, X, y, cv=splitter, scoring=SCORING)
        except Exception:
            # Um modelo específico pode falhar (ex.: parâmetro
            # incompatível com os dados) sem que isso interrompa a
            # avaliação dos demais modelos.
            logger.exception(f"Erro na cross-validation do modelo {model_name}")
            continue

        # cross_validate devolve um array de `cv` valores por métrica
        # (um por fold), em chaves "test_<metrica>". Antes de resumir,
        # logamos o resultado fold a fold para inspecionar a variação
        # entre eles (ex.: um fold muito pior que os outros pode indicar
        # um subconjunto de dados atípico).
        n_folds = len(scores[f"test_{SCORING[0]}"])
        for fold_idx in range(n_folds):
            fold_metrics = ", ".join(
                f"{metric}={scores[f'test_{metric}'][fold_idx]:.3f}"
                for metric in SCORING
            )
            logger.debug(f"[CV] {model_name} | fold {fold_idx + 1}/{n_folds}: {fold_metrics}")

        # Depois do detalhe por fold, resumimos cada métrica em média
        # (desempenho esperado) e desvio padrão (estabilidade entre folds).
        summary = {
            metric: {
                "mean": scores[f"test_{metric}"].mean(),
                "std": scores[f"test_{metric}"].std(),
            }
            for metric in SCORING
        }
        results[model_name] = summary

        logger.info(
            f"[CV] {model_name}: "
            + ", ".join(
                f"{metric}={values['mean']:.3f}±{values['std']:.3f}"
                for metric, values in summary.items()
            )
        )

    return results
