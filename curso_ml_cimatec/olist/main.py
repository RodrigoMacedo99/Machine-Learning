"""Roda o pipeline completo do Olist, do dataset bruto até as explicações do modelo.

Ordem: features -> treino (com validação) -> SHAP -> active learning.
Os dados brutos precisam estar em data/raw/.
"""

from loguru import logger
import typer

from module_olist.config import PROCESSED_DATA_DIR
from module_olist.features import main as build_features
from module_olist.modeling.active_learning import main as run_active_learning
from module_olist.modeling.explain import main as explain_model
from module_olist.modeling.train import main as train_model

app = typer.Typer()


@app.command()
def main(
    skip_active_learning: bool = typer.Option(
        False, help="Pula o active learning, que é a etapa mais lenta."
    ),
):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("1/4 Gerando features")
    build_features()

    logger.info("2/4 Treinando o modelo e comparando validações")
    train_model()

    logger.info("3/4 Explicando o modelo com SHAP")
    explain_model()

    if skip_active_learning:
        logger.info("4/4 Active learning pulado")
    else:
        logger.info("4/4 Rodando active learning")
        run_active_learning()

    logger.success("Pipeline concluído")


if __name__ == "__main__":
    app()
