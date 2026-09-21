from pathlib import Path

from loguru import logger
import typer

from module_olist.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import load_dataset, save_dataset
from module_olist.features import create_dataset
from module_olist.modeling.evaluate import evaluate_model
from module_olist.modeling.pipeline import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from module_olist.modeling.split import split_data
from module_olist.modeling.train import cross_validate_models, train_models

app = typer.Typer()


FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def run_pipeline(
	orders_path: Path,
	items_path: Path,
	customers_path: Path,
	features_path: Path | None = None,
) -> dict:
	"""
	Prepara os dados, treina os modelos e registra suas métricas.
    """
	orders, items, customers = load_dataset(orders_path, items_path, customers_path)
	dataset = create_dataset(orders, items, customers).dropna(subset=FEATURE_COLUMNS + ["is_late"])

	if features_path is not None:
		features_path.parent.mkdir(parents=True, exist_ok=True)
		save_dataset(dataset, features_path)

	x_train, x_test, y_train, y_test = split_data(
		dataset, features=FEATURE_COLUMNS, target="is_late"
	)
	cross_validate_models(x_train, y_train)
	models = train_models(x_train, y_train)
	if not models:
		raise RuntimeError("Nenhum modelo foi treinado.")

	evaluate_model(models, x_test, y_test)
	return models


@app.command()
def main(
	orders_path: Path = RAW_DATA_DIR / "olist_orders_dataset.csv",
	items_path: Path = RAW_DATA_DIR / "olist_order_items_dataset.csv",
	customers_path: Path = RAW_DATA_DIR / "olist_customers_dataset.csv",
	features_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> None:
	"""Treina e avalia os modelos de previsão de atraso da Olist."""
	logger.info("Iniciando pipeline de modelagem...")
	run_pipeline(orders_path, items_path, customers_path, features_path)
	logger.success("Pipeline concluído com sucesso.")


if __name__ == "__main__":
	app()
