from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from module_olist.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import load_dataset, save_dataset

app = typer.Typer()


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria as features temporais usadas pelo modelo."""
    data = data.copy()

    data["promised_days"] = (
        data["order_estimated_delivery_date"] - data["order_approved_at"]
    ).dt.total_seconds().div(86_400)

    purchase_timestamp = data["order_purchase_timestamp"]
    data["purchase_month"] = purchase_timestamp.dt.month
    data["purchase_weekday"] = purchase_timestamp.dt.dayofweek
    data["purchase_hour"] = purchase_timestamp.dt.hour
    data["purchase_day"] = purchase_timestamp.dt.day

    return data


def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Cria o alvo que indica se o pedido foi entregue com atraso."""
    orders = orders.copy()
    orders["is_late"] = (
        orders["order_delivered_customer_date"]
        > orders["order_estimated_delivery_date"]
    ).astype("int8")
    return orders


def aggregate_order_items(items: pd.DataFrame) -> pd.DataFrame:
    """Agrega os itens para manter uma única linha por pedido."""
    return items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )

def create_dataset(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Monta a base final de modelagem, com uma linha por pedido."""
    orders = create_features(create_target(orders))
    items_agg = aggregate_order_items(items)
    data = orders.merge(items_agg, on="order_id", how="left", validate="one_to_one")
    data = data.merge(
        customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    return data


@app.command()
def main(
    orders_path: Path = RAW_DATA_DIR / "olist_orders_dataset.csv",
    items_path: Path = RAW_DATA_DIR / "olist_order_items_dataset.csv",
    customers_path: Path = RAW_DATA_DIR / "olist_customers_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> None:
    """Gera a base de features a partir dos dados brutos e salva em CSV."""
    try:
        orders, items, customers = load_dataset(
            orders_path, items_path, customers_path
        )
        dataset = create_dataset(orders, items, customers)
        save_dataset(dataset, output_path)
    except Exception as exc:
        logger.error(f"Falha ao gerar as features: {exc}")
        raise

    logger.success(f"Features geradas com sucesso: {output_path}")


if __name__ == "__main__":
    app()