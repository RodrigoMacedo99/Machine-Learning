import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "purchase_day",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state",
]
TARGET = "is_late"


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste (80/20), estratificado pelo alvo."""
    X = data[FEATURES]
    y = data[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
