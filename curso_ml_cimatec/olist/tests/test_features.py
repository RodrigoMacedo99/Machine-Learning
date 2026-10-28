import pandas as pd
import pytest

from module_olist.features import (
    aggregate_order_items,
    create_dataset,
    create_features,
    create_target,
)
from module_olist.modeling.split import FEATURES, TARGET, split_data


@pytest.fixture
def orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01 10:00:00", "2018-02-03 18:30:00"]
            ),
            "order_approved_at": pd.to_datetime(["2018-01-01 11:00:00", "2018-02-03 19:00:00"]),
            "order_delivered_customer_date": pd.to_datetime(
                ["2018-01-05 10:00:00", "2018-02-02 10:00:00"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2018-01-04 00:00:00", "2018-02-10 00:00:00"]
            ),
        }
    )


@pytest.fixture
def items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "order_item_id": [1, 2, 1],
            "seller_id": ["s1", "s2", "s1"],
            "price": [10.0, 5.0, 20.0],
            "freight_value": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "customer_city": ["salvador", "recife"],
            "customer_state": ["BA", "PE"],
        }
    )


def test_create_target_flags_late_orders(orders):
    result = create_target(orders)

    assert result["is_late"].tolist() == [1, 0]
    assert result["is_late"].dtype == "int8"


def test_create_features_computes_promised_days_and_purchase_parts(orders):
    result = create_features(orders)

    assert result["promised_days"].tolist() == pytest.approx([2.541666, 6.208333], abs=1e-3)
    assert result["purchase_month"].tolist() == [1, 2]
    assert result["purchase_day"].tolist() == [1, 3]
    assert result["purchase_hour"].tolist() == [10, 18]
    assert result["purchase_weekday"].tolist() == [0, 5]


def test_aggregate_order_items_keeps_one_row_per_order(items):
    result = aggregate_order_items(items).set_index("order_id")

    assert result.loc["o1", "item_count"] == 2
    assert result.loc["o1", "seller_count"] == 2
    assert result.loc["o1", "total_price"] == 15.0
    assert result.loc["o1", "total_freight"] == 3.0
    assert result.index.is_unique


def test_create_dataset_has_one_row_per_order(orders, items, customers):
    result = create_dataset(orders, items, customers)

    assert len(result) == len(orders)
    assert result["order_id"].is_unique
    assert set(FEATURES + [TARGET]) <= set(result.columns)


def make_modeling_dataset(n_rows: int = 100) -> pd.DataFrame:
    data = {feature: range(n_rows) for feature in FEATURES if feature != "customer_state"}
    data["customer_state"] = ["BA", "PE"] * (n_rows // 2)
    data[TARGET] = [1, 0, 0, 0, 0] * (n_rows // 5)
    return pd.DataFrame(data)


def test_split_data_uses_80_20_and_keeps_target_ratio():
    dataset = make_modeling_dataset()

    X_train, X_test, y_train, y_test = split_data(dataset)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert list(X_train.columns) == FEATURES
    assert y_train.mean() == pytest.approx(y_test.mean())


def test_split_data_raises_when_feature_is_missing():
    dataset = make_modeling_dataset().drop(columns=["item_count"])

    with pytest.raises(KeyError):
        split_data(dataset)
