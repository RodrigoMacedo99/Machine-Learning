from pathlib import Path

import pandas as pd
import pytest

from module_olist.dataset import load_dataset, save_dataset

ORDERS_COLUMNS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


@pytest.fixture
def orders_csv(tmp_path: Path) -> Path:
    path = tmp_path / "orders.csv"
    pd.DataFrame(
        {
            "order_id": ["o1"],
            "customer_id": ["c1"],
            "order_status": ["delivered"],
            "order_purchase_timestamp": ["2018-01-01 10:00:00"],
            "order_approved_at": ["2018-01-01 11:00:00"],
            "order_delivered_carrier_date": ["2018-01-02 10:00:00"],
            "order_delivered_customer_date": ["2018-01-05 10:00:00"],
            "order_estimated_delivery_date": ["2018-01-10 00:00:00"],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def items_csv(tmp_path: Path) -> Path:
    path = tmp_path / "items.csv"
    pd.DataFrame({"order_id": ["o1"], "product_id": ["p1"]}).to_csv(path, index=False)
    return path


@pytest.fixture
def customers_csv(tmp_path: Path) -> Path:
    path = tmp_path / "customers.csv"
    pd.DataFrame({"customer_id": ["c1"], "customer_city": ["salvador"]}).to_csv(path, index=False)
    return path


def test_load_dataset_returns_three_dataframes(orders_csv, items_csv, customers_csv):
    orders, items, customers = load_dataset(orders_csv, items_csv, customers_csv)

    assert list(orders.columns) == ORDERS_COLUMNS
    assert len(orders) == 1
    assert len(items) == 1
    assert len(customers) == 1


def test_load_dataset_parses_date_columns(orders_csv, items_csv, customers_csv):
    orders, _, _ = load_dataset(orders_csv, items_csv, customers_csv)

    for column in DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(orders[column])


def test_load_dataset_raises_when_file_missing(tmp_path, items_csv, customers_csv):
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path, items_csv, customers_csv)


def test_load_dataset_raises_when_file_is_empty(tmp_path, items_csv, customers_csv):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")

    with pytest.raises(pd.errors.EmptyDataError):
        load_dataset(empty_path, items_csv, customers_csv)


def test_save_dataset_writes_csv_to_disk(tmp_path):
    dataset = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    output_path = tmp_path / "output.csv"

    save_dataset(dataset, output_path)

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(saved, dataset)


def test_save_dataset_raises_when_directory_missing(tmp_path):
    dataset = pd.DataFrame({"a": [1]})
    output_path = tmp_path / "missing_dir" / "output.csv"

    with pytest.raises(OSError):
        save_dataset(dataset, output_path)
