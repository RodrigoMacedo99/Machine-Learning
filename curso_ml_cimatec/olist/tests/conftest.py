import numpy as np
import pandas as pd
import pytest

from module_olist.modeling.split import FEATURES, TARGET


@pytest.fixture
def modeling_dataset() -> pd.DataFrame:
    """Dataset sintético com a mesma forma de features.csv e sinal entre features e alvo."""
    rng = np.random.default_rng(0)
    n_rows = 300

    data = pd.DataFrame(
        {
            "purchase_hour": rng.integers(0, 24, n_rows),
            "purchase_weekday": rng.integers(0, 7, n_rows),
            "purchase_month": rng.integers(1, 13, n_rows),
            "purchase_day": rng.integers(1, 29, n_rows),
            "item_count": rng.integers(1, 6, n_rows),
            "seller_count": rng.integers(1, 4, n_rows),
            "total_price": rng.uniform(5, 500, n_rows),
            "total_freight": rng.uniform(0, 50, n_rows),
            "customer_state": rng.choice(["BA", "PE", "SP"], n_rows),
        }
    )
    late_probability = 0.15 + 0.6 * (data["total_price"] > 250)
    data[TARGET] = (rng.random(n_rows) < late_probability).astype("int8")

    assert set(FEATURES) <= set(data.columns)
    return data
