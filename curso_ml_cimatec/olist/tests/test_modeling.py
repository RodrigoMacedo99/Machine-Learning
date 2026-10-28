import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from module_olist.modeling.active_learning import run_active_learning
from module_olist.modeling.explain import explain_predictions, importance_table
from module_olist.modeling.split import split_data
from module_olist.modeling.train import (
    build_model,
    compare_validation,
    cross_validate_model,
)


@pytest.fixture
def fitted_model(modeling_dataset):
    X_train, _, y_train, _ = split_data(modeling_dataset)
    return build_model().fit(X_train, y_train)


def test_build_model_returns_pipeline_that_predicts_probabilities(modeling_dataset):
    X_train, X_test, y_train, _ = split_data(modeling_dataset)

    model = build_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    assert isinstance(model, Pipeline)
    assert proba.shape == (len(X_test), 2)


def test_build_model_handles_missing_numeric_values(modeling_dataset):
    X_train, X_test, y_train, _ = split_data(modeling_dataset)
    X_test = X_test.copy()
    X_test.loc[X_test.index[0], "total_price"] = np.nan

    model = build_model().fit(X_train, y_train)

    assert np.isfinite(model.predict_proba(X_test)).all()


def test_cross_validate_model_reports_train_and_test_scores(modeling_dataset):
    X_train, _, y_train, _ = split_data(modeling_dataset)

    result = cross_validate_model(X_train, y_train, n_splits=3)

    assert {"roc_auc_treino", "roc_auc_teste", "f1_teste"} <= set(result)
    assert 0.5 < result["roc_auc_teste"] <= 1.0


def test_compare_validation_has_baseline_holdout_and_cv_rows(modeling_dataset):
    comparison = compare_validation(modeling_dataset)

    assert list(comparison["validacao"]) == [
        "baseline (classe majoritária)",
        "holdout 80/20",
        "cross-validation 5 folds (média)",
    ]
    assert "gap_roc_auc" in comparison.columns


def test_explain_predictions_returns_one_value_per_transformed_feature(
    modeling_dataset, fitted_model
):
    _, X_test, _, _ = split_data(modeling_dataset)

    shap_values, sample = explain_predictions(fitted_model, X_test, max_samples=50)
    importance = importance_table(shap_values)

    assert shap_values.values.shape == sample.shape
    assert len(sample) == 50
    assert "categorical__customer_state_BA" in sample.columns
    assert importance["mean_abs_shap"].is_monotonic_decreasing


@pytest.mark.parametrize("strategy", ["uncertainty", "random"])
def test_run_active_learning_grows_labeled_set_each_round(modeling_dataset, strategy):
    X, _, y, _ = split_data(modeling_dataset)

    curve = run_active_learning(
        X,
        y,
        strategy=strategy,
        n_initial=20,
        n_queries=10,
        n_rounds=3,
    )

    assert list(curve["n_labeled"]) == [20, 30, 40, 50]
    assert set(curve["strategy"]) == {strategy}
    assert curve["roc_auc_teste"].between(0, 1).all()
