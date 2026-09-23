"""Train a binary classifier on the Keras heart disease dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pandas as pd
from keras import layers

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
TARGET_FEATURE_NAME = "target"
NUMERIC_FEATURE_NAMES = ["age", "trestbps", "thalach", "oldpeak", "slope", "chol"]


def load_data(url: str = DATA_URL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the dataset and create deterministic train/validation splits."""
    dataframe = pd.read_csv(url)
    validation = dataframe.sample(frac=0.2, random_state=1337)
    training = dataframe.drop(validation.index)
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def _category_vocabularies(dataframe: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        feature_name: {
            value: index + 1
            for index, value in enumerate(sorted(dataframe[feature_name].astype(str).unique()))
        }
        for feature_name in dataframe.columns
        if feature_name not in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME]
    }


def _feature_arrays(
    dataframe: pd.DataFrame, vocabularies: dict[str, dict[str, int]]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features = dataframe.drop(columns=[TARGET_FEATURE_NAME]).copy()
    labels = dataframe[TARGET_FEATURE_NAME].to_numpy(dtype="float32")

    arrays: dict[str, np.ndarray] = {}
    for feature_name in features.columns:
        if feature_name in NUMERIC_FEATURE_NAMES:
            arrays[feature_name] = features[feature_name].to_numpy(dtype="float32")
        else:
            arrays[feature_name] = (
                features[feature_name]
                .astype(str)
                .map(vocabularies[feature_name])
                .fillna(0)
                .to_numpy(dtype="int32")
            )
    return arrays, labels


def build_model(training_dataframe: pd.DataFrame) -> keras.Model:
    """Build preprocessing layers and a dense classifier from the training data."""
    vocabularies = _category_vocabularies(training_dataframe)
    inputs: dict[str, keras.KerasTensor] = {}
    encoded_features: list[keras.KerasTensor] = []

    for feature_name in training_dataframe.columns:
        if feature_name == TARGET_FEATURE_NAME:
            continue

        if feature_name in NUMERIC_FEATURE_NAMES:
            input_tensor = keras.Input(shape=(1,), name=feature_name, dtype="float32")
            normalizer = layers.Normalization(axis=None, name=f"{feature_name}_normalizer")
            normalizer.adapt(training_dataframe[feature_name].to_numpy(dtype="float32"))
            encoded = normalizer(input_tensor)
        else:
            input_tensor = keras.Input(shape=(1,), name=feature_name, dtype="int32")
            encoded = keras.ops.one_hot(
                input_tensor, num_classes=len(vocabularies[feature_name]) + 1
            )

        encoded = layers.Flatten(name=f"{feature_name}_flatten")(encoded)
        inputs[feature_name] = input_tensor
        encoded_features.append(encoded)

    features = layers.concatenate(encoded_features, name="encoded_features")
    features = layers.Dense(32, activation="relu")(features)
    features = layers.Dropout(0.2)(features)
    output = layers.Dense(1, activation="sigmoid", name="prediction")(features)

    model = keras.Model(inputs=inputs, outputs=output, name="heart_disease_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
    )
    return model


def train(
    epochs: int = 50,
    batch_size: int = 32,
    model_path: str | Path = "models/heart_disease_classifier.keras",
) -> dict[str, Any]:
    """Train, evaluate, and save the classifier."""
    training_dataframe, validation_dataframe = load_data()
    vocabularies = _category_vocabularies(training_dataframe)
    training_features, training_labels = _feature_arrays(training_dataframe, vocabularies)
    validation_features, validation_labels = _feature_arrays(validation_dataframe, vocabularies)
    model = build_model(training_dataframe)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=8, restore_best_weights=True
        )
    ]
    model.fit(
        training_features,
        training_labels,
        validation_data=(validation_features, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    metrics = model.evaluate(validation_features, validation_labels, verbose=0, return_dict=True)

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    print(f"Validation metrics: {metrics}")
    print(f"Model saved to: {destination}")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", default="models/heart_disease_classifier.keras")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, model_path=args.model_path)


if __name__ == "__main__":
    main()