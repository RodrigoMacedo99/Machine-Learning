"""Treina uma das CNNs do CIFAR-10, avalia no teste e salva o modelo e as métricas.

Exemplos:
    uv run python -m src.models.train_model --version v1
    uv run python -m src.models.train_model --version v2
"""

from pathlib import Path

from src.gpu import preload_cuda_libraries

# Precisa acontecer antes de importar o TensorFlow, por isso fica acima dos demais imports.
preload_cuda_libraries()

import json  # noqa: E402

import click  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.data.cifar import load_cifar10  # noqa: E402
from src.models.cnn import BUILDERS  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

# Épocas padrão de cada versão: a v1 é só linha de base; a v2 precisa de mais épocas com regularização.
DEFAULT_EPOCHS = {"v1": 8, "v2": 40}


def callbacks_for(version: str) -> list[tf.keras.callbacks.Callback]:
    if version == "v1":
        return []
    return [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]


@click.command()
@click.option("--version", type=click.Choice(["v1", "v2"]), default="v2", show_default=True)
@click.option("--epochs", type=int, default=None, help="Padrão: 8 para v1 e 40 para v2.")
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--model-dir", type=click.Path(path_type=Path), default=MODELS_DIR, show_default=True)
@click.option("--report-dir", type=click.Path(path_type=Path), default=REPORTS_DIR, show_default=True)
def main(version, epochs, batch_size, model_dir, report_dir):
    """Treina a CNN escolhida e grava o modelo e as métricas de teste."""
    click.echo(f"GPUs disponíveis: {tf.config.list_physical_devices('GPU')}")
    tf.keras.utils.set_random_seed(42)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10()
    model = BUILDERS[version]()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    epochs = epochs or DEFAULT_EPOCHS[version]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_for(version),
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    metrics = {
        "version": version,
        "epochs_run": len(history.history["loss"]),
        "best_val_accuracy": max(history.history["val_accuracy"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"cifar10_{version}.keras")
    (report_dir / f"metrics_{version}.json").write_text(json.dumps(metrics, indent=2))
    click.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
