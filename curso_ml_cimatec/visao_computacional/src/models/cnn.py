"""Arquiteturas das CNNs usadas nos classificadores de imagens."""

import tensorflow as tf


def build_cnn_v1() -> tf.keras.Model:
    """Linha de base: duas convoluções com pooling e uma camada densa."""
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="cnn_v1",
    )


def _conv_block(filters: int) -> list[tf.keras.layers.Layer]:
    return [
        tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
    ]


def build_cnn_v2() -> tf.keras.Model:
    """Três blocos convolucionais com BatchNorm, aumento de dados e dropout."""
    layers = [
        tf.keras.Input(shape=(32, 32, 3)),
        # Aumento de dados: ativo só no treino.
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ]
    for filters in (32, 64, 128):
        layers += _conv_block(filters)
    layers += [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
    return tf.keras.Sequential(layers, name="cnn_v2")


BUILDERS = {"v1": build_cnn_v1, "v2": build_cnn_v2}
