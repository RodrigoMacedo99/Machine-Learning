"""Carregamento do CIFAR-10 com validação separada do teste."""

import tensorflow as tf

CLASSES = [
    "avião", "automóvel", "pássaro", "gato", "cervo",
    "cachorro", "sapo", "cavalo", "navio", "caminhão",
]


def load_cifar10(val_size: int = 5000):
    """Devolve ((X_train, y_train), (X_val, y_val), (X_test, y_test)) com pixels em [0, 1].

    A validação vem do fim do conjunto de treino, para acompanhar o overfitting sem tocar no teste.
    """
    (X, y), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y, y_test = y.ravel(), y_test.ravel()
    X = X.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    split = len(X) - val_size
    return (X[:split], y[:split]), (X[split:], y[split:]), (X_test, y_test)
