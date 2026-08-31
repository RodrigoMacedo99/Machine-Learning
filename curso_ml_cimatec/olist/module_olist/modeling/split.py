import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide os dados em conjuntos de treino e teste (estratificado pelo alvo).

    Args:
        data (pd.DataFrame): Os dados de entrada.
        features (list[str]): Colunas a usar como variáveis explicativas.
        target (str): Coluna a usar como variável alvo.
        test_size (float): Proporção reservada para teste.
        random_state (int): Semente para reprodutibilidade.
    """
    try:
        X = data[features]
        y = data[target]
    except KeyError as exc:
        raise KeyError(
            f"Colunas ausentes em 'data': {exc}. "
            f"Esperado features={features} e target='{target}'."
        ) from exc

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as exc:
        raise ValueError(f"Falha ao dividir os dados: {exc}") from exc

    return X_train, X_test, y_train, y_test
