from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

NUMERICAL_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state"
]   


def create_preprocessor() -> ColumnTransformer:
    """
    Cria o pré-processador para os dados.
    Retorna:
        ColumnTransformer: O pré-processador que aplica OneHotEncoder nas colunas categóricas e mantém as colunas numéricas inalteradas.
    """
    return ColumnTransformer(
        transformers=[
            # Nas colunas numéricas
            ("numeric", "passthrough", NUMERICAL_FEATURES),
            # Nas colunas categóricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Cria o pipeline de modelagem com Gradient Boosting.
    Retorna:
        Pipeline: O pipeline que combina o pré-processador e o modelo Gradient Boosting.
    """
    preprocessor = create_preprocessor()
    model = GradientBoostingClassifier(
            n_estimators=100, # Número de árvores na floresta
            learning_rate=0.1, # Taxa de aprendizado
            max_depth=3, # Profundidade máxima das árvores
            random_state=42, # Semente para reprodutibilidade
            verbose=0,
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def create_xgboost_pipeline() -> Pipeline:
    """
    Cria o pipeline de modelagem com XGBoost.
    Retorna:
        Pipeline: O pipeline que combina o pré-processador e o modelo XGBoost.
    """
    preprocessor = create_preprocessor()
    model = XGBClassifier(
            n_estimators=100, # Número de árvores na floresta
            learning_rate=0.1, # Taxa de aprendizado
            max_depth=3, # Profundidade máxima das árvores
            random_state=42, # Semente para reprodutibilidade
            verbosity=0,
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def create_lightgbm_pipeline() -> Pipeline:
    """
    Cria o pipeline de modelagem com LightGBM.
    Retorna:
        Pipeline: O pipeline que combina o pré-processador e o modelo LightGBM.
    """
    preprocessor = create_preprocessor()
    model = LGBMClassifier(
            n_estimators=100, # Número de árvores na floresta
            learning_rate=0.1, # Taxa de aprendizado
            max_depth=3, # Profundidade máxima das árvores
            random_state=42, # Semente para reprodutibilidade
            verbosity=-1,
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])