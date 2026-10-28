import numpy as np
from loguru import logger
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score)

def evaluate_model(
        models, x_test, y_test
) -> None:
    """
    Avalia os modelos treinados usando métricas de classificação.
    
    Args:
        models (dict): Um dicionário com os modelos treinados.
        x_test (np.ndarray): Os dados de teste.
        y_test (np.ndarray): As labels de teste.

    Returns:
        None: Apenas imprime as métricas de avaliação para cada modelo.
    """
    for name, model in models.items():
        try:
            y_proba = model.predict_proba(x_test)[:, 1]

            best_threshold = None
            best_f1 = -1
            best_precision = None
            best_recall = None

            for threshold in np.arange(0.05, 0.50, 0.01):
                y_pred = (y_proba >= threshold).astype(int) # Calcula as previsões binárias com base no limiar atual
                precision = precision_score(y_test, y_pred) # Calcula a precisão que é a taxa de verdadeiros positivos sobre o total de positivos previstos
                recall = recall_score(y_test, y_pred) # Calcula o recall que é a taxa de verdadeiros positivos
                f1 = f1_score(y_test, y_pred) # Calcula o F1 score que é a média harmônica entre precisão e recall

                # Atualiza os melhores valores se o F1 atual for melhor
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall

            roc_auc= roc_auc_score(y_test, y_proba) # Calcula a área sob a curva ROC que é uma métrica de desempenho para classificadores binários

            logger.info(f"Modelo: {name}")
            logger.info(f"Melhor Threshold: {best_threshold:.2f}")
            logger.info(f"Precision: {best_precision:.3f}")
            logger.info(f"Recall: {best_recall:.3f}")
            logger.info(f"F1: {best_f1:.3f}")
            logger.info(f"ROC-AUC: {roc_auc:.3f}")
        except Exception:
            logger.bind(model=name).exception(f"Erro ao avaliar o modelo {name}")

