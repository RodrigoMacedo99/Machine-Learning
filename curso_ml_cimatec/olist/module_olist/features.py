import pandas as pd

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features a partir do DataFrame fornecido.

    Args:
        data (pd.DataFrame): O DataFrame de entrada contendo os dados originais.

    Returns:
        pd.DataFrame: Um novo DataFrame contendo as features criadas.
    """
    data = data.copy()

    # Calcula quantos dias a empresa prometeu para realizar a entrega,
    # considerando como início o momento da aprovação do pagamento.
    data["promised_days"] = (
        data["order_estimated_delivery_date"]  # Data prometida para a entrega.
        - data["order_approved_at"]            # Data de aprovação do pagamento.
    ).dt.total_seconds().div(86_400)          # Converte segundos para dias.


    # Extrai o número do mês em que a compra foi realizada.
    # Exemplo: janeiro = 1, fevereiro = 2, ..., dezembro = 12.
    data["purchase_month"] = (
        data["order_purchase_timestamp"].dt.month
    )


    # Extrai o dia da semana em que a compra foi realizada.
    #
    # O Pandas representa os dias da seguinte forma:
    # 0 = segunda-feira
    # 1 = terça-feira
    # 2 = quarta-feira
    # 3 = quinta-feira
    # 4 = sexta-feira
    # 5 = sábado
    # 6 = domingo
    data["purchase_weekday"] = (
        data["order_purchase_timestamp"].dt.dayofweek
    )


    # Extrai a hora em que a compra foi realizada.
    # Os valores variam de 0 a 23.
    #
    # Exemplo:
    # 0  = meia-noite
    # 8  = 8 horas
    # 14 = 14 horas
    # 23 = 23 horas
    data["purchase_hour"] = (
        data["order_purchase_timestamp"].dt.hour
    )


    return data


def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Cria o alvo que indica se o pedido foi entregue com atraso."""
    orders = orders.copy()
    orders["target"] = (
        orders["order_delivered_customer_date"]
        > orders["order_estimated_delivery_date"]
    ).astype("int8")
    return orders


def aggregate_order_items(items: pd.DataFrame) -> pd.DataFrame:
    """Agrega os itens para manter uma única linha por pedido."""
    aggregated = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
    )
    return aggregated


def create_dataset(orders, items, customers):
    orders = create_target(orders)      
    items_agg = aggregate_order_items(items)
    data = orders.merge(items_agg, on="order_id", how="left", validate="one_to_one")
    data = data.merge(
        customers[{"customer_id", "customer_city", "customer_state"}],
        on="customer_id",
        how="left", 
        validate="many_to_one"
        )
    
    return data