import pandas as pd
from pathlib import Path
from loguru import logger

def load_dataset(orders_path: Path, items_path: Path, customers_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega o conjunto de dados Olist a partir dos caminhos especificados.

    Args:
        orders_path (Path): Caminho para o arquivo CSV de pedidos.
        items_path (Path): Caminho para o arquivo CSV de itens dos pedidos.
        customers_path (Path): Caminho para o arquivo CSV de clientes.

    Returns:
        tuple: Uma tupla contendo três DataFrames do pandas: orders, order_items e customers.
    """
    logger.info("Carregando dataset...")

    try:
        orders = pd.read_csv(
            orders_path,
            parse_dates=["order_purchase_timestamp",
                         "order_approved_at",
                         "order_delivered_carrier_date",
                         "order_delivered_customer_date",
                         "order_estimated_delivery_date"]

        )
        items = pd.read_csv(items_path)
        customers = pd.read_csv(customers_path)
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e.filename}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Erro ao interpretar o CSV: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Arquivo CSV vazio: {e}")
        raise

    logger.info("Dataset carregado com sucesso.")

    return orders, items, customers


def save_dataset(dataset: pd.DataFrame, path: Path) -> None:
    """
    Salva o DataFrame em um arquivo CSV no caminho especificado.

    Args:
        dataset (pd.DataFrame): O DataFrame a ser salvo.
        path (Path): Caminho para o arquivo CSV de destino.
    """
    logger.info(f"Salvando dataset em {path}...")
    try:
        dataset.to_csv(path, index=False)
        logger.success(f"Dataset salvo com sucesso: {path}")
    except Exception as e:
        logger.error(f"Erro ao salvar o dataset: {e}")
        raise

def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    # Seleciona apenas os pedidos que podem ser utilizados para construir
    # o histórico de entregas atrasadas e realizadas dentro do prazo.
    delivered_orders = orders.loc[
        # Mantém somente pedidos que foram efetivamente entregues.
        orders["order_status"].eq("delivered")

        # Remove pedidos sem a data real em que o cliente recebeu a compra.
        # Essa data é necessária para saber se o pedido atrasou.
        & orders["order_delivered_customer_date"].notna()

        # Remove pedidos sem a data de entrega prometida ao cliente.
        # Sem essa informação, não é possível comparar o previsto com o realizado.
        & orders["order_estimated_delivery_date"].notna()

        # Mantém somente pedidos com a data de aprovação do pagamento.
        # Esse é o momento definido para realizar a previsão.
        & orders["order_approved_at"].notna()
    ].copy()  # Cria uma cópia independente para evitar alterações no DataFrame original.


    # Cria a variável-alvo do problema:
    # 1 → pedido entregue depois da data prometida;
    # 0 → pedido entregue dentro do prazo ou antes da data prometida.
    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"]
        > delivered_orders["order_estimated_delivery_date"]
    ).astype("int8")  # Armazena 0 e 1 usando um tipo inteiro que ocupa menos memória.


    # Apresenta a quantidade total de pedidos antes da aplicação dos filtros.
    logger.info(f"Pedidos originais: {len(orders):,}")


    # Apresenta quantos pedidos permaneceram no recorte histórico.
    logger.info(f"Pedidos no recorte histórico: {len(delivered_orders):,}")


    # Mostra a quantidade de pedidos em cada classe:
    # 0 = entregue no prazo;
    # 1 = entregue com atraso.
    logger.info(
        delivered_orders["is_late"].value_counts(dropna=False)
    )

    return delivered_orders

def aggregate_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os itens de cada pedido em um único registro.

    Args:
        order_items (pd.DataFrame): DataFrame contendo os itens dos pedidos.

    Returns:
        pd.DataFrame: DataFrame agregado com informações consolidadas por pedido.
    """
    logger.info("Agregando itens do pedido...")

    aggregated_items = (
        order_items.groupby("order_id")
        .agg(
            total_order_value=pd.NamedAgg(column="price", aggfunc="sum"),
            total_freight_value=pd.NamedAgg(column="freight_value", aggfunc="sum"),
            total_items=pd.NamedAgg(column="order_item_id", aggfunc="count"),
        )
        .reset_index()
    )

    logger.info("Itens do pedido agregados com sucesso.")

    return aggregated_items

def aggregate_itens(items: pd.DataFrame) -> pd.DataFrame:
    # A tabela de itens possui uma linha para cada item presente no pedido.
    # Portanto, um mesmo order_id pode aparecer várias vezes.
    #
    # Como o objetivo é construir uma base com uma linha por pedido,
    # precisamos agrupar os itens antes de integrar essa tabela às demais.
    items_agg = (
        items.groupby(
            "order_id",       # Agrupa todos os itens pertencentes ao mesmo pedido.
            as_index=False,   # Mantém order_id como uma coluna comum.
        )
        .agg(
            # Conta quantas linhas de itens existem em cada pedido.
            # Um pedido com três produtos registrados terá item_count igual a 3.
            item_count=("order_item_id", "count"),

            # Conta quantos vendedores diferentes participam do pedido.
            # O nunique evita contar o mesmo vendedor mais de uma vez.
            seller_count=("seller_id", "nunique"),

            # Soma os preços dos itens para obter o valor total dos produtos
            # presentes no pedido.
            total_price=("price", "sum"),

            # Soma o frete de todos os itens para obter o valor total de frete
            # associado ao pedido.
            total_freight=("freight_value", "sum"),
        )
    )


    # Verifica se cada pedido aparece somente uma vez após a agregação.
    #
    # Se a condição for falsa, o Python interromperá a execução e lançará
    # um AssertionError. Essa checagem ajuda a garantir que a unidade de
    # análise da nova tabela é realmente o pedido.
    assert items_agg["order_id"].is_unique


    # Retrona as cinco primeiras linhas da tabela agregada.
    return items_agg
