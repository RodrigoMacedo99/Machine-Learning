"""Pandera schemas for the raw Olist datasets."""

import pandera.pandas as pa
from pandera.typing import Series

ORDER_STATUSES = [
    "delivered",
    "invoiced",
    "shipped",
    "processing",
    "unavailable",
    "canceled",
    "created",
    "approved",
]

PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]

# Bounding box (approx.) of Brazilian territory, used to flag geolocation
# points that fall outside the country as data-quality errors.
BRAZIL_LAT_RANGE = (-34.0, 6.0)
BRAZIL_LNG_RANGE = (-74.0, -32.0)


class OrdersSchema(pa.DataFrameModel):
    order_id: Series[str] = pa.Field(unique=True, nullable=False)
    customer_id: Series[str] = pa.Field(nullable=False)
    order_status: Series[str] = pa.Field(isin=ORDER_STATUSES, nullable=False)
    order_purchase_timestamp: Series[str] = pa.Field(nullable=False)
    order_approved_at: Series[str] = pa.Field(nullable=True)
    order_delivered_carrier_date: Series[str] = pa.Field(nullable=True)
    order_delivered_customer_date: Series[str] = pa.Field(nullable=True)
    order_estimated_delivery_date: Series[str] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True


class CustomersSchema(pa.DataFrameModel):
    customer_id: Series[str] = pa.Field(unique=True, nullable=False)
    customer_unique_id: Series[str] = pa.Field(nullable=False)
    customer_zip_code_prefix: Series[int] = pa.Field(ge=1000, le=99999, nullable=False)
    customer_city: Series[str] = pa.Field(nullable=False)
    customer_state: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, nullable=False)

    class Config:
        strict = True
        coerce = True


class OrderItemsSchema(pa.DataFrameModel):
    order_id: Series[str] = pa.Field(nullable=False)
    order_item_id: Series[int] = pa.Field(ge=1, nullable=False)
    product_id: Series[str] = pa.Field(nullable=False)
    seller_id: Series[str] = pa.Field(nullable=False)
    shipping_limit_date: Series[str] = pa.Field(nullable=False)
    price: Series[float] = pa.Field(gt=0, nullable=False)
    freight_value: Series[float] = pa.Field(ge=0, nullable=False)

    class Config:
        strict = True
        coerce = True


class OrderPaymentsSchema(pa.DataFrameModel):
    order_id: Series[str] = pa.Field(nullable=False)
    payment_sequential: Series[int] = pa.Field(ge=1, nullable=False)
    payment_type: Series[str] = pa.Field(isin=PAYMENT_TYPES, nullable=False)
    payment_installments: Series[int] = pa.Field(ge=0, le=24, nullable=False)
    payment_value: Series[float] = pa.Field(ge=0, nullable=False)

    class Config:
        strict = True
        coerce = True


class OrderReviewsSchema(pa.DataFrameModel):
    review_id: Series[str] = pa.Field(nullable=False)
    order_id: Series[str] = pa.Field(nullable=False)
    review_score: Series[int] = pa.Field(ge=1, le=5, nullable=False)
    review_comment_title: Series[str] = pa.Field(nullable=True)
    review_comment_message: Series[str] = pa.Field(nullable=True)
    review_creation_date: Series[str] = pa.Field(nullable=False)
    review_answer_timestamp: Series[str] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True


class ProductsSchema(pa.DataFrameModel):
    product_id: Series[str] = pa.Field(unique=True, nullable=False)
    product_category_name: Series[str] = pa.Field(nullable=True)
    product_name_lenght: Series[float] = pa.Field(ge=0, nullable=True)
    product_description_lenght: Series[float] = pa.Field(ge=0, nullable=True)
    product_photos_qty: Series[float] = pa.Field(ge=0, nullable=True)
    product_weight_g: Series[float] = pa.Field(gt=0, nullable=True)
    product_length_cm: Series[float] = pa.Field(gt=0, nullable=True)
    product_height_cm: Series[float] = pa.Field(gt=0, nullable=True)
    product_width_cm: Series[float] = pa.Field(gt=0, nullable=True)

    class Config:
        strict = True
        coerce = True


class SellersSchema(pa.DataFrameModel):
    seller_id: Series[str] = pa.Field(unique=True, nullable=False)
    seller_zip_code_prefix: Series[int] = pa.Field(ge=1000, le=99999, nullable=False)
    seller_city: Series[str] = pa.Field(nullable=False)
    seller_state: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, nullable=False)

    class Config:
        strict = True
        coerce = True


class GeolocationSchema(pa.DataFrameModel):
    geolocation_zip_code_prefix: Series[int] = pa.Field(ge=1000, le=99999, nullable=False)
    geolocation_lat: Series[float] = pa.Field(
        ge=BRAZIL_LAT_RANGE[0], le=BRAZIL_LAT_RANGE[1], nullable=False
    )
    geolocation_lng: Series[float] = pa.Field(
        ge=BRAZIL_LNG_RANGE[0], le=BRAZIL_LNG_RANGE[1], nullable=False
    )
    geolocation_city: Series[str] = pa.Field(nullable=False)
    geolocation_state: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, nullable=False)

    class Config:
        strict = True
        coerce = True


class ProductCategoryNameTranslationSchema(pa.DataFrameModel):
    product_category_name: Series[str] = pa.Field(unique=True, nullable=False)
    product_category_name_english: Series[str] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True


# Maps each raw CSV file to the schema that validates it. Shared by the
# validation pipeline (module_olist/validation.py) and by notebooks/tests
# that need to validate a single table on demand.
SCHEMA_REGISTRY: dict[str, type[pa.DataFrameModel]] = {
    "olist_orders_dataset.csv": OrdersSchema,
    "olist_customers_dataset.csv": CustomersSchema,
    "olist_order_items_dataset.csv": OrderItemsSchema,
    "olist_order_payments_dataset.csv": OrderPaymentsSchema,
    "olist_order_reviews_dataset.csv": OrderReviewsSchema,
    "olist_products_dataset.csv": ProductsSchema,
    "olist_sellers_dataset.csv": SellersSchema,
    "olist_geolocation_dataset.csv": GeolocationSchema,
    "product_category_name_translation.csv": ProductCategoryNameTranslationSchema,
}
