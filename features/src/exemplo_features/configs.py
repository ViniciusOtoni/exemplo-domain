import pyspark.sql.functions as F

from mlplatform.features.contract import feature_table


@feature_table(
    domain="exemplo",
    entity_keys=["customer_id"],
    timestamp_key="feature_ts",
    sources=["raw.transactions"],
    online=True,
)
def customer_transaction_features(sources, window):
    raw = sources["raw.transactions"]
    return (
        raw.filter((F.col("event_ts") >= F.lit(window.start)) & (F.col("event_ts") < F.lit(window.end)))
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("txn_count"),
            F.avg("amount").alias("avg_ticket"),
        )
        .withColumn("feature_ts", F.lit(window.end))
    )
