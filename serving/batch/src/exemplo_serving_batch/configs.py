from mlplatform.serving.contract import BatchServingConfig, register_serving_config
from mlplatform.serving.structure import InferenceBatchStruct

register_serving_config(
    BatchServingConfig(
        domain="exemplo",
        model_name="propensao_exemplo",
        spine_inference_table="workspace.exemplo.spine_inference",
        schedule_cron="0 0 6 * * ?",
        alias="champion",
        # Formato da tabela de predições, conferido antes da escrita.
        # `scored_at` e `model_version` não aparecem aqui: são gravadas pelo
        # framework. O label também não — ele só se materializa semanas depois
        # da inferência, e gravá-lo aqui tornaria a tabela mutável.
        output=InferenceBatchStruct(
            primary_key=["customer_id"],
            ts_date="reference_date",
            feature_cols=["txn_count", "avg_ticket"],
            predict_cols=["prediction"],
        ),
    )
)
