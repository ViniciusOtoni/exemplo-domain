"""Scoragem em lote da carteira, safra a safra.

Roda depois do fechamento do mês, sobre a safra corrente — inclusive as safras
cujo label ainda não amadureceu, que são justamente as que precisam de decisão.
"""

from mlplatform.serving.contract import BatchServingConfig, register_serving_config
from mlplatform.serving.structure import InferenceBatchStruct

register_serving_config(
    BatchServingConfig(
        domain="credito",
        model_name="pd_inadimplencia",
        spine_inference_table="workspace.credito.spine_inference",
        # 6h da manhã: depois do fechamento das posições do dia anterior.
        schedule_cron="0 0 6 * * ?",
        alias="champion",
        # Formato da tabela de predições, conferido antes da escrita.
        # `scored_at` e `model_version` não aparecem: são gravadas pelo
        # framework. O label também não — ele só se materializa 6 meses depois
        # da inferência, e gravá-lo aqui tornaria a tabela mutável, destruindo
        # a propriedade append-only de que o monitoramento depende.
        output=InferenceBatchStruct(
            primary_key=["customer_id"],
            ts_date="reference_date",
            feature_cols=[
                "utilizacao_limite",
                "comprometimento_renda",
                "share_rotativo",
                "atraso_max_6m",
                "renda_mensal",
                "tempo_relacionamento_meses",
                "n_produtos",
                "saldo_devedor_total",
            ],
            predict_cols=["prediction"],
            # Declarado para o monitoramento saber ONDE o desfecho vai
            # aparecer. Não faz o serving escrevê-lo.
            label_col="label_default",
        ),
    )
)
