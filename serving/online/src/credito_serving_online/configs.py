"""Endpoint de scoragem em tempo real.

Decisão de crédito no momento da solicitação: o cliente pede aumento de limite
ou um empréstimo, e a resposta precisa vir em milissegundos. As features vêm da
Online Feature Store no Lakebase, resolvidas pela linhagem gravada no modelo —
a chamada manda só a chave do cliente e a data.
"""

from mlplatform.serving.contract import OnlineServingConfig, register_serving_config

register_serving_config(
    OnlineServingConfig(
        domain="credito",
        model_name="pd_inadimplencia",
        alias="champion",
    )
)
