"""Monitoramento de drift do modelo de inadimplência.

Este é o domínio em que drift não é hipótese de laboratório: o comprometimento
de renda das famílias subiu 1,7 p.p. em 12 meses, e o mix massificado se
deteriora mais rápido que o de alta renda. Um modelo treinado nas safras de
2024 encontra em 2026 uma carteira que não é a mesma.
"""

from mlplatform.monitoring.contract import MonitoringConfig, register_monitoring_config

# Limiar do PSI, o índice padrão da indústria para drift de distribuição:
# <0.1 estável, 0.1–0.25 moderado, >0.25 significativo.
#
# As métricas limitadas a [0,1] — js_distance e afins — não servem aqui: o
# monitor só as calcula para colunas CATEGÓRICAS, e as observadas abaixo são
# todas numéricas. Escolher uma delas produziria nulo em toda medição, que é
# "sem drift" para sempre.
THRESHOLD = 0.25

# Drift das features. As três primeiras são as que carregam o sinal do
# problema — e são exatamente as que o ciclo de crédito está deslocando.
register_monitoring_config(
    MonitoringConfig(
        domain="credito",
        model_name="pd_inadimplencia",
        target_type="feature_table",
        target_table="workspace.credito_features.perfil_credito_cliente",
        columns=[
            "utilizacao_limite",
            "comprometimento_renda",
            "share_rotativo",
            "atraso_max_6m",
        ],
        threshold=THRESHOLD,
        schedule_cron="0 0 7 * * ?",
        # Compara contra a janela em que o modelo vigente foi treinado, e não
        # contra a safra anterior. A pergunta deixa de ser "mudou desde o mês
        # passado?" e passa a ser "afastou-se do que o modelo aprendeu?" — que
        # é a pergunta que decide retreino.
        baseline_timestamp_column="feature_ts",
    )
)

# Drift do score. Roda uma hora depois do de features, porque deslocamento de
# score só se investiga depois de saber se a entrada se deslocou.
register_monitoring_config(
    MonitoringConfig(
        domain="credito",
        model_name="pd_inadimplencia",
        target_type="predictions",
        target_table="workspace.credito_predictions.pd_inadimplencia",
        columns=["prediction"],
        threshold=THRESHOLD,
        schedule_cron="0 0 8 * * ?",
        # Sem baseline aqui, de propósito: não existem predições do período de
        # treino — o modelo ainda não existia. Para o score, comparar com a
        # safra anterior é o que faz sentido.
    )
)
