from mlplatform.monitoring.contract import MonitoringConfig, register_monitoring_config

# Limiar de 0.25 no PSI (population stability index), que é o default do
# framework e a convenção da indústria: <0.1 estável, 0.1–0.25 moderado,
# >0.25 significativo. As métricas limitadas a [0,1] — js_distance e afins —
# não servem aqui: o monitor só as calcula para colunas CATEGÓRICAS, e todas
# as observadas abaixo são numéricas. Escolher uma delas produziria nulo em
# toda medição, que é "sem drift" para sempre.
THRESHOLD = 0.25

# Drift das features: a distribuição das colunas que alimentam o modelo.
register_monitoring_config(
    MonitoringConfig(
        domain="exemplo",
        model_name="propensao_exemplo",
        target_type="feature_table",
        target_table="workspace.exemplo_features.customer_transaction_features",
        columns=["txn_count", "avg_ticket"],
        threshold=THRESHOLD,
        schedule_cron="0 0 7 * * ?",
        # Compara contra a janela em que o modelo vigente foi treinado, e não
        # contra a safra anterior. A pergunta muda: deixa de ser "mudou desde
        # ontem?" e passa a ser "afastou-se do que o modelo aprendeu?".
        baseline_timestamp_column="feature_ts",
    )
)

# Drift das predições: a distribuição do score. Roda uma hora depois do de
# features, porque só faz sentido investigar deslocamento de score depois de
# saber se a entrada se deslocou.
register_monitoring_config(
    MonitoringConfig(
        domain="exemplo",
        model_name="propensao_exemplo",
        target_type="predictions",
        target_table="workspace.exemplo_predictions.propensao_exemplo",
        columns=["prediction"],
        threshold=THRESHOLD,
        schedule_cron="0 0 8 * * ?",
        # Sem baseline aqui, de propósito: não existem predições do período de
        # treino — o modelo ainda não existia. Comparar com a safra anterior é
        # o que faz sentido para o score.
    )
)
