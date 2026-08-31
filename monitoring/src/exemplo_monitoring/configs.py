from mlplatform.monitoring.contract import MonitoringConfig, register_monitoring_config

# Drift das features: a distribuição das colunas que alimentam o modelo.
register_monitoring_config(
    MonitoringConfig(
        domain="exemplo",
        model_name="propensao_exemplo",
        target_type="feature_table",
        target_table="workspace.exemplo_features.customer_transaction_features",
        columns=["txn_count", "avg_ticket"],
        threshold=0.2,
        schedule_cron="0 0 7 * * ?",
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
        threshold=0.2,
        schedule_cron="0 0 8 * * ?",
    )
)
