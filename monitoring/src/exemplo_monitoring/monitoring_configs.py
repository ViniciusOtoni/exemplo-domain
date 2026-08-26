from monitoring_platform.contract import MonitoringConfig, register_monitoring_config

feature_drift = MonitoringConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    target_type="feature_table",
    target_table="workspace.exemplo_features.customer_transaction_features",
    columns=["txn_count", "avg_ticket"],
    threshold=0.2,
    schedule_cron="0 0 7 * * ?",
)
register_monitoring_config(feature_drift)

predictions_drift = MonitoringConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    target_type="predictions",
    target_table="workspace.exemplo_predictions.propensao_exemplo",
    columns=["prediction"],
    threshold=0.2,
    schedule_cron="0 0 8 * * ?",
)
register_monitoring_config(predictions_drift)
