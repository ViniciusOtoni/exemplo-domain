from serving_platform.contract import ServingConfig, register_serving_config

config = ServingConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    mode="batch",
    alias="champion",
    spine_inference_table="workspace.exemplo.spine_inference",
    schedule_cron="0 0 6 * * ?",
)

register_serving_config(config)
