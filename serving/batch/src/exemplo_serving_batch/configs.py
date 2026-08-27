from mlplatform.serving.contract import BatchServingConfig, register_serving_config

register_serving_config(
    BatchServingConfig(
        domain="exemplo",
        model_name="propensao_exemplo",
        spine_inference_table="workspace.exemplo.spine_inference",
        schedule_cron="0 0 6 * * ?",
        alias="champion",
    )
)
