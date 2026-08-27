from mlplatform.serving.contract import OnlineServingConfig, register_serving_config

register_serving_config(
    OnlineServingConfig(
        domain="exemplo",
        model_name="propensao_exemplo",
        alias="champion",
    )
)
