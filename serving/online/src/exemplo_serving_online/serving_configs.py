from serving_platform.contract import ServingConfig, register_serving_config

config = ServingConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    mode="online",
    alias="champion",
)

register_serving_config(config)
