from training_platform.contract import clear_registry, get_training_config


def setup_function():
    clear_registry()


def teardown_function():
    clear_registry()


def test_propensao_exemplo_is_registered_with_feature_lookups():
    import exemplo_training.training_configs  # noqa: F401  (dispara o registro)

    config = get_training_config("propensao_exemplo")

    assert config.domain == "exemplo"
    assert config.label_column == "label_default"
    assert config.feature_lookups[0].table_name == "workspace.exemplo_features.customer_transaction_features"
    assert abs(config.train_pct + config.val_pct + config.test_pct - 1.0) < 1e-9
