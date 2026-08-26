import pytest

from monitoring_platform.contract import clear_registry, get_monitoring_config


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_feature_and_predictions_drift_configs_are_registered():
    import exemplo_monitoring.monitoring_configs  # noqa: F401  (dispara o registro)

    feature_drift = get_monitoring_config("exemplo", "propensao_exemplo", "feature_table")
    predictions_drift = get_monitoring_config("exemplo", "propensao_exemplo", "predictions")

    assert feature_drift.target_table == "workspace.exemplo_features.customer_transaction_features"
    assert predictions_drift.target_table == "workspace.exemplo_predictions.propensao_exemplo"
