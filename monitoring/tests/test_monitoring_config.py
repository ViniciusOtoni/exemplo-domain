import importlib

import pytest

from mlplatform.monitoring.contract import clear_registry, get_monitoring_config


@pytest.fixture(autouse=True)
def _registered():
    """Recarrega o módulo de configs a cada teste.

    Importar de novo não basta: o módulo já está em `sys.modules` e o import
    vira no-op, então o registro que o `clear_registry` esvaziou nunca é
    refeito. O segundo teste do arquivo é que revela isso — com um só, o
    problema fica escondido.
    """
    import exemplo_monitoring.configs as configs

    clear_registry()
    importlib.reload(configs)
    yield
    clear_registry()


def test_feature_and_predictions_drift_are_registered():
    features = get_monitoring_config("exemplo", "propensao_exemplo", "feature_table")
    predictions = get_monitoring_config("exemplo", "propensao_exemplo", "predictions")

    assert features.target_table == "workspace.exemplo_features.customer_transaction_features"
    assert predictions.target_table == "workspace.exemplo_predictions.propensao_exemplo"


def test_the_monitored_columns_are_the_ones_the_pipeline_produces():
    """Uma coluna que não existe na tabela alvo não gera linha na tabela de
    métricas e é simplesmente ignorada — o job passa verde monitorando nada."""
    features = get_monitoring_config("exemplo", "propensao_exemplo", "feature_table")
    predictions = get_monitoring_config("exemplo", "propensao_exemplo", "predictions")

    assert features.columns == ["txn_count", "avg_ticket"]
    assert predictions.columns == ["prediction"]


def test_features_are_checked_before_predictions():
    """Deslocamento de score só se investiga depois de saber se a entrada se
    deslocou; os dois horários codificam essa ordem."""
    features = get_monitoring_config("exemplo", "propensao_exemplo", "feature_table")
    predictions = get_monitoring_config("exemplo", "propensao_exemplo", "predictions")

    assert features.schedule_cron == "0 0 7 * * ?"
    assert predictions.schedule_cron == "0 0 8 * * ?"
