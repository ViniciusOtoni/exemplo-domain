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


def test_the_metric_applies_to_the_column_types_being_watched():
    """As colunas observadas são numéricas. As métricas limitadas a [0,1] —
    js_distance, tv_distance, l_infinity_distance — só são calculadas para
    colunas CATEGÓRICAS: escolher uma delas aqui produziria nulo em toda
    medição, que é "sem drift" para sempre."""
    from mlplatform.monitoring.metrics import resolve

    features = get_monitoring_config("exemplo", "propensao_exemplo", "feature_table")

    assert not resolve(features.drift_metric).bounded, "métrica limitada a [0,1] não serve para numérica"
    assert features.drift_metric == "population_stability_index"


def test_the_threshold_follows_the_psi_convention():
    """PSI não tem teto, então o limiar não vem da escala: vem da convenção
    estabelecida (>0.25 significativo)."""
    features = get_monitoring_config("exemplo", "propensao_exemplo", "feature_table")
    predictions = get_monitoring_config("exemplo", "propensao_exemplo", "predictions")

    assert features.threshold == 0.25
    assert predictions.threshold == 0.25


def test_only_the_feature_table_compares_against_the_training_window():
    """A tabela de predições não tem baseline possível: não existem predições do
    período de treino — o modelo ainda não existia. Declarar uma ali faria o
    monitor comparar contra o vazio."""
    features = get_monitoring_config("exemplo", "propensao_exemplo", "feature_table")
    predictions = get_monitoring_config("exemplo", "propensao_exemplo", "predictions")

    assert features.baseline_timestamp_column == "feature_ts"
    assert predictions.baseline_timestamp_column is None
