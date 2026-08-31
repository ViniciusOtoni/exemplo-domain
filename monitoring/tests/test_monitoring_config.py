import importlib

import pytest

from mlplatform.monitoring.contract import clear_registry, get_monitoring_config
from mlplatform.monitoring.metrics import resolve


@pytest.fixture(autouse=True)
def _registered():
    """Recarrega o módulo de configs a cada teste.

    Importar de novo não basta: o módulo já está em `sys.modules` e o import
    vira no-op, então o registro que o `clear_registry` esvaziou nunca é
    refeito. O segundo teste do arquivo é que revela isso.
    """
    import credito_monitoring.configs as configs

    clear_registry()
    importlib.reload(configs)
    yield
    clear_registry()


def _features():
    return get_monitoring_config("credito", "pd_inadimplencia", "feature_table")


def _predictions():
    return get_monitoring_config("credito", "pd_inadimplencia", "predictions")


def test_both_targets_are_registered():
    assert _features().target_table == "workspace.credito_features.perfil_credito_cliente"
    assert _predictions().target_table == "workspace.credito_predictions.pd_inadimplencia"


def test_the_watched_columns_are_the_ones_the_cycle_is_moving():
    """Utilização de limite e comprometimento de renda são as variáveis que o
    ciclo de crédito está deslocando — comprometimento subiu 1,7 p.p. em 12
    meses. Monitorar outra coisa mediria ruído."""
    colunas = _features().columns

    assert "utilizacao_limite" in colunas
    assert "comprometimento_renda" in colunas


def test_the_metric_applies_to_numeric_columns():
    """As métricas limitadas a [0,1] — js_distance e afins — só são calculadas
    para colunas CATEGÓRICAS. Escolher uma delas aqui produziria nulo em toda
    medição, que é "sem drift" para sempre."""
    assert not resolve(_features().drift_metric).bounded
    assert _features().drift_metric == "population_stability_index"


def test_only_the_feature_table_compares_against_the_training_window():
    """A tabela de predições não tem baseline possível: não existem predições do
    período de treino — o modelo ainda não existia."""
    assert _features().baseline_timestamp_column == "feature_ts"
    assert _predictions().baseline_timestamp_column is None


def test_features_are_checked_before_predictions():
    """Deslocamento de score só se investiga depois de saber se a entrada se
    deslocou; os dois horários codificam essa ordem."""
    assert _features().schedule_cron == "0 0 7 * * ?"
    assert _predictions().schedule_cron == "0 0 8 * * ?"
