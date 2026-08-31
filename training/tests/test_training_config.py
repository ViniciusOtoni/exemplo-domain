import importlib

import pytest

from mlplatform.training.contract import clear_registry, get_training_config


@pytest.fixture
def config():
    """O registro acontece no efeito colateral do import.

    Sem o reload, o segundo teste que usasse esta fixture veria o registry
    vazio: o primeiro já importou o módulo, o teardown limpou o registry, e um
    `import` seguinte é no-op porque o módulo está em sys.modules. Num job real
    isso não aparece — cada processo importa uma vez só.
    """
    import credito_training.configs as configs

    # A ordem importa: o `import` já registra na primeira vez que roda, então
    # limpar só ANTES deixaria o reload registrando em cima e levantando
    # "already registered".
    clear_registry()
    importlib.reload(configs)

    yield get_training_config("pd_inadimplencia")
    clear_registry()


def test_the_model_is_registered_with_point_in_time_lookups(config):
    assert config.domain == "credito"
    assert config.label_column == "label_default"
    assert config.feature_lookups[0].table_name == "workspace.credito_features.perfil_credito_cliente"
    assert abs(config.train_pct + config.val_pct + config.test_pct - 1.0) < 1e-9


def test_the_lookup_resolves_at_the_safra_not_at_today(config):
    """Sem `timestamp_lookup_key`, o lookup traria a feature CORRENTE para
    rotular uma safra antiga — o modelo aprenderia com dado que não existia
    quando a decisão foi tomada."""
    assert config.feature_lookups[0].timestamp_lookup_key == "reference_date"


def test_the_metric_survives_class_imbalance(config):
    """Com 6% a 15% de default, acurácia premia o modelo que responde sempre
    'adimplente' — acerta 90% e não serve para decidir crédito."""
    assert config.metric == "roc_auc"


def test_every_combination_is_seeded(config):
    """Sem random_state, duas execuções com os mesmos hiperparâmetros produzem
    modelos diferentes — e reexecutar o pipeline deixa de ser reprodutível."""
    assert all("random_state" in h for h in config.hyperparameter_sets)
