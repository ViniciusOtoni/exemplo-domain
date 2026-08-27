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
    import exemplo_training.configs as configs

    # A ordem importa: o `import` já registra na primeira vez que roda, então
    # limpar só ANTES deixaria o reload registrando em cima e levantando
    # "already registered". Limpar entre o import e o reload garante exatamente
    # um registro, valendo tanto na primeira execução quanto nas seguintes.
    clear_registry()
    importlib.reload(configs)

    yield get_training_config("propensao_exemplo")
    clear_registry()


def test_propensao_exemplo_is_registered_with_feature_lookups(config):
    assert config.domain == "exemplo"
    assert config.label_column == "label_default"
    assert config.feature_lookups[0].table_name == "workspace.exemplo_features.customer_transaction_features"
    assert abs(config.train_pct + config.val_pct + config.test_pct - 1.0) < 1e-9


def test_every_combination_is_seeded(config):
    """Sem random_state, duas execuções com os mesmos hiperparâmetros produzem
    modelos diferentes — e reexecutar o pipeline deixa de ser reprodutível."""
    assert all("random_state" in h for h in config.hyperparameter_sets)
