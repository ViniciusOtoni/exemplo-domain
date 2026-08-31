import importlib

import pytest

from mlplatform.serving.contract import OnlineServingConfig, clear_registry, get_serving_config


@pytest.fixture(autouse=True)
def _registered():
    """Recarrega o módulo de configs a cada teste.

    Importar de novo não basta: o módulo já está em `sys.modules` e o import
    vira no-op, então o registro que o `clear_registry` esvaziou nunca é
    refeito. Com um teste só o problema fica escondido; o segundo o revela.
    """
    import credito_serving_online.configs as configs

    clear_registry()
    importlib.reload(configs)
    yield
    clear_registry()


def test_the_model_is_registered_as_online():

    config = get_serving_config("pd_inadimplencia")

    assert isinstance(config, OnlineServingConfig)
    assert config.alias == "champion"
