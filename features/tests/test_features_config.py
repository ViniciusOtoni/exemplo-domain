import importlib

import pytest

from mlplatform.features.contract import clear_registry, get_registry


@pytest.fixture(autouse=True)
def _registered():
    """Recarrega o módulo de configs a cada teste.

    Importar de novo não basta: o módulo já está em `sys.modules` e o import
    vira no-op, então o registro que o `clear_registry` esvaziou nunca é
    refeito. Com um teste só o problema fica escondido; o segundo o revela.
    """
    import credito_features.configs as configs

    clear_registry()
    importlib.reload(configs)
    yield
    clear_registry()


def test_perfil_credito_cliente_is_registered_online():

    spec = get_registry()["perfil_credito_cliente"]

    assert spec.domain == "credito"
    assert spec.entity_keys == ["customer_id"]
    assert spec.timestamp_key == "feature_ts"
    assert spec.online is True


def test_the_backfill_partitions_by_safra_not_by_customer():
    """Particionado por `customer_id`, cada janela de backfill substituía a
    partição inteira do cliente e apagava as safras anteriores dele — a feature
    store ficava sem histórico, e o lookup point-in-time sem sentido.

    Num domínio de crédito isso é fatal: safra É o conceito."""

    spec = get_registry()["perfil_credito_cliente"]

    assert spec.partition_cols() == ["feature_ts"]
