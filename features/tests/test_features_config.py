import pytest

from mlplatform.features.contract import clear_registry, get_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_customer_transaction_features_is_registered_online():
    import exemplo_features.configs  # noqa: F401  (dispara o registro via decorator)

    registry = get_registry()
    spec = registry["customer_transaction_features"]

    assert spec.domain == "exemplo"
    assert spec.entity_keys == ["customer_id"]
    assert spec.timestamp_key == "feature_ts"
    assert spec.online is True
