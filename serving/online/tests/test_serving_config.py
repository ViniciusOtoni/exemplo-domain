import pytest

from mlplatform.serving.contract import OnlineServingConfig, clear_registry, get_serving_config


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_propensao_exemplo_is_registered_as_online():
    import exemplo_serving_online.configs  # noqa: F401  (dispara o registro)

    config = get_serving_config("propensao_exemplo")

    assert isinstance(config, OnlineServingConfig)
    assert config.alias == "champion"
