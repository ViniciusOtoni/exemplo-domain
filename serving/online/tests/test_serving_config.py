import pytest

from serving_platform.contract import clear_registry, get_serving_config


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_propensao_exemplo_online_config_is_registered():
    import exemplo_serving_online.serving_configs  # noqa: F401  (dispara o registro)

    config = get_serving_config("propensao_exemplo")

    assert config.mode == "online"
    assert config.alias == "champion"
