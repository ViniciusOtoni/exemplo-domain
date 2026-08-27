import pytest

from mlplatform.serving.contract import BatchServingConfig, clear_registry, get_serving_config


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_propensao_exemplo_is_registered_as_batch():
    import exemplo_serving_batch.configs  # noqa: F401  (dispara o registro)

    config = get_serving_config("propensao_exemplo")

    assert isinstance(config, BatchServingConfig)
    assert config.spine_inference_table == "workspace.exemplo.spine_inference"
    assert config.schedule_cron == "0 0 6 * * ?"
