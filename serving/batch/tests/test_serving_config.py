import importlib

import pytest

from mlplatform.serving.contract import BatchServingConfig, clear_registry, get_serving_config


@pytest.fixture(autouse=True)
def _registered():
    """Recarrega o módulo de configs a cada teste.

    Importar de novo não basta: o módulo já está em `sys.modules` e o import
    vira no-op, então o registro que o `clear_registry` esvaziou nunca é
    refeito. Com um teste só o problema fica escondido; o segundo o revela.
    """
    import credito_serving_batch.configs as configs

    clear_registry()
    importlib.reload(configs)
    yield
    clear_registry()


def test_the_model_is_registered_as_batch():

    config = get_serving_config("pd_inadimplencia")

    assert isinstance(config, BatchServingConfig)
    assert config.spine_inference_table == "workspace.credito.spine_inference"
    assert config.schedule_cron == "0 0 6 * * ?"


def test_the_output_carries_the_features_the_monitoring_will_read():
    """O drift de dados compara a distribuição das features entre safras. Sem
    elas na tabela de saída, comparar exigiria reconstruir o join do
    FeatureLookup a posteriori, contra feature tables que já mudaram."""

    output = get_serving_config("pd_inadimplencia").output

    assert "utilizacao_limite" in output.feature_cols
    assert "comprometimento_renda" in output.feature_cols
    assert output.ts_date == "reference_date"


def test_the_label_is_declared_but_never_required():
    """O desfecho só se materializa 6 meses depois da inferência. Declará-lo diz
    ao monitoramento ONDE ele vai aparecer; exigi-lo reprovaria toda execução."""

    output = get_serving_config("pd_inadimplencia").output

    assert output.label_col == "label_default"
    assert "label_default" not in output.required_columns
