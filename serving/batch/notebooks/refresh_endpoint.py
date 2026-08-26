# Databricks notebook source
# Dependências resolvidas pelo Environment nativo do serverless (ver
# databricks.yml/scripts/generate_resources.py) -- sem %pip install nem
# sys.path hack aqui.
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")

# COMMAND ----------
import exemplo_serving_batch.serving_configs  # noqa: F401  (dispara o registro)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

from serving_platform.contract import get_serving_config
from serving_platform.naming import derive_endpoint_name

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
config = get_serving_config(model_name)

full_model_name = f"{catalog}.{config.domain}_models.{model_name}"
endpoint_name = derive_endpoint_name(config.domain, model_name)

# COMMAND ----------
client = WorkspaceClient()
client.serving_endpoints.update_config_and_wait(
    name=endpoint_name,
    served_entities=[
        ServedEntityInput(
            name=model_name,
            entity_name=f"{full_model_name}@{config.alias}",
            scale_to_zero_enabled=True,
            workload_size="Small",
        )
    ],
)
print(f"endpoint '{endpoint_name}' updated to current '{config.alias}' resolution")
