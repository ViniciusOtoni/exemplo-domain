# Databricks notebook source
# MAGIC %pip install "serving-platform @ git+https://github.com/ViniciusOtoni/serving-platform@v0.1.0"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")

# COMMAND ----------
# Num job deployado via DAB, o cwd do notebook é .../files/notebooks — a raiz do
# bundle (onde mora `examples/`) não está no sys.path por padrão. serving_platform
# já foi instalado no cluster pelo %pip install acima.
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import examples.serving_configs  # noqa: F401
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
