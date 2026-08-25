# Databricks notebook source
# MAGIC %pip install "feature-platform @ git+https://github.com/ViniciusOtoni/feature-platform@v0.2.0"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("feature_table", "")
dbutils.widgets.text("mode", "incremental")
dbutils.widgets.text("start_date", "")
dbutils.widgets.text("end_date", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")
dbutils.widgets.text("database_instance_name", "")

# COMMAND ----------
# Num job deployado via DAB, o cwd do notebook é .../files/notebooks — a raiz do
# bundle (onde mora `examples/`) não está no sys.path por padrão. feature_platform
# já foi instalado no cluster pelo %pip install acima (requirements-dev.txt só
# afeta o venv local de CI/dev, nunca o compute do notebook).
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import examples.features  # noqa: F401  (import dispara o registro via decorator)
from datetime import date

from feature_platform.contract import get_registry
from feature_platform.writer import WriteMode
from feature_platform.engine import run_feature_table

# COMMAND ----------
feature_table_name = dbutils.widgets.get("feature_table")
mode = WriteMode(dbutils.widgets.get("mode"))
start_date = dbutils.widgets.get("start_date") or None
end_date = dbutils.widgets.get("end_date") or None
catalog = dbutils.widgets.get("catalog")
git_commit = dbutils.widgets.get("git_commit")
git_branch = dbutils.widgets.get("git_branch")
database_instance_name = dbutils.widgets.get("database_instance_name")
# currentRunId() não está na whitelist do Py4J em compute serverless/shared access
# mode — levanta Py4JSecurityException. Cai para um id gerado localmente quando o
# contexto de job não expõe o run id dessa forma.
try:
    run_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
except Exception:
    import uuid

    run_id = str(uuid.uuid4())

registry = get_registry()
spec = registry[feature_table_name]

# COMMAND ----------
run_feature_table(
    spec=spec,
    spark=spark,
    catalog=catalog,
    mode=mode,
    today=date.today(),
    run_id=run_id,
    git_commit=git_commit,
    git_branch=git_branch,
    backfill_start=start_date,
    backfill_end=end_date,
    database_instance_name=database_instance_name,
)
