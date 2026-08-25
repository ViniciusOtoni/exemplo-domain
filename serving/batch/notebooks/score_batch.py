# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering "mlflow>=3.15.0" "serving-platform @ git+https://github.com/ViniciusOtoni/serving-platform@v0.1.0"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

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
from datetime import date, datetime

import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient

from serving_platform.contract import get_serving_config
from serving_platform.naming import derive_predictions_table_name
from serving_platform.quality import run_predictions_gate, gate_passed
from serving_platform.audit import RunRecord, write_run

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
git_commit = dbutils.widgets.get("git_commit")
git_branch = dbutils.widgets.get("git_branch")
config = get_serving_config(model_name)
try:
    run_id_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
except Exception:
    import uuid

    run_id_job = str(uuid.uuid4())

# COMMAND ----------
full_model_name = f"{catalog}.{config.domain}_models.{model_name}"
spine = spark.table(config.spine_inference_table)
input_row_count = spine.count()

fe = FeatureEngineeringClient()
predictions_df = fe.score_batch(
    model_uri=f"models:/{full_model_name}@{config.alias}",
    df=spine,
    result_type="double",
).withColumn("scored_at", F.current_timestamp())

# COMMAND ----------
prediction_column = "prediction"
predictions_pd = predictions_df.toPandas()
findings = run_predictions_gate(predictions_pd, prediction_column, input_row_count)
passed = gate_passed(findings)
predictions_table = derive_predictions_table_name(catalog, config.domain, model_name)

if not passed:
    write_run(
        spark,
        RunRecord(
            component="serving",
            entity_name=predictions_table,
            git_commit=git_commit,
            git_branch=git_branch,
            run_id=run_id_job,
            mode="batch",
            status="FAILED",
            window_start=date.today(),
            window_end=date.today(),
            run_ts=datetime.utcnow(),
        ),
    )
    failed_checks = [f.check for f in findings if f.status == "FAIL"]
    raise ValueError(f"predictions quality gate failed: {failed_checks}")

predictions_schema = predictions_table.rsplit(".", 1)[0]
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {predictions_schema}")
predictions_df.write.format("delta").mode("append").saveAsTable(predictions_table)

write_run(
    spark,
    RunRecord(
        component="serving",
        entity_name=predictions_table,
        git_commit=git_commit,
        git_branch=git_branch,
        run_id=run_id_job,
        mode="batch",
        status="SUCCESS",
        window_start=date.today(),
        window_end=date.today(),
        run_ts=datetime.utcnow(),
    ),
)
