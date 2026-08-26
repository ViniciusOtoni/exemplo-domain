# Databricks notebook source
# Dependências resolvidas pelo Environment nativo do serverless (ver
# databricks.yml/scripts/generate_resources.py) -- sem %pip install nem
# sys.path hack aqui.
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

# COMMAND ----------
import exemplo_serving_batch.serving_configs  # noqa: F401  (dispara o registro)
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
