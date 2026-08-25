# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering "training-platform @ git+https://github.com/ViniciusOtoni/training-platform@v0.2.0"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")

# COMMAND ----------
# Num job deployado via DAB, o cwd do notebook é .../files/notebooks — a raiz do
# bundle (onde mora `examples/`) não está no sys.path por padrão. training_platform
# já foi instalado no cluster pelo %pip install acima (requirements-dev.txt só
# afeta o venv local de CI/dev, nunca o compute do notebook).
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import examples.training_configs  # noqa: F401
import pyspark.sql.functions as F
import mlflow

from training_platform.contract import get_training_config
from training_platform.split import compute_split_dates

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
config = get_training_config(model_name)

# COMMAND ----------
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

spine = spark.table(config.spine_table)

fe = FeatureEngineeringClient()
feature_lookups = [
    FeatureLookup(
        table_name=fl.table_name,
        feature_names=fl.feature_names,
        lookup_key=fl.lookup_key,
        timestamp_lookup_key=fl.timestamp_lookup_key,
    )
    for fl in config.feature_lookups
]
training_set = fe.create_training_set(
    df=spine,
    feature_lookups=feature_lookups,
    label=config.label_column,
)
master = training_set.load_df()

# COMMAND ----------
distinct_dates = [
    row[config.reference_date_column]
    for row in spine.select(config.reference_date_column).distinct().collect()
]
train_end, val_end = compute_split_dates(distinct_dates, config.train_pct, config.val_pct, config.test_pct)
window_start = min(distinct_dates)
window_end = max(distinct_dates)

master_with_split = master.withColumn(
    "_split",
    F.when(F.col(config.reference_date_column) <= F.lit(train_end), "train")
    .when(F.col(config.reference_date_column) <= F.lit(val_end), "val")
    .otherwise("test"),
)

scratch_schema = f"{catalog}.training_scratch"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {scratch_schema}")
scratch_prefix = f"{scratch_schema}.{model_name}"
for split_name in ["train", "val", "test"]:
    (
        master_with_split.filter(F.col("_split") == split_name)
        .drop("_split", config.reference_date_column)
        .write.format("delta")
        .mode("overwrite")
        .saveAsTable(f"{scratch_prefix}_{split_name}")
    )

# COMMAND ----------
from databricks.sdk import WorkspaceClient

WorkspaceClient().workspace.mkdirs(f"/Shared/training-platform/{config.domain}")
mlflow.set_experiment(f"/Shared/training-platform/{config.domain}/{config.model_name}")
run = mlflow.start_run()
mlflow.log_params({"train_pct": config.train_pct, "val_pct": config.val_pct, "test_pct": config.test_pct})
mlflow.end_run()

dbutils.jobs.taskValues.set("mlflow_run_id", run.info.run_id)
dbutils.jobs.taskValues.set("window_start", window_start.isoformat())
dbutils.jobs.taskValues.set("window_end", window_end.isoformat())
