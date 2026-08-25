# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering "training-platform @ git+https://github.com/ViniciusOtoni/training-platform@v0.2.0"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

# COMMAND ----------
# Num job deployado via DAB, o cwd do notebook é .../files/notebooks — a raiz do
# bundle (onde mora `examples/`) não está no sys.path por padrão. training_platform
# já foi instalado no cluster pelo %pip install acima.
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import examples.training_configs  # noqa: F401
import json
from datetime import date, datetime
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

import training_platform
from training_platform.contract import get_training_config
from training_platform.pipeline import build_pipeline
from training_platform.pyfunc_model import FeaturePlatformModel
from training_platform.naming import derive_model_name, validate_model_name
from training_platform.audit import RunRecord, write_run

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
git_commit = dbutils.widgets.get("git_commit")
git_branch = dbutils.widgets.get("git_branch")
config = get_training_config(model_name)

mlflow_run_id = dbutils.jobs.taskValues.get(taskKey="prepare_training_set", key="mlflow_run_id")
window_start = dbutils.jobs.taskValues.get(taskKey="prepare_training_set", key="window_start")
window_end = dbutils.jobs.taskValues.get(taskKey="prepare_training_set", key="window_end")
best_hyperparameters = json.loads(
    dbutils.jobs.taskValues.get(taskKey="select_best_and_test", key="best_hyperparameters")
)
try:
    run_id_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
except Exception:
    import uuid

    run_id_job = str(uuid.uuid4())

# COMMAND ----------
scratch_prefix = f"{catalog}.training_scratch.{model_name}"
train_df = spark.table(f"{scratch_prefix}_train").toPandas()
feature_cols = [c for c in train_df.columns if c not in {config.label_column, *[fl.lookup_key for fl in config.feature_lookups]}]
X_train, y_train = train_df[feature_cols], train_df[config.label_column]

estimator = config.algorithm(**best_hyperparameters)
pipeline = build_pipeline(config.custom_transforms, estimator)
pipeline.fit(X_train, y_train)

pyfunc_class = config.pyfunc_model_class or FeaturePlatformModel
wrapped_model = pyfunc_class(pipeline)

# COMMAND ----------
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
spine = spark.table(config.spine_table)
training_set = fe.create_training_set(
    df=spine,
    feature_lookups=feature_lookups,
    label=config.label_column,
)

full_model_name = derive_model_name(catalog, config.domain, config.model_name)
validate_model_name(full_model_name)
mlflow.set_registry_uri("databricks-uc")

model_schema = full_model_name.rsplit(".", 1)[0]
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {model_schema}")

with mlflow.start_run(run_id=mlflow_run_id):
    fe.log_model(
        model=wrapped_model,
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=full_model_name,
        code_paths=[os.path.dirname(training_platform.__file__)],
    )
    mlflow.set_tag("git_commit", git_commit)
    mlflow.set_tag("git_branch", git_branch)

# COMMAND ----------
write_run(
    spark,
    RunRecord(
        component="training",
        entity_name=full_model_name,
        git_commit=git_commit,
        git_branch=git_branch,
        run_id=run_id_job,
        mode="train",
        status="SUCCESS",
        window_start=date.fromisoformat(window_start),
        window_end=date.fromisoformat(window_end),
        run_ts=datetime.utcnow(),
    ),
)
