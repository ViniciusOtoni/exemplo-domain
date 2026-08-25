# Databricks notebook source
# MAGIC %pip install "training-platform @ git+https://github.com/ViniciusOtoni/training-platform@v0.2.0"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")

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
import mlflow
from sklearn.metrics import get_scorer

from training_platform.contract import get_training_config
from training_platform.pipeline import build_pipeline

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
config = get_training_config(model_name)

mlflow_run_id = dbutils.jobs.taskValues.get(taskKey="prepare_training_set", key="mlflow_run_id")
mlflow.set_experiment(f"/Shared/training-platform/{config.domain}/{config.model_name}")

# COMMAND ----------
scratch_prefix = f"{catalog}.training_scratch.{model_name}"
train_df = spark.table(f"{scratch_prefix}_train").toPandas()
val_df = spark.table(f"{scratch_prefix}_val").toPandas()

feature_cols = [c for c in train_df.columns if c not in {config.label_column, *[fl.lookup_key for fl in config.feature_lookups]}]
X_train, y_train = train_df[feature_cols], train_df[config.label_column]
X_val, y_val = val_df[feature_cols], val_df[config.label_column]

scorer = get_scorer(config.metric) if isinstance(config.metric, str) else config.metric
metric_name = config.metric if isinstance(config.metric, str) else "custom_metric"

# COMMAND ----------
results = []
with mlflow.start_run(run_id=mlflow_run_id):
    for i, hyperparams in enumerate(config.hyperparameter_sets):
        with mlflow.start_run(run_name=f"combo_{i}", nested=True):
            estimator = config.algorithm(**hyperparams)
            pipeline = build_pipeline(config.custom_transforms, estimator)
            pipeline.fit(X_train, y_train)
            if len(X_val) > 0:
                metric_value = float(scorer(pipeline, X_val, y_val))
            else:
                metric_value = float("nan")
            mlflow.log_params(hyperparams)
            mlflow.log_metric(metric_name, metric_value)
            results.append({"hyperparameters": hyperparams, "metric": metric_value})

dbutils.jobs.taskValues.set("hyperparameter_results", json.dumps(results))
