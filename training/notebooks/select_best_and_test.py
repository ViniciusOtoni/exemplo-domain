# Databricks notebook source
# Dependências resolvidas pelo Environment nativo do serverless (ver
# databricks.yml/resources/training_pipeline.job.yml) -- sem %pip install nem
# sys.path hack aqui.
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

# COMMAND ----------
import exemplo_training.training_configs  # noqa: F401  (dispara o registro)
import json
from datetime import date, datetime
import mlflow
from sklearn.metrics import get_scorer

from training_platform.contract import get_training_config
from training_platform.pipeline import build_pipeline
from training_platform.selection import select_best
from training_platform.quality import run_sanity_gate, gate_passed
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
results = json.loads(
    dbutils.jobs.taskValues.get(taskKey="fit_and_compare_hyperparams", key="hyperparameter_results")
)
# currentRunId() levanta Py4JSecurityException em compute serverless/shared access
# mode — cai para um id gerado localmente quando o contexto de job não expõe o run id.
try:
    run_id_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
except Exception:
    import uuid

    run_id_job = str(uuid.uuid4())

# COMMAND ----------
best_hyperparameters = select_best(
    [(r["hyperparameters"], r["metric"]) for r in results], config.metric_direction
)

scratch_prefix = f"{catalog}.training_scratch.{model_name}"
train_df = spark.table(f"{scratch_prefix}_train").toPandas()
test_df = spark.table(f"{scratch_prefix}_test").toPandas()

feature_cols = [c for c in train_df.columns if c not in {config.label_column, *[fl.lookup_key for fl in config.feature_lookups]}]
X_train, y_train = train_df[feature_cols], train_df[config.label_column]
X_test, y_test = test_df[feature_cols], test_df[config.label_column]

scorer = get_scorer(config.metric) if isinstance(config.metric, str) else config.metric
metric_name = config.metric if isinstance(config.metric, str) else "custom_metric"

estimator = config.algorithm(**best_hyperparameters)
pipeline = build_pipeline(config.custom_transforms, estimator)
pipeline.fit(X_train, y_train)
if len(X_test) > 0:
    test_metric = float(scorer(pipeline, X_test, y_test))
else:
    test_metric = float("nan")

findings = run_sanity_gate(test_metric, num_predictions=len(X_test))
passed = gate_passed(findings)

# COMMAND ----------
with mlflow.start_run(run_id=mlflow_run_id):
    mlflow.log_params({f"best__{k}": v for k, v in best_hyperparameters.items()})
    mlflow.log_metric(f"test_{metric_name}", test_metric)

if not passed:
    write_run(
        spark,
        RunRecord(
            component="training",
            entity_name=model_name,
            git_commit=git_commit,
            git_branch=git_branch,
            run_id=run_id_job,
            mode="train",
            status="FAILED",
            window_start=date.fromisoformat(window_start),
            window_end=date.fromisoformat(window_end),
            run_ts=datetime.utcnow(),
        ),
    )
    failed_checks = [f.check for f in findings if f.status == "FAIL"]
    raise ValueError(f"sanity gate failed: {failed_checks}")

dbutils.jobs.taskValues.set("best_hyperparameters", json.dumps(best_hyperparameters))
