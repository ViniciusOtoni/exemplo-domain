from sklearn.ensemble import RandomForestClassifier

from training_platform.contract import FeatureLookupSpec, TrainingConfig, register_training_config

config = TrainingConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    algorithm=RandomForestClassifier,
    hyperparameter_sets=[
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 8},
    ],
    feature_lookups=[
        FeatureLookupSpec(
            table_name="workspace.exemplo_features.customer_transaction_features",
            feature_names=["txn_count", "avg_ticket"],
            lookup_key="customer_id",
            timestamp_lookup_key="reference_date",
        )
    ],
    spine_table="workspace.exemplo.spine_train",
    label_column="label_default",
    reference_date_column="reference_date",
    train_pct=0.6,
    val_pct=0.2,
    test_pct=0.2,
    metric="roc_auc",
    metric_direction="maximize",
)

register_training_config(config)
