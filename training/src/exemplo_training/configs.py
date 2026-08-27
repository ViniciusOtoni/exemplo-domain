from sklearn.ensemble import RandomForestClassifier

from mlplatform.training.contract import FeatureLookupSpec, TrainingConfig, register_training_config

config = TrainingConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    algorithm=RandomForestClassifier,
    # random_state fixo: sem ele o RandomForest fita diferente a cada execução,
    # e duas execuções com os mesmos hiperparâmetros produziam modelos distintos.
    # O framework já garante que o modelo avaliado é o registrado; isto garante
    # que reexecutar o pipeline reproduz o mesmo modelo.
    hyperparameter_sets=[
        {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        {"n_estimators": 200, "max_depth": 8, "random_state": 42},
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
