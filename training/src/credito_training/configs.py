"""Modelo de probabilidade de default PF.

Prevê se o cliente entra em 90+ dias de atraso na janela de performance de 6
meses após a safra. Não é só política de crédito: desde a Resolução CMN 4.966,
em vigor a partir de 1º/1/2025, o banco provisiona por PERDA ESPERADA — estima
na originação quanto vai perder ao longo da vida do ativo. A probabilidade de
default virou peça contábil.

O split é temporal por construção: treino nas safras antigas, teste nas
recentes. Split aleatório aqui seria vazamento — o modelo veria o futuro de um
cliente ao ser avaliado no passado dele.
"""

from sklearn.ensemble import GradientBoostingClassifier

from mlplatform.training.contract import FeatureLookupSpec, TrainingConfig, register_training_config

FEATURES = [
    "utilizacao_limite",
    "comprometimento_renda",
    "share_rotativo",
    "atraso_max_6m",
    "renda_mensal",
    "tempo_relacionamento_meses",
    "n_produtos",
    "saldo_devedor_total",
]

config = TrainingConfig(
    domain="credito",
    model_name="pd_inadimplencia",
    # Gradient boosting em vez de random forest: com variáveis contínuas
    # correlacionadas — utilização e comprometimento andam juntas — o boosting
    # separa melhor a faixa de risco alto, que é onde a decisão de crédito
    # acontece. A floresta tende a suavizar demais a cauda.
    algorithm=GradientBoostingClassifier,
    # random_state fixo: sem ele duas execuções com os mesmos hiperparâmetros
    # produzem modelos distintos, e reexecutar o pipeline deixa de reproduzir.
    hyperparameter_sets=[
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "random_state": 42},
    ],
    feature_lookups=[
        FeatureLookupSpec(
            table_name="workspace.credito_features.perfil_credito_cliente",
            feature_names=FEATURES,
            lookup_key="customer_id",
            # A resolução é point-in-time: a spine traz a safra, e o lookup
            # busca a feature VIGENTE naquela safra. É o que impede o modelo de
            # aprender com dado que não existia quando a decisão foi tomada.
            timestamp_lookup_key="reference_date",
        )
    ],
    # Só safras com janela de performance fechada. As 6 mais recentes ficam
    # fora: rotulá-las hoje marcaria como adimplente quem apenas ainda não teve
    # tempo de atrasar.
    spine_table="workspace.credito.spine_train",
    label_column="label_default",
    reference_date_column="reference_date",
    train_pct=0.6,
    val_pct=0.2,
    test_pct=0.2,
    # ROC-AUC, e não acurácia: com ~6% a ~15% de default, um modelo que responde
    # sempre "adimplente" acerta 90% e não serve para nada.
    metric="roc_auc",
    metric_direction="maximize",
)

register_training_config(config)
