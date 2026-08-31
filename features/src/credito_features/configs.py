"""Features de risco de crédito PF, por safra.

O que pressiona a inadimplência hoje não é desemprego — o mercado de trabalho
segue aquecido — é o preço da dívida. O comprometimento de renda das famílias
está em 28,9% (+1,7 p.p. em 12 meses) e os juros do rotativo em 15,1% ao mês.
Por isso as variáveis que carregam sinal aqui são utilização de limite,
comprometimento de renda e participação do rotativo.

Cada execução produz UMA safra: o `feature_ts` é o fim da janela. É o que
permite ao `FeatureLookup` com `timestamp_lookup_key` resolver ponto-no-tempo —
usar a posição de hoje para rotular o passado seria vazamento.
"""

import pyspark.sql.functions as F
from pyspark.sql import Window

from mlplatform.features.contract import feature_table

# Meses de histórico usados nas features comportamentais. O atraso máximo em 6
# meses distingue quem teve um tropeço pontual de quem está em deterioração —
# a posição do mês sozinha não faz essa distinção.
JANELA_COMPORTAMENTAL = 6


@feature_table(
    domain="credito",
    entity_keys=["customer_id"],
    timestamp_key="feature_ts",
    sources=["raw.credito_posicoes"],
    online=True,
)
def perfil_credito_cliente(sources, window):
    posicoes = sources["raw.credito_posicoes"]

    # A janela comportamental olha PARA TRÁS a partir do fim da safra. Ler
    # além da janela de escrita é deliberado: a feature da safra M resume os
    # 6 meses até M, e não só o mês M.
    inicio_comportamental = F.add_months(F.lit(window.end), -JANELA_COMPORTAMENTAL)
    historico = posicoes.filter(
        (F.col("posicao_ts") > inicio_comportamental) & (F.col("posicao_ts") <= F.lit(window.end))
    )

    comportamento = historico.groupBy("customer_id").agg(
        F.max("dias_atraso").alias("atraso_max_6m"),
    )

    # A posição corrente é a do fim da janela — a foto do cliente na safra.
    ultima = Window.partitionBy("customer_id").orderBy(F.col("posicao_ts").desc())
    corrente = (
        historico.withColumn("_rn", F.row_number().over(ultima))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    return (
        corrente.join(comportamento, "customer_id")
        .select(
            "customer_id",
            "segmento",
            F.col("renda_mensal"),
            F.col("tempo_relacionamento_meses"),
            F.col("n_produtos"),
            F.col("atraso_max_6m"),
            # Utilização do limite: o sinal antecedente mais forte de estresse.
            # Sobe antes do atraso aparecer.
            F.round(F.col("saldo_utilizado") / F.col("limite_cartao"), 4).alias("utilizacao_limite"),
            # Comprometimento de renda: a métrica central do ciclo atual.
            F.round(F.col("parcela_mensal") / F.col("renda_mensal"), 4).alias("comprometimento_renda"),
            # Participação do rotativo no saldo. É a tese que separa os balanços
            # do 2T26: quem tem mix garantido e alta renda segurou a qualidade
            # de crédito; quem tem massificado sem garantia, não.
            F.round(
                F.col("saldo_rotativo") / F.when(F.col("saldo_utilizado") > 0, F.col("saldo_utilizado")),
                4,
            ).alias("share_rotativo"),
            F.col("saldo_utilizado").alias("saldo_devedor_total"),
        )
        .withColumn("feature_ts", F.lit(window.end))
    )
