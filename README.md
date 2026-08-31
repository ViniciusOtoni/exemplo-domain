# Risco de crédito sobre uma plataforma de ML

Um modelo que estima probabilidade de inadimplência, e o ecossistema que faz esse modelo chegar em produção sem virar projeto de seis meses.

![Endividamento das famílias no Brasil](docs/img/endividamento-familias.png)

Oito em cada dez famílias brasileiras estão endividadas. Em 2015 eram seis.

## O primeiro problema: o crédito piorou, e não foi por desemprego

O mercado de trabalho seguiu aquecido. O que apertou foi o preço da dívida.

| indicador | valor | variação em 12 meses |
|---|---|---|
| comprometimento de renda | 29,7% | +1,9 p.p. |
| endividamento sobre renda | 49,9% | +1,3 p.p. |
| inadimplência das famílias | 5,3% | +1,4 p.p. |
| juros do rotativo | 451,5% ao ano | |

Dados do Banco Central, referência fevereiro e março de 2026, via [Agência Brasil](https://agenciabrasil.ebc.com.br/economia/noticia/2026-04/juros-elevados-mantem-pressao-sobre-endividamento-das-familias).

Nos balanços do segundo trimestre, um banco se descolou dos outros.

![Inadimplência acima de 90 dias por banco, 2T26](docs/img/npl-bancos-2t26.png)

E aqui está a leitura que os números soltos escondem: **NPL estável não é qualidade de crédito estável**. O Bradesco subiu apenas 0,2 p.p. no indicador, e mesmo assim elevou a provisão em 21,4%, para R$ 10,85 bilhões. O estoque de 90 dias é indicador atrasado. A provisão antecipa.

Um modelo de probabilidade de default tenta encurtar essa defasagem mais um passo: estimar quem vai atrasar antes de o atraso existir.

Desde a [Resolução CMN 4.966](https://www.deloitte.com/br/pt/services/audit-assurance/perspectives/resolucao-cmn-4966.html), em vigor desde janeiro de 2025, isso deixou de ser só política de crédito. O banco provisiona por perda esperada, estimada na originação. A PD virou peça contábil.

## O segundo problema: o modelo existe, mas não chega

O cientista de dados entrega um notebook com AUC de 0,78. Entre esse notebook e uma decisão de crédito em produção há um vão que costuma consumir meses:

| etapa | o que normalmente acontece |
|---|---|
| feature store | cada domínio escreve a sua, com regras próprias de janela e chave |
| treino | split aleatório, que vaza o futuro e infla a métrica |
| registro | modelo salvo sem linhagem, ninguém sabe quais features entraram |
| serving | batch e online reimplementados separadamente, com lógicas que divergem |
| monitoramento | notebook agendado que ninguém lê |
| retreino | manual, quando alguém percebe que o modelo caiu |

Nada disso é difícil isoladamente. O custo está em fazer tudo de novo a cada modelo, e em cada reimplementação divergir um pouco da anterior.

## A proposta: o domínio declara, a plataforma executa

Um bundle deste repositório tem quatro coisas. Nenhum notebook, nenhum `databricks.yml`, nenhum script de deploy.

```
features/
├── conf/variables.yml        # 3 linhas
├── pyproject.toml            # dependência e entry point
├── src/credito_features/
│   └── configs.py            # o que é específico do domínio
└── tests/
```

O arquivo de configuração inteiro:

```yaml
component: features
domain_package: credito_features
catalog: workspace
```

Declarar uma feature table é escrever uma função e marcá-la:

```python
@feature_table(
    domain="credito",
    entity_keys=["customer_id"],
    timestamp_key="feature_ts",
    sources=["raw.credito_posicoes"],
    online=True,
)
def perfil_credito_cliente(sources, window):
    ...
```

O resto (bundle DAB, job, schedule, permissões, tabela particionada, chave primária, sincronização com o Lakebase) é gerado pela esteira.

## Como o ecossistema foi montado

![Arquitetura do ecossistema](docs/img/arquitetura.png)

O caminho que o diagrama descreve, com o que cada trecho resolve.

**Feature store.** As features são gravadas por safra, e o `FeatureLookup` resolve ponto no tempo: ao montar o conjunto de treino da safra de março, ele busca a feature vigente em março, não a de hoje. Sem isso o modelo aprende com dado que não existia quando a decisão foi tomada.

**Treino.** O split é temporal, não aleatório. O modelo treina nas safras antigas e é avaliado nas recentes. Cada combinação de hiperparâmetros vira um run aninhado no MLflow, comparável lado a lado. O vencedor é testado uma vez, passa por um gate de sanidade e só então promove o alias `champion`.

**Serving.** O mesmo modelo, com a mesma linhagem, atende os dois modos. Em lote, lendo a tabela Delta. Em tempo real, lendo o Lakebase. O domínio não escreve código para nenhum dos dois.

**Monitoramento.** O Lakehouse Monitoring compara a safra corrente contra a janela em que o modelo foi treinado. A pergunta não é "mudou desde o mês passado", é "afastou-se do que o modelo aprendeu". Quando afasta além do limiar, dispara um retreino no GitHub. O candidato fica registrado e não promovido: alguém precisa aprovar.

## O que muda no ciclo de vida

Os números abaixo saíram da execução real deste repositório, não de estimativa.

| etapa | antes | com a plataforma |
|---|---|---|
| declarar uma feature table | notebook + job + tabela criados à mão | uma função decorada |
| montar conjunto de treino | join manual, risco de vazamento | `FeatureLookup` resolvendo ponto no tempo, 24 safras |
| comparar hiperparâmetros | células soltas no notebook | runs aninhados no MLflow, métrica por combinação |
| promover modelo | alguém move o alias | automático após o gate, ou manual quando o gatilho foi drift |
| servir em lote | script próprio | job gerado, 30 mil clientes pontuados |
| servir em tempo real | endpoint montado à mão | endpoint gerado, resolvendo features pela linhagem |
| detectar drift | ninguém detecta | monitor por safra, com veredito gravado |
| retreinar | manual | disparado por drift, com aprovação humana antes de promover |

## O que isso vale no crédito

**Ponto no tempo evita métrica inflada.** Um modelo que enxerga o futuro entrega AUC alto e falha em produção. O nosso deu 0,7631 no teste, que é a faixa de um modelo de PD honesto. Um número muito acima disso seria sintoma de vazamento, não de qualidade.

**Split temporal mede a perda real.** Da validação (0,7862) para o teste (0,7631) o modelo cai 2,3 pontos. Essa queda é a deterioração da carteira aparecendo na medida. Um split aleatório teria escondido isso e prometido uma performance que não se sustenta.

**Drift com baseline na janela de treino aponta o que mudou.** No ciclo atual, o atraso comportamental foi a variável mais deslocada. Faz sentido de negócio: é a mais próxima do desfecho. Quando ela se move, a inadimplência já está a caminho.

**Retreino com aprovação evita trocar um problema por outro.** Drift diz que o mundo mudou, não que o modelo novo é melhor. Ele pode ter treinado sobre o mesmo dado deslocado e aprendido o deslocamento. Por isso o candidato fica registrado sem servir ninguém até alguém comparar as duas runs e decidir.

Para uma carteira de varejo, encurtar a defasagem entre deterioração e reação significa cobrar antes, renegociar antes e provisionar com estimativa em vez de com o retrovisor.

## O padrão que adotamos

Um framework único, consumido como pacote, com contrato declarativo por componente. O domínio importa `mlplatform`, declara suas configs e não conhece Spark, DABs nem MLflow.

Como outras empresas resolveram o mesmo problema:

| plataforma | abordagem | o que o domínio escreve |
|---|---|---|
| Michelangelo (Uber) | plataforma proprietária completa, com UI própria | configuração na plataforma |
| Metaflow (Netflix) | biblioteca de fluxo em Python, infraestrutura plugável | o fluxo inteiro, como código |
| Bighead (Airbnb) | conjunto de serviços integrados | integração com cada serviço |
| Databricks nativo | Feature Engineering, MLflow, Lakehouse Monitoring, DABs | a cola entre eles |
| este ecossistema | framework único sobre o nativo | só o que é específico do domínio |

A diferença está na última coluna. Metaflow dá liberdade e pede que cada time escreva o fluxo; Michelangelo padroniza e pede que o time se mova dentro da plataforma. Aqui o nativo do Databricks faz o trabalho pesado, e o framework existe para que o domínio não precise escrever a cola entre as peças, que é justamente onde as implementações divergiam.

## Estrutura

Cinco bundles, um por componente, todos consumindo a mesma versão do framework.

```
features/        perfil_credito_cliente, 8 features por safra
training/        pd_inadimplencia, GradientBoosting, split temporal
serving/batch/   scoragem mensal da carteira
serving/online/  endpoint para decisão no momento da solicitação
monitoring/      drift de features e de score
```

A CI valida os cinco em toda abertura de PR: testes, lint, versão e sintaxe do bundle. O merge na `main` deploya.

## Rodando localmente

```bash
cd features   # ou training, serving/batch, serving/online, monitoring
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

O framework vem pinado por URL de release no `pyproject.toml` de cada bundle, o que permite subir um componente sem arrastar os outros.
