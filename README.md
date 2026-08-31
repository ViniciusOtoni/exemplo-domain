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

## O segundo problema: o modelo existe, mas não chega

O cientista de dados dá conta da parte difícil. Ele entende a regra de negócio, conversa com a área de crédito, escolhe as variáveis que fazem sentido e treina um modelo que funciona.

O problema começa depois. Sem uma plataforma madura por baixo, esse modelo pode levar meses até servir alguém, e às vezes não chega. Cada etapa vira um projeto próprio:

| etapa | o que normalmente acontece |
|---|---|
| feature store | a mesma feature é recalculada em vários lugares, e os valores não batem entre si |
| treino | cada pessoa treina do seu jeito, sem um padrão de MLOps que valha para todo mundo |
| registro | modelo salvo sem linhagem, ninguém sabe quais features entraram |
| serving | sem uma feature store madura por trás, batch e online são construídos separados e passam a divergir |
| monitoramento e retreino | ninguém enxerga o drift de forma tangível, e cada safra nova levanta a dúvida de retreinar sem resposta objetiva |

Nada disso é difícil isoladamente. O custo está em fazer tudo de novo a cada modelo, e em cada reimplementação divergir um pouco da anterior.

## A proposta: o domínio declara, a plataforma executa

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

### O padrão por trás dessa escolha

Nenhuma dessas decisões é original. Toda empresa que colocou ML em escala esbarrou no mesmo gargalo e construiu alguma camada para resolvê-lo.

| plataforma | abordagem | o que o domínio escreve |
|---|---|---|
| Michelangelo (Uber) | plataforma proprietária completa, com UI própria | configuração na plataforma |
| Metaflow (Netflix) | biblioteca de fluxo em Python, infraestrutura plugável | o fluxo inteiro, como código |
| Bighead (Airbnb) | conjunto de serviços integrados | integração com cada serviço |
| Databricks nativo | Feature Engineering, MLflow, Lakehouse Monitoring, DABs | a cola entre eles |
| este ecossistema | framework único sobre o nativo | só o que é específico do domínio |

A Uber construiu o Michelangelo depois de constatar que cada time reimplementava a mesma infraestrutura e poucos modelos chegavam em produção. A Netflix atacou o mesmo problema pelo lado oposto, com uma biblioteca que o cientista importa em vez de uma plataforma fechada. O Airbnb integrou ferramentas que já existiam. As três soluções custaram anos de engenharia dedicada só para ter uma base.

Hoje essa base vem pronta. O Databricks entrega feature store com resolução no tempo, registro com linhagem, serving nos dois modos e monitoramento. Reconstruir isso seria desperdício. O que faltava é a camada de cima, que define como o domínio declara o que quer, e é exatamente aí que as implementações divergem quando cada time resolve sozinho.

É essa camada que o framework ocupa, e a última coluna da tabela é o motivo: quanto menos o domínio escreve, menos existe para divergir entre dois modelos do mesmo banco.

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

No fim, o ganho é tempo. Quanto antes o banco identifica que um cliente vai ter dificuldade de pagar, mais cedo ele pode oferecer uma renegociação que caiba no bolso da pessoa. Quando essa informação só aparece depois que a dívida já venceu, sobra ao banco absorver a perda e ao cliente sair com o nome sujo.

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
