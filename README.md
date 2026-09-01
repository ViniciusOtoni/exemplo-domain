# Risco de crédito sobre uma plataforma de ML

**Atenção**: esse repositório tem o intuito de realizar um comparativo entre um ecossistema sem padrões de MLOps e outro onde temos uma plataforma mais madura e preparada, e quais são os efeitos e ganhos disso. Não vamos nos aprofundar em conceitos de Ciência de Dados como estatística etc. Peguei esse problema de qualidade de crédito apenas por ser algo recente e estar atrelado à empresa na qual eu trabalho.

![Endividamento das famílias no Brasil](docs/img/endividamento-familias.png)

**Oito** em cada dez famílias brasileiras estão endividadas. Em 2015 eram **seis**.

## Problema da qualidade de crédito no setor bancário

| Indicador | Valor | Variação em 12 meses |
| --- | --- | --- |
| Comprometimento de renda | 29,7% | +1,9 p.p. |
| Endividamento sobre renda | 49,9% | +1,3 p.p. |
| Inadimplência das famílias | 5,3% | +1,4 p.p. |
| Juros do rotativo | 451,5% ao ano | — |

Dados do Banco Central, referência fevereiro e março de 2026, via [Agência Brasil](https://agenciabrasil.ebc.com.br/economia/noticia/2026-04/juros-elevados-mantem-pressao-sobre-endividamento-das-familias).

Dessa forma, podemos ter a visibilidade de que boa parte da renda da família média brasileira já está comprometida antes mesmo de cair na conta e que, se levarmos em consideração a atual taxa de juros do Brasil, a *Taxa Selic*, a probabilidade de não cumprimento do crédito pode ser grande. Não vou entrar no mérito dos motivos desse maior endividamento, mas é fato que isso é um grande alerta para todo o setor bancário, levando em consideração que um dos braços mais relevantes é a sua qualidade de crédito.

No primeiro semestre, o banco Itaú conseguiu estabilizar o aumento da inadimplência, e podemos levantar uma série de questionamentos:

* Será que os clientes do banco Itaú são melhores? (Acredito que não exista uma diferença grande aqui comparando com os outros players.)
* Será que o Itaú tem um LLM extremamente capacitado para diagnosticar esses calotes?
* Como o Itaú está conseguindo manter essa qualidade?
---

![Inadimplência acima de 90 dias por banco, 2T26](docs/img/npl-bancos-2t26.png)

E se eu te contar que um modelo de Machine Learning tradicional já é suficiente para realizar ótimas predições? Novamente, não vamos entrar em conceitos estatísticos aqui para definir qual é o melhor algoritmo para esse cenário, as melhores features, os hiperparâmetros etc. Mas sim: como podemos potencializar esse delivery para afetar o cliente final da forma mais rápida e eficaz?

## O problema de não ter uma plataforma consolidada para ML

O cientista até consegue definir qual será o algoritmo que melhor se encaixa para o problema, quais serão os hiperparâmetros utilizados e a normalização. Porém, ele já começa a se deparar com alguns problemas:

* Quais são as features? Ele não tem uma Feature Store madura o suficiente para evidenciar quais são as Feature Tables que ele deveria consumir.
* Como posso treinar o meu modelo e ter a confiança de que não vou ter problemas durante a etapa de treinamento, como Data Leakage, por exemplo?
* Como faço para promover o meu modelo? Quais são os caminhos para servir esse meu modelo?
* Como sei que meu modelo está performando bem após ser implementado?

| Etapa | O que normalmente acontece |
| --- | --- |
| Feature store | a mesma feature é recalculada em vários lugares, e os valores não batem entre si |
| Treino | cada pessoa treina do seu jeito, sem um padrão de MLOps que valha para todo mundo |
| Registro | modelo salvo sem linhagem, ninguém sabe quais features entraram |
| Serving | sem uma feature store madura por trás, batch e online são construídos separados e passam a divergir |
| Monitoramento e retreino | ninguém enxerga o drift de forma tangível, e cada safra nova levanta a dúvida de retreinar sem resposta objetiva |

Nada disso é difícil isoladamente. O custo está em fazer tudo de novo a cada modelo, e em cada reimplementação divergir um pouco da anterior.

## Como a plataforma resolve esses problemas

O objetivo é simples: dar o "caminho das pedras" para que o cientista, de forma autônoma, consiga criar as suas features e publicá-las em nossa Feature Store — fomentando o reúso e tendo mais confiabilidade nos dados —, treinar o seu modelo de forma simples e com boas práticas/padrões estabelecidos pelo mercado, servir o seu modelo e, por fim, ter a visibilidade da necessidade de retreino do modelo conforme as novas safras.

O ganho aqui é deixar a experiência do cientista mais dinâmica (logo, ele consegue produtizar os seus modelos e chegar na etapa final sem muitos problemas), aplicar padrões e confiabilidade em toda a jornada e, por fim, também economizar custos em todo o processo para a instituição, evitando a redundância entre os processos.

### Exemplo para as Feature Tables

```text
features/
├── conf/variables.yml
├── pyproject.toml
├── src/credito_features/
│   └── configs.py            # Features do domínio
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

Todo o enxoval será abstraído para o cientista: ele vai apenas criar a sua Feature Table em nossa Feature Store e consumir.

## Como o ecossistema foi montado

![Arquitetura do ecossistema](docs/img/arquitetura.png)

**Feature store.** As feature tables são criadas `point-in-time` e utilizando também o recurso de `feature-lookup`. Dessa forma, existe um vínculo intrínseco das features com o modelo.

**Treino.** O split entre os 3 datasets é feito de forma temporal: treinamos o modelo com dados do passado, geramos `child runs` para comparar qual é o modelo com melhor desempenho e declaramos o `champion`. Por fim, testamos o modelo com "dados do futuro", nos quais o modelo não tinha visibilidade durante a etapa de treinamento, e validamos a sua sanidade.

**Serving.** Temos dois processos de inferência. O **online**, onde servimos um endpoint para consumo do modelo que consome `Synced Tables` (também conhecidas como Online Tables) para recuperar os dados frescos das Feature Tables. E também o processo **batch**, onde geramos um workflow e consumimos as features do nosso catálogo de dados.

**Monitoramento.** O Lakehouse Monitoring compara a safra corrente contra a janela em que o modelo foi treinado. A pergunta não é "mudou desde o mês passado", é "afastou-se do que o modelo aprendeu". Quando afasta além do limiar, dispara um retreino no GitHub. O candidato fica registrado e não promovido: alguém precisa aprovar.

### O padrão por trás dessa escolha

Eu não reinventei a roda. Isso já é um padrão que alguns players de mercado adotam. Seguem exemplos:

| Plataforma | Abordagem | O que o domínio escreve |
| --- | --- | --- |
| Michelangelo (Uber) | plataforma proprietária completa, com UI própria | configuração na plataforma |
| Metaflow (Netflix) | biblioteca de fluxo em Python, infraestrutura plugável | o fluxo inteiro, como código |
| Bighead (Airbnb) | conjunto de serviços integrados | integração com cada serviço |

Todas essas empresas identificaram o mesmo problema: redundância de infraestrutura, demora para produtização de seus modelos e poucos ganhos. A decisão foi estruturar a plataforma conforme a necessidade.

### Qual foi o caminho da *minha* solução

Hoje meu ecossistema está totalmente atrelado ao Databricks. Talvez com um certo viés pelo fato de a empresa em que eu trabalho utilizar a ferramenta, mas é fato que, com ela, conseguimos centralizar todas as soluções pertinentes a esse problema.

Conseguimos estruturar a nossa Feature Store e ainda aplicar governança de dados nas tabelas; também conseguimos utilizar o `Lakebase` para transacionar as Feature Tables com baixa latência para o cenário da inferência online, além de conseguir promover o reúso do nosso componente sobre toda a plataforma.

## O que muda no ciclo de vida

Agora o cientista, idealmente, não gastaria mais tanto tempo se questionando sobre como servir o modelo dele e gerar impacto para a instituição, mas sim focaria apenas no que tange à sua jornada. O resto, a plataforma irá abstrair.

E para a instituição? Qual é o ganho em ter essa plataforma?

No fim, o ganho é tempo e impacto na qualidade de seu crédito. Quanto antes o modelo estiver sendo consumido de forma confiável e governada, mais cedo o banco poderá sugerir uma renegociação para o cliente ou até mesmo evitar de emprestar crédito para possíveis clientes inadimplentes, gerando um impacto real em sua receita líquida.
