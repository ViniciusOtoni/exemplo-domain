# exemplo-domain

Repositório de domínio para `exemplo` (produto `propensao_exemplo`): consome os
quatro componentes da plataforma de MLOps (`feature-platform`, `training-platform`,
`serving-platform`, `monitoring-platform`) via pip, cada um pinado a uma tag semver
específica em `platform.yml`.

## Estrutura

Um subdiretório por componente, cada um seu próprio bundle DAB independente:

```
features/     # feature-platform@v0.2.0 — customer_transaction_features (online=True)
training/     # training-platform@v0.2.0 — treino do propensao_exemplo
serving/
  batch/      # serving-platform@v0.1.0 — scoragem em lote (workflow agendado)
  online/     # serving-platform@v0.1.0 — Model Serving endpoint (FeatureLookup automático)
monitoring/   # monitoring-platform@v0.1.0 — drift de features e de predições
```

`serving/` é dividido em duas trilhas (`batch/` e `online/`) em vez de um único
bundle porque o registro de `ServingConfig` do `serving-platform` só aceita uma
config por `model_name` por processo — cada trilha roda o próprio deploy/CI,
registrando o mesmo modelo com `mode` diferente.

## CI/CD

`.github/workflows/deploy.yml` usa o reusable workflow do
[`mlops-platform`](https://github.com/ViniciusOtoni/mlops-platform), com uma matriz
sobre os cinco bundles (`features`, `training`, `serving/batch`, `serving/online`,
`monitoring`). Requer os secrets `DATABRICKS_HOST`/`DATABRICKS_TOKEN` configurados
neste repositório.

## Rodando localmente

Cada subdiretório é independente:

```bash
cd features   # ou training, serving/batch, serving/online, monitoring
python -m venv .venv
./.venv/Scripts/pip install -r requirements-dev.txt
./.venv/Scripts/pytest
python scripts/generate_resources.py   # onde existir
databricks bundle deploy -t dev
```

## Infraestrutura Lakebase (Online Feature Store)

A feature table `customer_transaction_features` está marcada `online=True` e
sincronizada com o Database Instance `exemplo-lakebase` (Lakebase, `CU_1`) via um
Database Catalog cuja `database_name` é `workspace` — precisa ser igual ao catalog
Unity Catalog da tabela de origem (ver `feature-platform`, plano de implementação,
task 11, para o porquê). `features/databricks.yml` já aponta
`database_instance_name` para essa instância por padrão.

## Verificado ao vivo

- `features/`: backfill e sync online confirmados contra o workspace real, via
  `feature-platform` instalado por pip (não do próprio repositório do framework).
- `training/`: pipeline de 4 tasks (prepare → fit/compare → select/test → register)
  confirmado contra o workspace real — nova versão do modelo `propensao_exemplo`
  registrada.
- `serving/batch`: scoragem em lote confirmada, predições gravadas em
  `workspace.exemplo_predictions.propensao_exemplo`.
- `serving/online`: endpoint de Model Serving deployado, chegou a
  `DEPLOYMENT_READY` e uma chamada de teste (`customer_id` + `reference_date`, sem
  features no payload) retornou uma predição real — confirma resolução automática
  de `FeatureLookup` contra o Online Feature Store. Endpoint derrubado após o teste
  (`databricks serving-endpoints delete`) para não gerar custo contínuo; redeploy é
  só rodar o bundle de novo.
- `monitoring/`: os dois jobs de drift (`feature_table` e `predictions`) rodaram de
  ponta a ponta, com métricas reais gravadas em `workspace.platform_monitoring.drift_metrics`
  (`status=PASS` em ambos nesta rodada).
- **CI (`.github/workflows/deploy.yml`)**: confirmado ao vivo, os 5 bundles fazendo
  deploy com sucesso via GitHub Actions após um push real em `main` (run
  [32804848226](https://github.com/ViniciusOtoni/exemplo-domain/actions/runs/32804848226)).

## Gap conhecido: endpoint online e custo do CI

O job `serving/online` do CI faz deploy real do `model_serving_endpoint` a cada push
em `main`, mas **não o derruba automaticamente** — o reusable workflow do
`mlops-platform` só sabe fazer deploy, não tem um step de teardown. Isso significa
custo contínuo real na Free Edition depois de qualquer merge, até alguém rodar
`databricks serving-endpoints delete <nome>` manualmente. Decisão consciente por
enquanto (2026-08-25): não resolvido ainda — ou adicionar um step de teardown no job
`serving/online`, ou trocar seu trigger de `push` para `workflow_dispatch` (deploy
manual, só quando alguém for testar de verdade).
