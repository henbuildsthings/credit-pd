# credit-pd

A small, end-to-end credit default (probability of default, PD) scoring service trained on the OpenML `credit-g` dataset. It's a learning project, built one module at a time, for practising MLOps fundamentals: reproducible data splits, experiment tracking and a model registry with MLflow, a single deployable sklearn Pipeline, policy-driven decision bands, and a containerised scoring API. Everything runs locally.

## Quickstart

> Placeholder: these targets don't exist yet and will be added as modules are built.

```bash
make setup     # install dependencies with uv
make data      # download credit-g and create the seeded splits
make train     # train the pipeline and log the run to MLflow
make test      # run the test suite
make serve     # start the scoring API locally
```
