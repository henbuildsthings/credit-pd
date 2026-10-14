# LEARNING.md

Running log of what was built, why, and what I should be able to explain.

## Working from the terminal

Generated from the `Makefile` and `src/`; keep it in sync when targets change.

### Opening the project

1. VS Code → File → Open Recent → `credit-pd [WSL: Ubuntu]`.
2. Check the bottom-left corner says **WSL: Ubuntu** (otherwise you're on Windows and `uv`/`make` won't be found).
3. Open a terminal (Ctrl+`) and start Claude Code: `claude` (new session) or `claude --continue` (resume the last one).
4. `git status` — see where you left off.

### How a command runs

```
make train                                 Makefile: names the task, exports MLFLOW_DISABLE_AGENT_HINT=1
 └─ uv run python -m credit_pd.train       uv: runs inside .venv, synced to uv.lock first
     └─ credit_pd/train.py                 reads config.yaml (via config.py)
         ├─ reads   data/processed/splits.parquet, data/manifest.json, uv.lock, git
         └─ writes  mlflow.db (runs, registry) + mlruns/ (artifacts, model)
```

Every target follows the same chain: `make <target>` → `uv run ...` → a module in `src/credit_pd/`.

### Make targets

| Target | Runs | What it does | Reads | Writes / changes | Re-run safe? |
|---|---|---|---|---|---|
| `help` (default) | grep over Makefile | Lists targets | `Makefile` | — | yes |
| `setup` | `uv sync`, `pre-commit install` | Installs deps and git hooks | `pyproject.toml`, `uv.lock` | `.venv/`, `.git/hooks/pre-commit` | yes |
| `lint` | `ruff check`, `ruff format --check` | Checks lint and formatting; fails, doesn't fix | `*.py` | — | yes |
| `format` | `ruff check --fix`, `ruff format` | Auto-fixes lint and formatting | `*.py` | `*.py` in place | yes (review the diff) |
| `test` | `pytest -v` | All 17 tests, incl. integration (model sanity on real data) | `tests/`, `data/processed/splits.parquet` | `.pytest_cache/`; MLflow only in temp dirs | yes |
| `test-fast` | `pytest -v -m "not integration"` | Unit, smoke and ops tests; no data needed (~8 s) | `tests/` | `.pytest_cache/`; MLflow only in temp dirs | yes |
| `loc` | `scripts/loc.py` | Code / total lines vs the ~800 budget | `src/`, `tests/` | — | yes |
| `ingest` | `credit_pd.data` | Fetch/load, validate, hash, split; checks manifest | `config.yaml`, `data/manifest.json`, OpenML if raw missing | `data/raw/`, `data/processed/splits.parquet`, manifest on first run | yes (fails loudly on mismatch) |
| `train` | `credit_pd.train` | Fits 3 models, picks on val, tests winner, registers it | `config.yaml`, splits, manifest, `uv.lock`, git | `mlflow.db`, `mlruns/` — **a new model version every run** | yes, but adds a version |
| `models` | `credit_pd.registry list` | Table of versions, aliases, AUCs, hashes, git_dirty | `mlflow.db` | — | yes |
| `promote VERSION=n REASON="..."` | `credit_pd.registry promote` | Moves the `champion` alias, tags who/when/why, smoke-scores 3 test rows | `mlflow.db`, `mlruns/`, splits, git | alias + version tags in `mlflow.db` | changes what's deployed; refuses dirty runs |
| `mlflow-ui` | `mlflow ui --port 5000` | Web UI for runs and registry (runs until Ctrl+C) | `mlflow.db`, `mlruns/` | — | yes |
| `serve` | placeholder | Scoring API (Module 6) | — | — | yes |
| `monitor` | placeholder | Drift on `future` (Module 9) | — | — | yes |
| `clean` | `rm`, `find` | Removes `.pytest_cache`, `.ruff_cache`, `__pycache__` | — | deletes caches | yes |

### Typical workflows

**First-time setup (fresh clone)**
```bash
git clone https://github.com/henbuildsthings/credit-pd.git && cd credit-pd
make setup          # .venv + git hooks
make ingest         # downloads data; must match the committed manifest
make train          # first model version
```

**Daily start**
```bash
git status && git log --oneline -5
make setup          # cheap; picks up dependency changes
make models         # what's registered, what's champion
```

**Change code → lint → test → commit → train → promote**
```bash
make format && make lint
make test
git add -A && git commit -m "..."
make train          # note the new version number
make models         # compare it with the current champion
make promote VERSION=<n> REASON="why this one"
```
Commit **before** training: the run tags `git_sha` and `git_dirty`. A model trained on uncommitted code can't be traced to a commit, so `promote` refuses it.

**Inspect models**
```bash
make models
make mlflow-ui      # open http://localhost:5000 in the Windows browser; Ctrl+C to stop
```

**Roll back the champion**
```bash
make mlflow-ui      # champion version → tag previous_champion shows what it replaced
make promote VERSION=<previous> REASON="rollback: ..."
```

**Deliberate data update**
```bash
uv run python -m credit_pd.data --force   # re-download, rewrite data/manifest.json
git diff data/manifest.json               # review what changed
git add data/manifest.json && git commit -m "Data: ..."
make train                                # retrain on the new data (after the commit)
```

### Git basics we use

| Command | What it does |
|---|---|
| `git status` | What changed, what's staged, ahead/behind `origin` |
| `git add -A` | Stage everything: new, modified, deleted |
| `git commit -m "..."` | Commit staged changes; pre-commit hooks run here |
| `git push` | Send commits to GitHub |
| `git log --oneline` | One line per commit |
| `git restore <file>` | Throw away uncommitted changes to a file (can't be undone) |

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `uv: command not found` | Terminal is PowerShell/Windows, not WSL | Reopen the folder as `[WSL: Ubuntu]`; check bottom-left |
| `lockfile needs to be updated` | `pyproject.toml` changed without re-locking (seen with `--locked`) | `uv lock`, then commit `uv.lock` |
| Commit fails, hook says "files were modified" | Hook auto-fixed formatting or whitespace | Review, then `git add -A && git commit -m "..."` again |
| `Refusing to promote ... uncommitted code` | That version was trained with `git_dirty=true` | Commit, `make train`, promote the new version |
| `mlflow ui`: port 5000 already in use | An old UI is still running | Ctrl+C it in its terminal, or `ss -ltnp \| grep :5000` and `kill <pid>` |

## Concepts

### MLflow: generic tool, our conventions

Generic MLflow gives you runs (params, metrics, artifacts, tags), a UI, and a model registry with versions and aliases. It's the same MLflow that Databricks / Unity Catalog uses. What MLflow does *not* decide, and this project does:
- **Reproducibility tags** on every training run: `git_sha`, `git_dirty`, `data_sha256`, `uv_lock_sha256`, `seed`.
- **Rules:** only the winner is registered; promotion is a manual gate (`make promote`); no promotion of dirty versions; rollback = move the `champion` alias.
- **What a model is:** the full sklearn Pipeline (preprocessing + model), logged with a signature on raw features.

### Test types

| Type | Checks | Catches | Here |
|---|---|---|---|
| Unit | One function on small, hand-made data | Logic bugs: schema too loose, hash unstable, split not stratified | `test_data.py`, `test_features.py` |
| Model sanity (performance floor) | Real model on real data clears minimum metrics | A broken pipeline: leakage removed, features dropped, wrong target | `test_model.py` (`integration`) |
| Smoke | The entry point runs end to end | Wiring: imports, config keys, MLflow calls, return values | `test_pipeline.py` |
| Ops | The tooling around the model | Registry/list/promote bugs: invisible champion, dirty promotion | `test_registry.py` |

### Flaky tests

A flaky test passes or fails without the code changing. Causes and fixes:
- **Network** (downloads, APIs) → fixtures with local data; `sample_df` is generated, never fetched.
- **Unseeded randomness** → seed everything (`np.random.default_rng(0)`, `random_state=seed`).
- **Time/date dependence** → don't read the clock in assertions.
- **Exact float equality** → thresholds (`auc > 0.70`) or `pytest.approx`.

### Performance floor

Set the floor below the observed metric by a noise margin: val AUC 0.743 with a standard error of ~0.04 → floor 0.70. A floor catches breakage, not small regressions — a 0.01 drop is within noise and needs a proper comparison (CV, bootstrap), not a unit test.

### Tests never touch real state

Every MLflow test gets a throwaway tracking DB and artifact folder in `tmp_path` (`tmp_mlflow` fixture). Otherwise tests would register junk versions in the real `mlflow.db`, or move the real champion. Verified: the real `mlflow.db` checksum is unchanged after `make test`.

## Decisions

- **Deployable unit = full sklearn Pipeline.** Preprocessing lives inside the pipeline; the API does no preprocessing of its own.
- **Split once, seeded, stratified: 55/15/15/15 train/val/test/future.** The "future" slice is never used in training.
- **Decision bands are policy.** They're kept in `config.yaml` and logged with each model.
- **Local MLflow is the registry of record.** The promoted model is exported as a GitHub Release asset for CI and Docker.
- **Every run logs** the git SHA, `uv.lock` hash, data SHA-256 and seed.
- **Dependency split: runtime = imported by `src/`, dev = only tools/tests.** Runtime: pandas, pyarrow, scikit-learn, pandera, mlflow, fastapi, uvicorn, pyyaml. Dev group (`uv add --dev`): pytest, httpx, ruff, pre-commit.
- **Large-file guard (500 KB) blocks data/models from git; `uv.lock` is excluded because it must be committed.**
- **Data is fetched by pinned OpenML `data_id` (31), never by name.** Names can point at a new dataset version; ids can't.
- **Data version = content hash, not file hash.** SHA-256 over column names, dtypes and `pd.util.hash_pandas_object(df, index=False)`, so re-writing the same data to parquet gives the same hash.
- **`data/manifest.json` is committed; raw and processed data are not.** The manifest is the data's identity card: source, id, shape, columns, hash, seed, split sizes and default rates.
- **Ingest is strict and idempotent.** Any mismatch with the manifest (hash, seed, split sizes) raises and writes nothing. Data only changes deliberately, with `--force`.
- **Schema is a pandera `DataFrameModel` with `strict=True`.** Explicit types, allowed category values taken from the data, ranges on age, amount and duration. Validation errors list every failure at once (`lazy=True`).
- **Splits are stored as one parquet file with a `split` column.** Integer split sizes, so float fractions can't turn 150 into 149.
- **Preprocessing lives inside the sklearn Pipeline, fitted on train only.** Scaling and one-hot encoding are part of the saved model, so serving applies exactly the transforms learned in training.
- **Train fits, val selects (by AUC), test is touched once, for the winner only.** Looking at test for every candidate would turn it into a second val set.
- **Baseline = `DummyClassifier(strategy="prior")`.** Predicts the training default rate for everyone (AUC 0.5); any real model must clearly beat it.
- **Sensible defaults, no hyperparameter tuning.** Model settings live in `config.yaml` under `models`; with 550 training rows, tuning would mostly fit noise.
- **`joblib` is declared explicitly in `pyproject.toml`** because `src/` imports it, even though scikit-learn already pulls it in.
- **All three models are logged as nested child runs; only the winner is registered.** Runs keep the evidence for every candidate, including the losers; the registry holds only production candidates.
- **The `champion` alias is set only by an explicit `make promote` (a manual gate).** Each promotion tags the version with `promoted_by`, `promoted_at`, `reason` and `previous_champion`.
- **The API loads the model by alias, so rollback = move the alias.** No code change or redeploy.
- **`promote` refuses versions trained with `git_dirty=true`.** Enforced, with no override flag: commit and retrain.
- **Every training run is tagged with** `git_sha`, `git_dirty`, `data_sha256`, `uv_lock_sha256` and `seed`.
- **MLflow's agent hint is silenced** (`MLFLOW_DISABLE_AGENT_HINT=1` in the Makefile): noisy logs hide real warnings.
- **Tests are offline, deterministic and isolated.** A generated, seeded `sample_df` fixture instead of downloads; a throwaway MLflow (`tmp_mlflow`) per test, never the real `mlflow.db`.
- **Integration tests on real data skip if the data is missing** (`make ingest` first), with a clear reason.
- **Performance floor = 0.70 val AUC** (observed 0.743 minus a noise margin); **logreg must beat the baseline by 0.10 AUC.**
- **`train()` takes a config and data** so tests can drive it; with no arguments it uses `config.yaml` and the splits file.
- **The `data_sha256` tag hashes the rows actually trained on**, not the manifest value; for real runs the two are identical.
- **Dev environment:** WSL2 Ubuntu, with the project in the Linux filesystem (not OneDrive or `/mnt/c`).

## Commands

### Module 1 — Repo and environment

- `uv init --package --name credit-pd --python 3.12` — create `pyproject.toml`, `.python-version` and the `src/credit_pd/` package layout.
- `uv add <pkg>` / `uv add --dev <pkg>` — add a runtime dependency or a dev-group dependency to `pyproject.toml` and update the lock and venv.
- `uv lock` — resolve all dependencies (including transitive ones) and write exact pinned versions with hashes to `uv.lock`.
- `uv sync` — make `.venv` match the lock file (installs, upgrades and removes packages as needed).
- `uv sync --locked` — same, but fail if `uv.lock` is out of date with `pyproject.toml` instead of re-locking; use in CI and Docker.
- `make setup` — `uv sync` plus `uv run pre-commit install` to install the git hooks.
- `make lint` — `ruff check` and `ruff format --check`; fails without changing files.
- `git add -A && git commit -m "..."` — stage all changes (new, modified, deleted) and commit; pre-commit hooks run here.
- `gh repo create --public --source=. --push` — create the public GitHub repo from the local folder, add it as `origin` and push.
- `git config user.email "<id>+<user>@users.noreply.github.com"` — commit with GitHub's noreply address so the real email isn't published.

### Module 2 — Data ingestion

- `make ingest` — run `python -m credit_pd.data`: fetch or load the raw data, validate, hash, split, and check against `data/manifest.json`.
- `uv run python -m credit_pd.data --force` — re-download and rewrite the manifest (for a deliberate data update).
- `python -m <package.module>` — run a module inside the installed package as a script, so the package imports work.

### Module 3 — Training pipeline

- `make train` — run `python -m credit_pd.train`: fit all models on train, compare on val, test the winner once, save `models/model.joblib` and `reports/metrics.json`.

### Module 4 — MLflow tracking and registry

- `make models` — list registered versions with aliases, val/test AUC, data hash, git SHA, git_dirty and creation time.
- `make promote VERSION=n REASON="..."` — move the `champion` alias to version n, tag who/when/why/previous, and smoke-score 3 test rows. Refuses dirty versions.
- `make mlflow-ui` — MLflow web UI on http://localhost:5000 (Ctrl+C to stop).

### Module 5 — Testing

- `make test` — run all tests (`pytest -v`), including integration tests on real data.
- `make test-fast` — everything except integration (`pytest -v -m "not integration"`); no data needed.
- `uv run pytest -v -m integration` — only the model sanity tests on real data.

## Interview questions

### Module 1 — Repo and environment

**Q1. What's the difference between `pyproject.toml`, `uv.lock` and `.venv`?**
`pyproject.toml` declares the direct dependencies as version ranges. `uv.lock` pins exact versions of all transitive dependencies, with hashes. Commit both for reproducibility. `.venv` is not committed: it's large, OS-specific and can be rebuilt from the lock.

**Q2. Why do the hooks fix files but still block the commit, and why run the same checks in CI too?**
A human must review automatic changes before they're committed. Pre-commit is optional and easy to bypass (`--no-verify`, hooks not installed, edits made on the web), so CI is the enforced gate. Convenience runs locally; enforcement runs on the server.

**Q3. The same code gives different results on two machines. What could differ, and how do you prevent it?**
The Python version, the OS, or transitive dependency versions. Prevent it with a committed lock file plus `--locked` installs, a pinned Python version, and developing on the same OS as production. (A mismatch in model version is a registry issue — Module 4.)

### Module 2 — Data ingestion

**Q1. Why hash the content instead of the file bytes?**
File bytes change with the parquet writer version, compression or row-group layout even when the data is identical. A content hash over column names, dtypes and values only changes when the data does. Including dtypes catches silent type drift, such as an int column becoming float.

**Q2. Why fail on a manifest mismatch instead of overwriting it?**
Every model is trained on, and logs, a specific data hash. If upstream data changes silently, past results can't be reproduced and nobody knows why the metrics moved. Failing loudly forces a deliberate `--force`, which shows up as a manifest change in git.

**Q3. Why validate with a strict schema at ingest, and not just in the model pipeline?**
Fail fast, at the boundary where the bad data comes in, with a clear message. A strict schema catches missing or extra columns, new category values and out-of-range numbers before they reach training, where they'd show up as a confusing error or, worse, not at all. The same schema can be reused later to check serving inputs.

**Q4. Why a "future" split, and why stratify?**
"Future" is held out from all training and tuning, to simulate later data for monitoring and final checks. Stratifying keeps the default rate (30%) the same in every split, so the small 150-row splits aren't skewed by chance.

### Module 3 — Training pipeline

**Q1. How does putting preprocessing in the Pipeline prevent train/serve skew and leakage?**
The fitted preprocessing travels inside the model object, so serving applies identical transforms with no re-implementation in the API. Fitting the scaler before splitting would leak val/test statistics (means, variances) into training and inflate the scores.

**Q2. Logistic regression beat gradient boosting by 0.006 AUC on 150 val rows. What do you conclude?**
That it's noise: the AUC standard error is about 0.04 with 45 defaults. Call the two equivalent and prefer logistic regression as simpler and better calibrated. For real certainty, use cross-validation or bootstrap confidence intervals.

**Q3. Two models have the same AUC, but one predicts a mean PD of 25% when 30% actually default. Which do you deploy?**
The calibrated one. Decision bands use the absolute PD level, and AUC is blind to it. Otherwise, recalibrate (Platt scaling or isotonic regression) on held-out data.

### Module 4 — MLflow tracking and registry

**Q1. What's the difference between an MLflow run and a model version?**
A run is one experiment's evidence: params, metrics, artifacts and tags, including for the models that lost. A version is a production candidate with a stable name and number. Log everything; register only the winner.

**Q2. Why does the API load an alias instead of a version number?**
The API asks for "champion", so promoting or rolling back just moves the alias — no code change, no redeploy. Rollback is `make promote VERSION=<previous>`, with the previous version found via the `previous_champion` tag.

**Q3. An auditor asks how a specific decision was made. How do you trace it?**
The prediction log gives the model version (Module 9) → its training run → `git_sha` (with `git_dirty=false`), `data_sha256` (compare with the manifest), `uv_lock_sha256` and `seed`. Check out that commit, run `make ingest` (the hash must match) and `make train` to reproduce the model. The promotion tags say who promoted it, when and why.

### Module 5 — Testing

**Q1. What makes tests flaky, and how do you fix it?**
Network → fixtures with local data; unseeded randomness → seeds; time dependence → no clock in assertions, or inject the time; exact float equality → thresholds or `pytest.approx`. Flaky tests are worse than they look: they teach people to ignore red.

**Q2. Why does every test get a throwaway MLflow?**
Tests run everywhere and often. Touching the real registry would register junk versions or move the real champion. Tests must be isolated and free of side effects.

**Q3. A useless model passed "beats baseline". What went wrong, and how do you know your tests work?**
There was no margin: 0.503 > 0.500 is noise. Require baseline + 0.10. You verify tests by breaking the code on purpose (shuffling labels, reintroducing the alias bug) and checking that they fail.

## Lessons

- **Module 1:** the pre-commit large-file guard flagged `uv.lock` (a false positive) — tune guardrails, don't disable them.
- **Module 2:** pandera's default error message dumps the whole schema for one extra column. Wrapping it to show only column, check and failing value was worth the extra lines. The line budget is a guide: trimming 153 → 139 was free, but going further meant dropping safeguards, so I stopped.
- **Module 3:** my prediction (GBM ~0.80 AUC) was wrong — logistic regression won 0.743 vs 0.737 on val; small, mostly linear tabular data favours GLM-like models. GBM under-predicted the level by 4.5 pp (mean_pred 0.255 vs 0.300), which AUC can't see. With 15 rows per decile, calibration bins carry ~12 pp of noise, and the val/test AUC gap (0.74 vs 0.79) is noise too. Loading a joblib/pickle file executes code — only load trusted files.
- **Module 4:** ops tooling can fail silently too — list showed no champion while one was set; Module 5 tests should cover registry list.
- **Module 4:** my prediction of 4 versions was wrong — there were 2, because only winners are registered (runs = evidence, versions = candidates). Versions 1–2 were trained on a dirty tree, so their code couldn't be reproduced → the rule is now enforced in `promote`.
- **Module 5:** flaky-test causes — I picked network + time, and missed unseeded randomness and exact float equality. I'd also have pointed the tests at the real `mlflow.db`; a per-test throwaway registry is the fix.
- **Module 5 (sabotage):** shuffled training labels were caught only by the AUC floor. "Beats baseline" passed on noise (now fixed with a margin). Mean PD passed — correctly: ranking and level are independent, so you need both checks. My prediction (the first two fail) was half right. Safe revert trick: stage the current file before `git restore`, so it restores today's version, not the last commit.
