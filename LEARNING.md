# LEARNING.md

Running log of what was built, why, and what I should be able to explain.

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

## Lessons

- **Module 1:** the pre-commit large-file guard flagged `uv.lock` (a false positive) — tune guardrails, don't disable them.
- **Module 2:** pandera's default error message dumps the whole schema for one extra column. Wrapping it to show only column, check and failing value was worth the extra lines. The line budget is a guide: trimming 153 → 139 was free, but going further meant dropping safeguards, so I stopped.
