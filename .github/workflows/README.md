# The four workflows

Modelled on the DR-AutoML project's structure: numbered, each with one job, so
that when something fails you know which stage it was.

```
git push
   │
   ├──► 1 · Deploy code to Kaggle ─────────► kernel starts on a T4
   │                                              │
   └──► 2 · Deploy Databricks + SETUP             │  embeds, registers to UC,
        (catalog, schemas, volumes)               │  uploads vectors
                                                  │
                                    repository_dispatch
                                                  │
                                                  ▼
                                    3 · Run Databricks PIPELINE
                                    ingest, evaluate, gate, serve


   daily 03:00 UTC
        │
        ▼
   4 · Drift check ──drift?──► re-push Kaggle ──► back into 3
```

## Why 1 and 2 both run on push

Kaggle uploads into `/Volumes/fashion_dev/silver/kaggle_inbox` and registers a
model into the `fashion_dev` catalog. Neither exists on a fresh workspace.

Workflow 2 creates them. It runs on every push because setup is idempotent and
takes about a minute — far cheaper than a Kaggle run that does all its GPU work
and then fails at the upload because a volume was missing.

## Why 3 is separate

It must run **after** Kaggle, not alongside it. Evaluating before the embeddings
land would score the previous run's vectors — a bug that produces plausible
numbers and no error at all.

The callback also means nothing sits waiting. An earlier version had GitHub
polling Kaggle for up to an hour, burning Actions minutes on a runner doing
nothing but sleeping.

## Why 4 exists

Everything else triggers on a commit. Drift does not: the catalogue turns over,
photo styles shift, and the model file is identical while its real quality
moves. Workflow 4 is the only thing watching for that.

## The five secrets

| Secret | Direction |
|---|---|
| `KAGGLE_API_TOKEN` | GitHub → Kaggle. From kaggle.com/settings/api → Generate New Token |
| `KAGGLE_USERNAME` | builds the kernel id |
| `DATABRICKS_HOST` | GitHub and Kaggle → Databricks |
| `DATABRICKS_TOKEN` | same. Do **not** add API scopes — a scoped token cannot register models |
| `GH_DISPATCH_TOKEN` | Kaggle → GitHub. A PAT with `repo` scope |

`GH_REPO` is not a secret; it comes from `github.repository`.

## Skipping the expensive parts

```bash
git commit -m "fix typo in README [skip kaggle]"
```

Workflow 1 honours `[skip kaggle]`. Workflow 2 always runs, which is fine — it
is a minute.
