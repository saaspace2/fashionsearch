# The Kaggle half

Embedding 10,000 products on Databricks serverless CPU takes about 45 minutes.
On a Kaggle T4 it takes two or three. So the GPU work runs there and Databricks
keeps governance — registry, evaluation, gate, monitoring.

**It is fully automatic.** One `git push` runs everything, in order.

```
git push
   │
   ├─ validate    tests, lint, bundle validate
   ├─ kaggle      push the kernel, run it on a T4, wait for it to finish
   └─ databricks  deploy, then run the pipeline over Kaggle's output
```

The jobs are chained deliberately. Databricks must not start until Kaggle has
written its embeddings, or notebook 05b would ingest the previous run's output.

## Setup — four secrets, once

All four live in **GitHub**: repo → Settings → Secrets and variables → Actions.

| Secret | Where to get it |
|---|---|
| `KAGGLE_USERNAME` | kaggle.com → Settings → API → Create New Token |
| `KAGGLE_KEY` | the `key` field of that same `kaggle.json` |
| `DATABRICKS_HOST` | `https://dbc-xxxx-yyyy.cloud.databricks.com`, no trailing slash |
| `DATABRICKS_TOKEN` | Databricks → Settings → Developer → Access tokens |

Nothing to configure on Kaggle. No kernel to create, no dataset to upload, no
username to edit in a file — the workflow generates `kernel-metadata.json` from
`KAGGLE_USERNAME` at push time.

When generating the Databricks token, **do not add API scopes**. A scoped token
cannot register models and the run will fail at that step.

## How the credentials reach Kaggle

This is the part that makes automation possible, and it is not obvious.

Kaggle Secrets (Add-ons → Secrets) are managed in the UI and are **stripped
every time a kernel is pushed by the CLI**. A kernel relying on them would lose
its credentials on every automated push and abort — which is why the DR-AutoML
project's workflow says the kernel has to be started by hand.

Dataset attachments behave differently: they are part of `kernel-metadata.json`,
which the kernel owns, so they survive CLI pushes. So the workflow writes the
Databricks credentials into a **private Kaggle Dataset** and attaches it.

The trade: that dataset holds the token in cleartext. It is private and readable
only by you, and the token should be short-lived. Real trade, not a free win.

## What each push does

| Job | Roughly | Skip with |
|---|---|---|
| validate | 1 min | — |
| kaggle | 5–20 min | `[skip kaggle]` |
| databricks | 10–45 min | `[skip pipeline]` |

Put the tag anywhere in the commit message:

```bash
git commit -m "fix typo in README [skip kaggle] [skip pipeline]"
```

Worth using. A README edit should not cost an hour of compute.

## Fine-tuning

Off by default. Turn it on by setting `"finetune": True` in the `DEFAULTS` dict
in `run_on_kaggle.py`. Three epochs on 10,000 pairs takes about 15 minutes on a
T4.

Fine-tuning is the one thing here that genuinely needs a GPU, which is exactly
why it belongs on this side of the split.

## The line that matters

In the Kaggle job log:

```
registered fashion_dev.ml.fashion_encoder version 1 in Unity Catalog
```

Registering from Kaggle, over the REST API with a token, uses a different upload
path than registering from inside a Databricks serverless notebook — where the
cluster's assumed IAM role is denied write access to UC managed storage.

Once that line appears the model is in Unity Catalog, and notebook 07 can create
a Model Serving endpoint. That single difference is why the GPU split also
unblocks serving.

## If the Kaggle job fails

The workflow prints the kernel URL. Open it and read the log — the failure is
almost always a missing secret or a Databricks token without permission.

Databricks still runs afterwards regardless (`if: always()`), falling back to
CPU embedding in notebook 05. A failed Kaggle run makes the pipeline slow, not
broken.
