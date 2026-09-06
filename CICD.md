# Automatic deployment: VS Code → GitHub → Databricks

Once this is set up, editing a file in VS Code and pushing it means your
Databricks workspace updates by itself. No CLI commands, no manual deploy.

Setup takes about ten minutes and you only do it once.

---

## What actually happens

```
  You edit a file in VS Code
            │
            │  git push
            ▼
  GitHub receives the commit
            │
            │  GitHub Actions wakes up automatically
            ▼
  1. Runs the tests   (broken YAML? missing file? committed a token?)
            │
            │  only if they pass
            ▼
  2. Runs 'databricks bundle deploy'
            │
            ▼
  Your Databricks workspace now has the new code
```

The file that controls this is `.github/workflows/deploy.yml`. GitHub finds it
automatically — you never run it yourself.

**Nothing deploys if the tests fail.** That is the point. A typo that would have
broken a job at 3am now stops at the push instead.

---

## Step 1 — Create a Databricks token

GitHub needs permission to talk to your workspace.

1. In Databricks, click your name (top right) → **Settings**
2. → **Developer** → **Access tokens** → **Manage**
3. **Generate new token**. Comment: `github-actions`. Lifetime: 90 days.
4. **Copy it now.** It starts with `dapi` and Databricks will never show it again.

> Treat this like a password. Anyone holding it can do anything in your workspace.
> Never paste it into a file in the repo — the tests in `tests/` will actually
> fail the build if you do, which is deliberate.

---

## Step 2 — Put it into GitHub as a secret

1. Go to your repo on github.com
2. **Settings** → **Secrets and variables** → **Actions**
3. **New repository secret**, twice:

| Name | Value |
|---|---|
| `DATABRICKS_HOST` | `https://dbc-xxxx-yyyy.cloud.databricks.com` (no trailing slash) |
| `DATABRICKS_TOKEN` | the `dapi...` string from Step 1 |

Secrets are write-only. Nobody, including you, can read them back — which is why
you had to copy the token in Step 1.

---

## Step 3 — Push and watch it run

```bash
git add .
git commit -m "Set up automatic deployment"
git push
```

Open your repo on github.com and click the **Actions** tab. You'll see a run
appear within a few seconds. Click it to watch each step live.

Green tick: it deployed. Red cross: click the failed step to see exactly which
line broke — the log tells you.

From now on, every `git push` to `main` does this automatically.

---

## Step 4 — Confirm it landed

In Databricks, go to **Jobs & Pipelines**. You should see the FashionSearch jobs, and
their "last modified" time should match your push.

---

## Working in VS Code day to day

Install the **Databricks** extension for VS Code (search "Databricks" in the
Extensions panel). It gives you bundle-aware autocomplete and lets you run a
single job without pushing, which is useful while iterating.

Your normal loop becomes:

```bash
# edit files in VS Code
git add .
git commit -m "Adjust the confidence threshold"
git push
# GitHub deploys it. Go and check the Actions tab.
```

While actively debugging, deploy straight from your machine instead — faster than
waiting for CI:

```bash
databricks bundle deploy -t dev
databricks bundle run fashion_setup -t dev
```

Then push when it works.

---

## Two things that confuse people

### Databricks Git folders are a separate copy

If you connected a **Git folder** in the Databricks UI, that is a *different*
mechanism from this one. A Git folder is a checkout you browse and edit inside
Databricks, and it does **not** update on push — you have to click Pull.

The bundle deploy in this guide writes to a different location entirely
(`/Workspace/Users/you/.bundle/...`), and that is what the jobs actually run.

Pick one and stick to it, or you will spend an afternoon wondering why your fix
did not take effect. For automated deployment, use the bundle and treat the Git
folder as read-only browsing.

### Deploying is not running

`bundle deploy` uploads your code and updates the job *definitions*. It does not
execute anything.

That is deliberate — you rarely want a push to kick off an eight-hour GPU training
run. Jobs start from their own triggers: a schedule, a file arriving, or a table
being updated. To run one on demand:

```bash
databricks bundle run fashion_setup -t dev
```

If you *do* want a specific job to run after every deploy, add this to the
`deploy-dev` job in `.github/workflows/deploy.yml`:

```yaml
      - name: Run setup after deploying
        run: databricks bundle run fashion_setup -t dev
```

Only do this for cheap, idempotent jobs. `fashion_setup` qualifies; a training job
does not.

---

## Making the tests useful

`tests/test_repo_integrity.py` runs on every push and takes under a second. It
checks four things:

- every Python file parses
- every YAML file parses
- **every job task points at a file that actually exists** — this one earns its
  keep, because a task referencing a renamed script deploys perfectly and then
  fails hours later at runtime
- no access token has been committed

Add your own tests to that folder as you go. Anything that would waste your time
if it broke silently belongs there.

---

## Common failures

| What you see | What it means |
|---|---|
| `Error: cannot resolve host` | `DATABRICKS_HOST` secret is missing, misspelled, or has a trailing slash |
| `401 Unauthorized` | Token expired (they do, after the lifetime you chose) or was pasted with whitespace. Generate a new one and update the secret |
| `cannot resolve variable service_principal` | The workflow tried the prod target. Only tagged releases should do that — check your ref |
| Tests fail on `test_every_task_points_at_a_real_file` | You renamed or moved a script but not the `python_file:` line in `resources/*.yml` |
| Actions tab shows nothing at all | The workflow file must be at exactly `.github/workflows/deploy.yml`, and you must have pushed it to `main` |
| `PERMISSION_DENIED` on deploy | The token belongs to a user without workspace write access. On Free Edition you are admin, so this usually means the wrong token |

---

## Rotating the token

Tokens expire. When yours does, deploys start failing with 401. Generate a new
one (Step 1), update the `DATABRICKS_TOKEN` secret (Step 2), and re-run the
failed workflow from the Actions tab. Nothing else changes.

Set a calendar reminder a week before expiry — the failure mode is a deploy that
silently stops happening, which is easy to miss.
