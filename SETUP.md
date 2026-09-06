# Setup: from download to a running Databricks job

Follow these in order. Steps 0 to 8 take about an hour, most of it waiting.

---

## Step 0 — What you need first

**A GitHub account.** Sign up at github.com.

**Git installed.** Run `git --version`. If that fails: Windows → git-scm.com;
Mac → `xcode-select --install`; Linux → `sudo apt install git`.

**Python 3.10+.** Run `python3 --version` (Windows: `python --version`).

**A Databricks account.** Sign up for **Free Edition** at databricks.com. No credit
card. One thing to know now: Free Edition is *serverless only*, which is why every
job in this repo is written without classic cluster definitions. They will all
deploy; §10 explains which ones can actually run.

---

## Step 1 — Unzip

```bash
cd ~
unzip ~/Downloads/fashionsearch-2026.zip
cd fashionsearch-2026
ls          # should show README.md, databricks.yml, resources/, src/
```

---

## Step 2 — Tell Git who you are

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## Step 3 — Create the GitHub repo

On github.com click **+** → **New repository**. Name it `fashionsearch-2026`.
**Do not tick "Add a README"** — this folder already has one and ticking it creates
a conflict you then have to untangle.

Then, from inside the folder:

```bash
git init
git add .
git commit -m "Initial commit: FashionSearch 2026"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fashionsearch-2026.git
git push -u origin main
```

When it asks for a password, **your GitHub password will not work** — GitHub stopped
accepting those in 2021. Go to Settings → Developer settings → Personal access tokens
→ Tokens (classic) → Generate new token, tick `repo`, and paste that string as the
password. Save it; GitHub won't show it again.

---

## Step 4 — Get your workspace URL

Log into Databricks and look at the address bar:
`https://dbc-a1b2c3d4-e5f6.cloud.databricks.com`

Copy up to and including `.com`. Nothing after.

---

## Step 5 — Install the CLI and log in

```bash
pip install databricks-cli
databricks auth login --host https://YOUR-WORKSPACE-URL
```

Press Enter to accept the default profile name. A browser opens; approve it. Check:

```bash
databricks current-user me      # should print your email
```

---

## Step 6 — Point the bundle at your workspace

Open `databricks.yml`. Under `targets: dev: workspace:` replace the placeholder
`host:` with your real URL from Step 4. Save.

---

## Step 7 — Validate

```bash
databricks bundle validate -t dev
```

Checks the config without changing anything. Complaints about the `prod` target's
permission groups are expected — that target isn't set up. You only need `dev` clean.

---

## Step 8 — Deploy and run setup

```bash
databricks bundle deploy -t dev
databricks bundle run fashion_setup -t dev
```

The first command uploads code and creates job *definitions* — nothing runs. The
second runs the one job that works out of the box: it creates the `fashion_dev`
catalog with six schemas and four volumes.

Takes a couple of minutes, mostly waiting for serverless compute to start.

Verify: in the Databricks UI, click **Catalog**. You should see `fashion_dev`
containing `raw`, `bronze`, `silver`, `gold`, `ml`, `monitoring`.

That is a real end-to-end deployment: laptop → GitHub → cloud workspace → real objects.

---

## Step 9 — Connect GitHub to Databricks (optional)

**Workspace** → **Create** → **Git folder**. Paste your repo URL, pick GitHub, Create.
If it asks for credentials, go to Settings → Linked accounts → Git integration and
paste the same token from Step 3.

---

## Step 10 — What runs, and what doesn't

Read this before running anything else, or you'll think you broke something.

| Job | Status | Why |
|---|---|---|
| `fashion_setup` | **Works** | Pure SQL on serverless |
| `fashion_data_pipeline` | Needs data + a detector | `bronze.products` and `bronze.posts` are empty, and there is no registered detector model |
| `fashion_train_and_gate` | Needs AI Runtime | Preview feature, specific regions, paid accounts. Also needs training pairs |
| `fashion_build_index` | Needs Vector Search | Not available on Free Edition |
| `fashion_monitor` | Runs, reports nothing | No production traffic to measure |

None of this is a defect. It is the normal situation for any MLOps pipeline: the
plumbing is the easy part and the data is the hard part.

---

## Step 11 — Making something real happen

The useful next move is to bring in data you can actually get.

**Option A — use the original author's published assets.** They are on Hugging Face:

- `yainage90/fashion-object-detection` — the trained detector
- `yainage90/fashion-image-feature-extractor` — the trained encoder
- `yainage90/onthelook-fashion-anchor-positive-images` — anchor/positive pairs

Download a few thousand pairs, upload the images to
`/Volumes/fashion_dev/raw/product_images/` and `/Volumes/fashion_dev/raw/post_images/`,
and insert matching rows into `bronze.products` and `bronze.posts`. The pair-building
job will then genuinely run.

**Option B — start with evaluation, not training.** This is the better learning path,
and it needs no GPU at all.

1. Take 2,000 fashion images from any public dataset.
2. Embed them with an off-the-shelf CLIP model on CPU — slow but fine at this size.
3. Hand-label 100 queries with their correct answers.
4. Compute Recall@10 with brute-force cosine similarity, about 50 lines.
5. Then swap in the author's fine-tuned encoder and compute Recall@10 again.

The gap between those two numbers is the entire value of the original project,
expressed as one figure you produced yourself. You will learn more from that
afternoon than from getting the full pipeline to deploy.

---

## Common errors

**`Error: cannot resolve variable service_principal`** — you ran validate against
`prod`. Use `-t dev`.

**`PERMISSION_DENIED: cannot create catalog`** — on Free Edition you should be
workspace admin already. If not, use the default catalog: change `catalog: fashion_dev`
in `databricks.yml` to whichever catalog already exists under **Catalog**.

**`No module named 'databricks.vector_search'`** — Vector Search isn't on Free Edition.
Skip `fashion_build_index`; use FAISS locally instead to learn the same concepts.

**`bundle deploy` hangs** — usually auth. Re-run `databricks auth login`.
