# The app

Upload a photo, get ranked products. Plus a dashboard of the live model and its
metrics, and a drift view.

Three tabs:

| Tab | What it does | Reads from |
|---|---|---|
| **Search** | Upload an image → serving endpoint → rank the catalogue | endpoint + `silver.product_embeddings` |
| **Dashboard** | Live model version, Recall@20, NDCG@20, per-slice table, trend | `gold.retrieval_metrics` |
| **Drift** | PSI and distance between the last two snapshots | `monitoring.drift_snapshots` |

## Running it locally

```bash
pip install -r app/requirements.txt

export DATABRICKS_HOST=https://dbc-xxxx-yyyy.cloud.databricks.com
export DATABRICKS_TOKEN=dapi...
export WAREHOUSE_ID=...          # see below
export CATALOG=fashion_dev
export ENDPOINT=fashion-encoder-endpoint

streamlit run app/app.py
```

### Finding WAREHOUSE_ID

Databricks → **SQL Warehouses** → click yours → the ID is in the connection
details, and in the URL after `/warehouses/`.

The app reads Delta tables through the Statement Execution API rather than a SQL
driver, so all it needs is the SDK — no ODBC, nothing to compile.

## Deploying as a Databricks App

```bash
databricks apps create fashionsearch
databricks sync app/ /Workspace/Users/you@example.com/fashionsearch-app
databricks apps deploy fashionsearch \
  --source-code-path /Workspace/Users/you@example.com/fashionsearch-app
```

Set `WAREHOUSE_ID` in `app.yaml` first. Credentials are injected automatically
when running as an App, so `DATABRICKS_HOST` and `DATABRICKS_TOKEN` are not
needed there.

## Notes on how it behaves

**The catalogue is cached for ten minutes.** A few thousand 128-float vectors is
a couple of megabytes — small enough to hold in memory, and re-fetching it per
search would make every query slow for no reason.

**Search runs in the app, not on Databricks.** The endpoint embeds one image;
comparing that vector against the catalogue is a single matrix multiply and is
faster done locally. At millions of products you would want a vector index
instead, which is a different design.

**Missing images show a placeholder.** Only a few hundred product images are
uploaded to the volume — vectors for everything, pictures for a few. Rows whose
`image_path` still carries a `kaggle://` marker simply have no picture.

**Both endpoint shapes are handled.** Whether the endpoint serves the combined
search model (which detects, crops and embeds) or the encoder alone, the app
works. With the encoder-only version there is no detected category, so the
category filter has nothing to restrict to.
