# FashionSearch — getting the model and data

**Neither the trained weights nor the datasets are in this zip.** They are far too
large (hundreds of MB to gigabytes) and most carry licences that forbid
redistribution.

They do not need to be. Databricks has outbound internet, so notebooks 01 and 03
fetch them into your workspace directly. That is what the numbered pipeline is
for.

---

## Run the whole thing

```powershell
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run fashion_pipeline -t dev 
```

Seven tasks run in order: setup, ingest, freeze the eval set, import the model,
register it, batch inference, gate. Roughly 20–40 minutes on serverless CPU.

Or run the notebooks one at a time from the workspace UI — they are numbered and
each one prints what to run next.

---

## Where the model comes from

Notebook 03 downloads the original author's published checkpoints. These are
real, public and directly usable — no training required.

| Hugging Face repo | Role |
|---|---|
| `yainage90/fashion-object-detection` | detector, 7 garment categories |
| `yainage90/fashion-image-feature-extractor` | encoder, 128-number embedding |

## Where the data comes from

Notebook 01 downloads `yainage90/onthelook-fashion-anchor-positive-images` —
anchor–positive pairs crawled from Korean fashion platforms.

`config.yaml` defaults to **2000 items**, deliberately. Free Edition has no GPU,
and embedding 290,000 images on CPU takes days. Two thousand produces real
Recall@k numbers in a few minutes. Raise `data.sample_size` when you have GPU
compute.

### Licensing

That data was scraped from commercial sites. Fine for learning; genuinely risky
commercially, since most platforms' terms prohibit it and a model carries its
data's provenance permanently. Every table has a `license_ok` flag — set it
honestly and the pair-building step will respect it.

---

## What you get at the end

`fashion_dev.gold.retrieval_metrics` — one row per slice per model version, with
Recall@k, MRR and NDCG@k. That table is the thing the original project has no
equivalent of, and it is what lets you answer "did this change help?" with a
number instead of nine screenshots.
