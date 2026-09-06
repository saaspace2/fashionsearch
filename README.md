# FashionSearch 2026

A modernized rebuild of [yainage90/fashion-visual-search](https://github.com/yainage90/fashion-visual-search),
plus the retrieval evaluation and MLOps pipeline on Databricks that the original never had.

> **Architecture diagrams:** see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the
> data flow between the models (online and offline) and the request control flow.

> **Note on `src/`:** `fashionsearch/` (config, metrics, promotion) is imported by
> the notebooks and is live code. The other `src/` folders — `ingest/`, `curate/`,
> `train/`, `evaluate/`, `index/`, `serve/` — are reference implementations of the
> larger design described below. They are not wired into `jobs_pipeline.yml`, which
> uses the numbered notebooks instead. Read them for the ideas; run the notebooks.

---

## 1. What the original project did

The goal is **image-to-image search in the fashion domain**. A user sees an outfit photo — a
street-style post, a screenshot, a friend's picture — and wants to buy the jacket in it. The
system has to find visually similar products in a shop catalogue.

The author's framing is correct and worth repeating: this is *not* the same problem as generic
similar-image search. Generic search asks "what other photos look like this photo?" Fashion
search asks "which of these 200,000 products is the same garment as the one in the top-left of
this photo?" That needs two things a generic system doesn't have.

**Object detection.** Find the individual fashion items inside a busy photo. A fine-tuned
`microsoft/conditional-detr-resnet-50`, trained on ModaNet and Fashionpedia with the label sets
collapsed to seven classes: bag, bottom, dress, hat, outer, shoes, top.

**An image encoder.** Turn each detected crop into a 128-number **embedding** such that the same
garment photographed two different ways lands in nearly the same place. A
`microsoft/swin-base-patch4-window7-224` backbone with a 128-dimensional fully-connected layer
bolted on and L2 normalisation applied.

**A third model nobody talks about.** A category classifier, used only to build the training
data — it labels product thumbnails so they can be paired with detections of the same category.

### The training data, which is the actual work

The author is blunt about this: data collection, preprocessing and labelling was **over 90% of
the project**. That estimate matches most production retrieval systems.

User posts were crawled from onthelook.co.kr and kream.co.kr. Each post pairs a styled photo
with the product thumbnails it references. Run detection on the post, run category
classification on the thumbnail, match them by category, and you get an **anchor–positive
pair**: a crop from the wild, and a clean catalogue shot of the same thing. About 290,000 pairs
across six categories.

### The training method

**Contrastive learning with in-batch negatives** — the technique behind CLIP. Take a batch of
anchor–positive pairs. For each anchor, its own positive should be pulled close in embedding
space, and *every other positive in the batch* should be pushed away. With a batch of 256, each
anchor learns from 255 negatives for free.

The author first tried triplet loss with explicitly constructed 1:1:1 anchor/positive/negative
triples and found in-batch contrastive learning clearly better. That's the expected result and
the reason is worth internalising: triplet loss gives you one negative per step, in-batch gives
you hundreds, and contrastive learning is overwhelmingly bottlenecked by negative count.

---

## 2. The gap that matters

The technical choices in this project were sound for 2024. Some have aged, and those are listed
in §3. But there is one omission that is categorically more serious than the rest.

### There are no retrieval metrics

The "Experiments" section shows nine query images and their results. They look good. That is a
demonstration, not an evaluation.

Without numbers you cannot answer any of the questions that decide whether a search system is
worth deploying:

- Is the new encoder better than the old one, or does it just fail differently?
- Does it work as well on bags as it does on shoes? (It almost certainly does not.)
- If we truncate embeddings from 128 to 64 dimensions and halve our serving cost, how much
  accuracy do we lose?
- Is the system worse on user photos taken indoors, at night, or at an angle?
- Did last week's retrain make anything worse?

Every one of these is answerable with a labelled evaluation set and three standard metrics.
Building that set is a few days of work. Not building it means every subsequent decision is made
on vibes, and it is why so many visual search projects plateau at "looks impressive in the demo".

§8 of this README, and `src/evaluate/04_evaluate_retrieval.py`, are about closing this gap. If
you take one thing from this repo, take that file.

### It also cannot scale as written

The experiment searches ~15,000 thumbnails per category by computing similarity against every
single one. That is **brute-force search**, and it is fine at 15,000 and impossible at 2 million.
A real catalogue needs an **approximate nearest neighbour (ANN) index** — a data structure that
finds the closest matches without comparing against everything. §7 covers this.

---

## 3. What to change, and why

| Component | Original (2024) | FashionSearch 2026 | Reason |
|---|---|---|---|
| Detector | Conditional DETR + ResNet-50 | RT-DETRv2 or D-FINE | Faster, more accurate, NMS-free, exports cleanly to ONNX |
| New categories | Retrain from scratch | Open-vocabulary teacher (OWLv2/Grounding DINO) auto-labels, then distil | Add "scarf" without a labelling campaign |
| Encoder backbone | Swin-base (2021) | SigLIP 2, or Marqo-FashionSigLIP | Fashion-pretrained; sigmoid loss scales to much larger batches than softmax |
| Embedding size | Fixed 128 | 512 with Matryoshka training | One model serves 64-d for cheap recall and 512-d for reranking |
| Loss | InfoNCE, in-batch negatives | Same, plus mined hard negatives | In-batch negatives get too easy after a few epochs |
| Search | Brute-force numpy | HNSW ANN index with metadata filters | 15k works; 2M does not |
| Precision | float32 | int8 or binary quantised index | 4–32× less memory, tiny accuracy cost |
| Ranking | Single stage | Recall then rerank | Cheap wide net, then expensive precise scoring on the top 200 |
| Text | None | Joint image–text space | "same jacket but in black" |
| Evaluation | 9 example images | Recall@k, MRR, NDCG@k, per category | The whole point |
| Packaging | Jupyter notebooks | Modules, tests, CI, bundles | Notebooks do not deploy |

---

## 4. How a search actually works at runtime

```
User uploads a photo
      │
      ▼
[1] DETECTOR ─────────► boxes: {top, bag, shoes} with confidences
      │
      ▼
[2] CROP + CHOOSE ────► user taps the jacket, or we default to the largest/most central item
      │
      ▼
[3] ENCODER ──────────► one 512-number embedding for that crop
      │
      ├──── truncate to 64 numbers (Matryoshka) ───┐
      │                                            ▼
      │                              [4] ANN RECALL over the whole
      │                                  catalogue, filtered to
      │                                  category = "outer" and
      │                                  in_stock = true
      │                                  → top 500 candidates
      │                                            │
      └──── full 512 numbers ─────────────────────►│
                                                   ▼
                                    [5] RERANK the 500 by exact
                                        similarity, then blend in
                                        business signals
                                                   │
                                                   ▼
                                        [6] TOP 20 RESULTS
```

Steps 4 and 5 are the part the original does not have, and they are the difference between a
notebook and a service. The cheap first pass looks at everything using short embeddings; the
expensive second pass looks at 500 things using full embeddings. Total cost is dominated by the
cheap pass, total quality is dominated by the expensive one.

---

## 5. Unity Catalog layout

```
fashion                                  -- catalog
├── raw
│   ├── VOLUME product_images            -- catalogue thumbnails
│   ├── VOLUME post_images               -- styled/user photos
│   └── VOLUME query_images              -- real queries from production
├── bronze
│   ├── products                         -- one row per catalogue item
│   ├── posts                            -- one row per styled photo + its linked products
│   └── search_events                    -- production queries, results shown, clicks
├── silver
│   ├── detections                       -- every detected item in every post image
│   ├── pairs                            -- anchor(crop) ↔ positive(thumbnail) training pairs
│   ├── product_embeddings               -- catalogue vectors (backs the search index)
│   └── hard_negatives                   -- mined confusions, refreshed each cycle
├── gold
│   ├── train_pairs                       -- frozen, versioned training manifest
│   ├── eval_queries                      -- frozen eval set WITH relevance judgments
│   └── retrieval_metrics                 -- Recall@k / MRR / NDCG per model per slice
└── ml
    ├── MODEL fashion_detector
    ├── MODEL fashion_encoder
    └── INDEX product_search_index         -- Databricks Vector Search
```

---

## 6. Building the training pairs

`src/curate/03_build_pairs.py`. The logic mirrors the original but adds the things that stop it
degrading over time.

1. Run the detector over every post image → `silver.detections`.
2. For each post, match each detection to the linked product by **category agreement**.
3. Reject pairs where detector confidence is low, the crop is tiny, or the crop is mostly
   occluded — a 30×40 pixel blur teaches the encoder nothing useful.
4. **Deduplicate.** Crawled catalogues are full of the same product listed by three sellers. If
   near-duplicate products end up split between train and eval, your eval score is inflated and
   you will not find out until production disagrees with you.
5. Split by **product ID, never by row.** Every crop of product X goes to one side of the split.
   Splitting randomly leaks the answer.

That fifth point is the single most common way visual search evaluations end up wrong, and it is
completely invisible until you deploy.

---

## 7. The search index

Databricks Vector Search with a Delta Sync index over `silver.product_embeddings`. Sync is
triggered after each catalogue refresh, so new products become searchable without a rebuild.

**Metadata filters do more work than people expect.** Filtering to `category = 'outer' AND
in_stock = true AND region = 'KR'` before the vector search runs shrinks the candidate pool by
an order of magnitude, which makes the search both faster and more accurate — the index is no
longer offered the chance to return a beautiful match that is out of stock in the wrong country.

**Quantisation.** Store the index as int8 rather than float32: a quarter of the memory, and the
recall loss is well under a percent for embeddings that were L2-normalised during training.
Binary quantisation (1 bit per dimension) gives 32× and is viable if you rerank the top 500 with
full-precision vectors afterwards, which you are doing anyway.

**Index freshness is a monitored metric, not an assumption.** A fashion catalogue turns over
constantly. An index four days stale returns sold-out products, which users experience as the
search being broken.

---

## 8. Evaluation — the part that was missing

### 8.1 Build the evaluation set once

Take 1,000–2,000 real query images. For each, have a human mark which catalogue products are
correct matches. Store these **relevance judgments** in `gold.eval_queries`. Then freeze it and
never train on it.

Two or three days of work. Everything below becomes possible, and stays possible for years.

### 8.2 The three metrics

**Recall@k** — of all the correct products, what fraction appeared in the top *k*? Recall@20
answers "if the user scrolls one screen, do they see it?" This is the primary metric.

**MRR (Mean Reciprocal Rank)** — 1 divided by the rank of the first correct result, averaged.
First place scores 1.0, fifth scores 0.2. Rewards getting one thing exactly right.

**NDCG@k** — like Recall@k but position-weighted and able to handle graded relevance ("exact
match" scoring higher than "very similar"). The most complete of the three and the one to report
if you report only one.

### 8.3 Slices, again

Report every metric **per category and per query condition**, never only as one number:

| Dimension | Slices |
|---|---|
| Category | top, bottom, outer, dress, shoes, bag, hat |
| Query source | clean catalogue shot, styled post, user phone photo, screenshot |
| Item size in frame | large (>30% of image), medium, small (<8%) |
| Occlusion | unoccluded, partly hidden, heavily cropped |

Aggregate numbers are dominated by whichever category has the most queries. Bags and hats are
usually small and rare, so they are exactly what an average will hide — and hats are where a
generic encoder fails hardest, because hats are small, similar, and heavily occluded by hair.

### 8.4 The promotion gate

`src/evaluate/04_evaluate_retrieval.py` compares the candidate against the current production
model on the frozen eval set and requires all of:

1. NDCG@20 at least as good as the champion, overall.
2. Recall@20 not worse on **any** category slice beyond a small tolerance.
3. Recall@20 not worse on the small-item and occluded slices — protected explicitly, because
   they are rare and an average will always trade them away.
4. Embedding p95 latency within budget on the serving hardware.
5. Every protected slice actually had enough eval queries to measure. Silence is not a pass.

Pass and the model is tagged `@candidate`. A human promotes to `@shadow`, then `@production`.

### 8.5 Online metrics are the real ones

Offline metrics predict; they do not decide. Once deployed, what matters is click-through rate
on search results, add-to-cart rate, zero-result rate, and the fraction of sessions where the
user reformulates (a strong signal the first search failed). `bronze.search_events` records all
of it, and those clicks come back as training data in §9.

---

## 9. The feedback loop

Production search logs are the highest-quality training data available, and they are free.

- A user searches, then clicks result #3 and buys it → that is a **new positive pair**, from the
  real distribution, at zero labelling cost.
- Results #1 and #2, shown and ignored, are **hard negatives**: things the current model thought
  were close but a human rejected. Far more valuable than random negatives, which the model
  learned to separate weeks ago.
- Queries returning nothing clicked point at gaps — an unlisted category, a new style, a
  photography convention the encoder has not seen.

`src/curate/03_build_pairs.py` folds all three back into the next training manifest. This is what
makes a deployed search system improve on its own, and it is why shipping a mediocre version and
measuring it beats polishing offline for another month.

---

## 10. Repo layout

```
fashionsearch-2026/
├── databricks.yml                      # Declarative Automation Bundle
├── resources/
│   ├── jobs_setup.yml                  # run this first
│   ├── jobs_data.yml                   # ingest → detect → pairs
│   ├── jobs_training.yml               # train → evaluate → gate
│   ├── jobs_index.yml                  # embed catalogue → build/sync index
│   └── jobs_monitoring.yml             # drift, freshness, online metrics
├── src/
│   ├── setup/    00_create_workspace_objects.py
│   ├── ingest/   01_bronze_ingest.py
│   ├── curate/   02_detect_items.py     03_build_pairs.py
│   ├── train/    train_encoder.py       losses.py
│   ├── evaluate/ 04_evaluate_retrieval.py
│   ├── index/    05_build_vector_index.py
│   ├── serve/    search_service.py
│   └── monitor/  06_monitor.py
└── .github/workflows/cicd.yml
```

## 11. Getting started

```bash
pip install databricks-cli
databricks auth login --host https://YOUR-WORKSPACE.cloud.databricks.com

# edit databricks.yml: replace the placeholder host with yours
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run fashion_setup -t dev     # creates catalog, schemas, volumes
```

`SETUP.md` walks through this from zero, including GitHub.

## 12. Scope

This is a working skeleton with real API surfaces, not a finished product. The training loop is
deliberately thin — swap in your own. What is meant to be complete and directly usable is the
shape: the pair-construction rules in §6, the two-stage retrieval in §4, and above all the
evaluation in §8, because that is what the original lacks and what takes longest to get right.
