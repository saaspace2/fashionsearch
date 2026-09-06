-- FashionSearch table definitions.
-- Applied by notebooks/00_unity_catalog_setup.py, which substitutes ${catalog}.
-- Kept here rather than inline in the notebook so schema changes show up in a
-- diff and can be applied outside a notebook if needed.

CREATE TABLE IF NOT EXISTS ${catalog}.bronze.products (
    product_id   STRING  COMMENT 'Stable catalogue identifier',
    image_path   STRING  COMMENT 'UC Volume path to the thumbnail',
    category     STRING  COMMENT 'top|bottom|outer|dress|shoes|bag|hat',
    brand        STRING,
    title        STRING,
    price        DOUBLE,
    currency     STRING,
    in_stock     BOOLEAN,
    region       STRING,
    source       STRING  COMMENT 'Where this listing came from',
    license_ok   BOOLEAN COMMENT 'Cleared for model training use',
    ingested_at  TIMESTAMP,
    updated_at   TIMESTAMP
) USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
COMMENT 'The searchable catalogue. One row per product.';

CREATE TABLE IF NOT EXISTS ${catalog}.bronze.posts (
    post_id         STRING,
    image_path      STRING,
    linked_products ARRAY<STRING> COMMENT 'product_ids this post references',
    source          STRING,
    captured_at     TIMESTAMP,
    license_ok      BOOLEAN,
    ingested_at     TIMESTAMP
) USING DELTA
COMMENT 'Styled photos linked to the products they contain. Anchors come from here.';

CREATE TABLE IF NOT EXISTS ${catalog}.bronze.search_events (
    event_id          STRING,
    event_ts          TIMESTAMP,
    session_id        STRING,
    query_image       STRING,
    selected_category STRING,
    model_version     STRING,
    index_version     STRING,
    results_shown     ARRAY<STRING> COMMENT 'product_ids in rank order',
    clicked           ARRAY<STRING>,
    added_to_cart     ARRAY<STRING>,
    latency_ms        DOUBLE,
    n_results         INT,
    reformulated      BOOLEAN COMMENT 'User searched again within the session',
    query_condition   STRING,
    fallback_fired    STRING  COMMENT 'Which degradation path was taken, if any'
) USING DELTA
COMMENT 'Production search log. Closes the feedback loop: clicks become training pairs.';

CREATE TABLE IF NOT EXISTS ${catalog}.gold.retrieval_metrics (
    slice_dim     STRING,
    slice_value   STRING,
    n_queries     BIGINT,
    recall_at_1   DOUBLE, recall_at_5  DOUBLE, recall_at_10 DOUBLE,
    recall_at_20  DOUBLE, recall_at_50 DOUBLE,
    ndcg_at_1     DOUBLE, ndcg_at_5    DOUBLE, ndcg_at_10   DOUBLE,
    ndcg_at_20    DOUBLE, ndcg_at_50   DOUBLE,
    mrr           DOUBLE,
    model_name    STRING,
    model_version STRING,
    evaluated_at  TIMESTAMP,
    gate_passed   BOOLEAN
) USING DELTA
COMMENT 'One row per slice per evaluation. The permanent record of how every model version behaved. This is what you show when someone asks whether it got better.';

CREATE TABLE IF NOT EXISTS ${catalog}.monitoring.alerts (
    payload   STRING,
    kind      STRING,
    severity  STRING,
    raised_at TIMESTAMP
) USING DELTA;
