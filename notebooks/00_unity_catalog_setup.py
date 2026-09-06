# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Unity Catalog setup
# MAGIC
# MAGIC Creates the catalog, schemas, volumes and tables the pipeline needs.
# MAGIC Safe to re-run: everything is `IF NOT EXISTS`.
# MAGIC
# MAGIC **Run this first.** Every later notebook assumes these exist.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from fashionsearch.config import load_config

cfg = load_config()
cat = cfg.catalog.name
print(f"Target catalog: {cat}")

# COMMAND ----------
# MAGIC %md ## Catalog and schemas

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {cat} COMMENT 'FashionSearch visual search'")
for schema in cfg.catalog.schemas:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cat}.{schema}")
    print(f"  schema {cat}.{schema}")

# COMMAND ----------
# MAGIC %md ## Volumes — where images actually live

# COMMAND ----------
for logical, path in dict(cfg.catalog.volumes).items():
    _, _, _, schema, name = path.split("/")[:5]
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {cat}.{schema}.{name}")
    print(f"  volume {cat}.{schema}.{name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Tables
# MAGIC
# MAGIC The DDL lives in `unity_catalog_ddl.sql` rather than inline here, so the
# MAGIC schema is reviewable in a diff and can be applied outside the notebook.

# COMMAND ----------
ddl_path = pathlib.Path.cwd().parent / "unity_catalog_ddl.sql"
statements = [s.strip() for s in ddl_path.read_text().split(";") if s.strip()]

for stmt in statements:
    spark.sql(stmt.replace("${catalog}", cat))

print(f"applied {len(statements)} DDL statements")

# COMMAND ----------
display(spark.sql(f"SHOW TABLES IN {cat}.bronze"))

# COMMAND ----------
# MAGIC %md
# MAGIC Next: **01_ingest_data** downloads the real dataset from Hugging Face.
