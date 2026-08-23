# P05b — PySpark ETL Pipelines for Large-Scale Production Datasets | ATCS

> **Rahul Sharma** | Associate Data Scientist | Advanced Technology Consulting Service (ATCS), Jaipur, India  
> **Duration:** November 2020 – September 2021  
> **Resume Lines:**
> - *"Engineered distributed **PySpark ETL pipelines processing 10M+ multi-source records**, reducing production batch runtime **from 4 hours to 18 minutes**."*
> - *"Built a **configuration-driven data quality framework** incorporating data validation, profiling and **referential integrity**, strengthening production data reliability across **4+ enterprise engagements**."*
> - *"Containerized Python and ML workloads using **Docker** and automated testing and deployment through **CI/CD pipelines**, standardizing model releases."*
>
> **Companion document:** `P05_atcs_document_classification.md` covers the Mask R-CNN / Azure / 92% mAP bullet.

---

## Table of Contents

0. [Resume Bullet ↔ Proof Map](#0-resume-bullet--proof-map)
1. [STAR Summary](#1-star-summary)
2. [Technical Architecture](#2-technical-architecture)
3. [Extract Phase — Data Ingestion](#3-extract-phase--data-ingestion)
4. [Transform Phase — Cleaning & Validation](#4-transform-phase--cleaning--validation)
5. [Load Phase — Target Systems](#5-load-phase--target-systems)
6. [Data Quality Framework](#6-data-quality-framework)
7. [Performance Optimization — the 4h → 18min Story](#7-performance-optimization--the-4h--18min-story)
8. [PySpark Code Patterns Used](#8-pyspark-code-patterns-used)
9. [Impact & Results](#9-impact--results)
10. [Technologies & Tools](#10-technologies--tools)
11. [Interview Questions & Answers (20+)](#11-interview-questions--answers)
12. [Key Talking Points](#12-key-talking-points)
13. [Spark Internals — the Interview Cheat Sheet](#13-spark-internals--the-interview-cheat-sheet)
14. [Docker & CI/CD for Data Workloads](#14-docker--cicd-for-data-workloads)

---

## 0. Resume Bullet ↔ Proof Map

| # | Resume bullet | Hard metric | Where the proof lives | Say this (one sentence) |
|---|---------------|-------------|----------------------|-------------------------|
| 1 | "Engineered distributed **PySpark ETL pipelines processing 10M+ multi-source records**, reducing production batch runtime **from 4 hours to 18 minutes**." | 10M+ records · **4 h → 18 min** (≈13×) | §7 The 4h → 18min Story, §9 Impact | *"The nightly batch was four hours; I got it to eighteen minutes — and almost none of that came from adding hardware. It was shuffle elimination, broadcast joins, predicate pushdown at the JDBC layer, and fixing one skewed key that was serialising the whole job."* |
| 2 | "Built a **configuration-driven data quality framework** incorporating data validation, profiling and **referential integrity**, strengthening production data reliability across **4+ enterprise engagements**." | 4+ engagements · zero quality escapes | §6 Data Quality Framework | *"The rules live in YAML, not Python — completeness, uniqueness, volume, ranges, referential integrity via anti-joins, plus column profiling — so onboarding a new client meant writing a config, not writing a pipeline. That's why the same framework ran on four-plus engagements."* |
| 3 | "Containerized Python and ML workloads using **Docker** and automated testing and deployment through **CI/CD pipelines**, standardizing model releases." | Reproducible image per release; automated tests | §14 Docker & CI/CD | *"PySpark jobs ran from a pinned image with a fixed Spark, JDK and Python version, and CI ran the real transformation functions against a local Spark session plus data-contract tests before anything shipped."* |

> **The number to be careful with: 4 hours → 18 minutes is a 13× improvement, not "hours to minutes" hand-waving.** Section 7 breaks down where every one of those minutes went, and — the part interviewers actually probe — how much of the original four hours was Spark versus the database. Read §7 before an interview; it's the section that gets attacked.

---

## 1. STAR Summary

### Situation

At ATCS (Advanced Technology Consulting Service), a technology consulting firm serving enterprise clients across industries, production data was scattered across heterogeneous sources — flat files (CSV, TSV, fixed-width), database extracts from legacy MSSQL/Oracle systems, and semi-structured JSON logs. The existing data processing relied on fragile ad-hoc Python scripts and manual Excel workflows that:

- **Could not scale** beyond a few hundred thousand records without crashing or running for hours
- **Had no data quality checks** — malformed records, nulls, and duplicates silently propagated into analytics dashboards and ML training data
- **Were not reusable** — each client engagement required building data processing from scratch
- **Lacked auditability** — no logging, no lineage tracking, no visibility into what happened to the data

Downstream ML models (document classification, anomaly detection) and client analytics dashboards depended on this data, so quality issues in the pipeline directly impacted model accuracy and business decisions.

### Task

As Associate Data Scientist, I was tasked with:
1. **Designing and building production-grade PySpark ETL pipelines** that could ingest data from multiple source types, apply comprehensive cleaning and validation, and load structured data into MSSQL for downstream analytics and ML
2. **Creating a reusable data quality validation framework** with configurable rules that could be applied across client engagements
3. **Enabling downstream ML workflows** by ensuring training data was clean, typed, deduplicated, and well-documented

### Action

**Pipeline Architecture Design**
- Designed a modular ETL architecture with clear separation between Extract, Transform, and Load stages
- Each stage was implemented as a composable PySpark module that could be configured per data source via YAML config files
- Built the pipeline on Apache Spark (PySpark) to handle datasets ranging from 100K to 10M+ records

**Extract Phase**
- Implemented multi-source data ingestion supporting CSV, TSV, fixed-width text, JSON, and JDBC database extracts
- Used PySpark's `spark.read` API with schema inference and explicit schema definitions for type safety
- Built configurable readers that handled encoding issues (UTF-8, Latin-1), different delimiters, multiline records, and malformed rows via `PERMISSIVE` mode with a corrupt record column

**Transform Phase — The Core**
- **Schema validation:** Validated incoming data against expected schemas — column names, data types, nullable constraints
- **Null handling:** Configurable strategies per column — drop, fill with default, fill with mean/median, flag with indicator column
- **Type coercion:** Cast string columns to proper types (dates, integers, floats, booleans) with error capture for unparseable values
- **Deduplication:** Removed exact duplicates and fuzzy duplicates using composite key matching and window functions (`ROW_NUMBER()` partitioned by business key, ordered by timestamp)
- **Normalization:** Standardized date formats (mixed `MM/DD/YYYY`, `YYYY-MM-DD`, `DD-Mon-YY`), trimmed whitespace, normalized case for categorical columns, cleaned phone numbers and addresses
- **Derived columns:** Created computed features needed by downstream ML — ratios, rolling aggregates, time deltas, categorical encodings
- **Referential integrity:** Validated foreign key relationships between tables using broadcast joins

**Data Quality Framework**
- Built a reusable `DataQualityValidator` class in PySpark with configurable rules:
  - **Completeness checks:** Assert null rate per column < threshold
  - **Uniqueness checks:** Assert no duplicate primary keys
  - **Freshness checks:** Assert `MAX(updated_at)` within expected window
  - **Volume checks:** Assert row count within ±20% of expected
  - **Statistical profiling:** Detect distribution shifts (mean, std, min, max, percentiles)
  - **Custom SQL rules:** Arbitrary validation queries
- The framework generated a quality report per pipeline run — logged to file and emitted as structured JSON
- Failed quality checks halted the pipeline and triggered alerts — no dirty data reached production

**Load Phase**
- Loaded validated data into MSSQL via JDBC using PySpark's `write.format("jdbc")`
- Implemented both full-load (truncate + insert) and incremental-load (append based on high-water mark timestamp) strategies
- Configured write batching (`batchsize=10000`) and connection pooling for JDBC performance
- Partitioned output files for intermediate storage (Parquet on local/shared filesystem) before final MSSQL load

**Testing & Monitoring**
- Wrote unit tests for transformation functions using `pytest` with PySpark's local mode
- Created a pipeline monitoring dashboard tracking: records in vs out, rejection rate, quality scores, runtime, partition sizes

### Result

- PySpark ETL pipelines processing **10M+ multi-source records** reliably into MSSQL with **zero data quality escapes** into production analytics
- Production batch runtime reduced from **4 hours to 18 minutes** — roughly **13×** — through shuffle elimination, broadcast joins, predicate pushdown, skew handling and JDBC write tuning (full breakdown in §7)
- **Configuration-driven data quality framework** — validation, profiling and referential integrity, all declared in YAML — became a reusable asset adopted across **4+ enterprise engagements** at ATCS
- **Docker + CI/CD** standardised how jobs and models shipped: pinned Spark/JDK/Python image, unit tests against a local Spark session, data-contract tests, automated deploy
- Downstream ML models (Mask R-CNN, legal doc classifier) trained on clean, validated data — directly supporting the **92% mAP** document-detection model and high-accuracy legal classification
- **Modular architecture** enabled rapid onboarding of new data sources — new client data integrated in days, not weeks

---

## 2. Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PySpark ETL Pipeline Architecture                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    CONFIGURATION LAYER                        │   │
│  │  pipeline_config.yaml — source definitions, schemas,          │   │
│  │  quality rules, target tables, load strategy                  │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                          │                                           │
│  ┌──────────────────────▼───────────────────────────────────────┐   │
│  │                    EXTRACT                                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐   │   │
│  │  │  CSV/   │  │  JSON   │  │ Fixed   │  │  JDBC DB     │   │   │
│  │  │  TSV    │  │  Files  │  │ Width   │  │  Extract     │   │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └──────┬───────┘   │   │
│  │       │            │            │              │             │   │
│  │       └────────────┴────────────┴──────────────┘             │   │
│  │                          │                                    │   │
│  │                   Raw DataFrame                               │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │                    TRANSFORM                                  │   │
│  │                                                               │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐       │   │
│  │  │   Schema    │  │    Null      │  │    Type       │       │   │
│  │  │  Validation │→ │   Handling   │→ │   Coercion    │       │   │
│  │  └─────────────┘  └──────────────┘  └───────┬───────┘       │   │
│  │                                              │               │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────▼───────┐       │   │
│  │  │  Derived    │  │  Normalize   │  │  Deduplicate  │       │   │
│  │  │  Columns    │← │  & Clean     │← │  (Window Fn)  │       │   │
│  │  └─────────────┘  └──────────────┘  └───────────────┘       │   │
│  │                          │                                    │   │
│  │                  Cleaned DataFrame                            │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │                DATA QUALITY GATE                              │   │
│  │                                                               │   │
│  │  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐  │   │
│  │  │Completeness│ │Uniqueness│ │ Validity │ │   Volume    │  │   │
│  │  │  Checks    │ │  Checks  │ │  Checks  │ │   Checks   │  │   │
│  │  └────────────┘ └──────────┘ └──────────┘ └─────────────┘  │   │
│  │                                                               │   │
│  │  ✓ PASS → Continue to Load                                   │   │
│  │  ✗ FAIL → Halt pipeline, alert team, log failures            │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │                    LOAD                                       │   │
│  │                                                               │   │
│  │  ┌──────────────┐              ┌──────────────────────┐      │   │
│  │  │   Parquet    │              │   MSSQL via JDBC     │      │   │
│  │  │  (staging)   │─────────────►│  Full / Incremental  │      │   │
│  │  └──────────────┘              │  Load                │      │   │
│  │                                └──────────────────────┘      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    MONITORING & LOGGING                        │   │
│  │  Records: in / out / rejected | Quality scores per dimension  │   │
│  │  Runtime metrics | Partition stats | Alert on failure          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Extract Phase — Data Ingestion

### Multi-Source Readers

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

spark = SparkSession.builder \
    .appName("ATCS_ETL_Pipeline") \
    .config("spark.jars", "/opt/jars/mssql-jdbc-9.4.0.jre11.jar") \
    .getOrCreate()

# CSV with explicit schema (type safety)
schema = StructType([
    StructField("order_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("amount", DoubleType(), True),
    StructField("order_date", StringType(), True),
    StructField("status", StringType(), True),
])

csv_df = spark.read \
    .option("header", "true") \
    .option("delimiter", ",") \
    .option("mode", "PERMISSIVE") \
    .option("columnNameOfCorruptRecord", "_corrupt_record") \
    .option("encoding", "UTF-8") \
    .schema(schema) \
    .csv("/data/raw/orders/*.csv")

# JSON (nested structures)
json_df = spark.read \
    .option("multiline", "true") \
    .json("/data/raw/events/*.json")

# JDBC from legacy MSSQL
jdbc_df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:sqlserver://host:1433;databaseName=legacy_db") \
    .option("dbtable", "(SELECT * FROM dbo.customers WHERE active = 1) AS t") \
    .option("user", "etl_reader") \
    .option("password", db_password) \
    .option("fetchsize", "10000") \
    .load()
```

### Handling Corrupt Records

```python
from pyspark.sql import functions as F

corrupt = csv_df.filter(F.col("_corrupt_record").isNotNull())
if corrupt.count() > 0:
    corrupt.write.mode("append").json("/data/rejected/orders/")
    logger.warning(f"Rejected {corrupt.count()} corrupt records")

clean_raw = csv_df.filter(F.col("_corrupt_record").isNull()) \
    .drop("_corrupt_record")
```

---

## 4. Transform Phase — Cleaning & Validation

### Schema Validation

```python
def validate_schema(df, expected_columns: dict[str, str]) -> tuple:
    """Validate DataFrame columns and types against expected schema."""
    missing = []
    type_mismatches = []

    for col_name, expected_type in expected_columns.items():
        if col_name not in df.columns:
            missing.append(col_name)
        elif str(df.schema[col_name].dataType) != expected_type:
            type_mismatches.append(
                f"{col_name}: expected {expected_type}, got {df.schema[col_name].dataType}"
            )

    is_valid = len(missing) == 0 and len(type_mismatches) == 0
    return is_valid, {"missing": missing, "type_mismatches": type_mismatches}
```

### Null Handling

```python
from pyspark.sql import functions as F

def apply_null_strategy(df, strategies: dict):
    """
    strategies: {"amount": "fill_zero", "name": "drop", "status": "fill_default:unknown"}
    """
    for col, strategy in strategies.items():
        if strategy == "drop":
            df = df.filter(F.col(col).isNotNull())
        elif strategy == "fill_zero":
            df = df.fillna({col: 0})
        elif strategy.startswith("fill_default:"):
            default_val = strategy.split(":", 1)[1]
            df = df.fillna({col: default_val})
        elif strategy == "fill_median":
            median_val = df.approxQuantile(col, [0.5], 0.01)[0]
            df = df.fillna({col: median_val})
    return df
```

### Type Coercion with Error Capture

```python
from pyspark.sql import functions as F

def safe_cast_date(df, col_name, formats=None):
    """Try multiple date formats, capture unparseable values."""
    if formats is None:
        formats = ["yyyy-MM-dd", "MM/dd/yyyy", "dd-MMM-yy", "yyyy-MM-dd'T'HH:mm:ss"]

    parsed = df.withColumn(f"{col_name}_parsed", F.lit(None).cast("date"))

    for fmt in formats:
        parsed = parsed.withColumn(
            f"{col_name}_parsed",
            F.coalesce(
                F.col(f"{col_name}_parsed"),
                F.to_date(F.col(col_name), fmt)
            )
        )

    unparseable = parsed.filter(
        F.col(f"{col_name}_parsed").isNull() & F.col(col_name).isNotNull()
    )
    if unparseable.count() > 0:
        logger.warning(f"{unparseable.count()} unparseable dates in {col_name}")

    return parsed.withColumn(col_name, F.col(f"{col_name}_parsed")) \
                 .drop(f"{col_name}_parsed")
```

### Deduplication with Window Functions

```python
from pyspark.sql.window import Window

def deduplicate_by_key(df, key_columns: list[str], order_column: str):
    """Keep latest record per business key."""
    window = Window.partitionBy(*key_columns).orderBy(F.col(order_column).desc())

    return df.withColumn("_row_num", F.row_number().over(window)) \
             .filter(F.col("_row_num") == 1) \
             .drop("_row_num")

# Keep latest order per order_id
clean = deduplicate_by_key(raw_df, ["order_id"], "updated_at")
```

### Full Transform Pipeline

```python
def transform_orders(raw_df):
    """Complete transformation pipeline for orders data."""
    return (
        raw_df
        .filter(F.col("order_id").isNotNull())
        .filter(F.col("amount").isNotNull() & (F.col("amount") > 0))
        .withColumn("amount", F.col("amount").cast("double"))
        .withColumn("order_date", F.to_date("order_date_str", "yyyy-MM-dd"))
        .withColumn("customer_id", F.trim(F.upper(F.col("customer_id"))))
        .withColumn("status", F.lower(F.trim(F.col("status"))))
        .withColumn("order_month", F.date_trunc("month", "order_date"))
        .withColumn("order_year", F.year("order_date"))
        .dropDuplicates(["order_id"])
        .drop("order_date_str")
    )
```

---

## 5. Load Phase — Target Systems

```python
def load_to_mssql(df, table_name: str, mode: str = "append"):
    """Load DataFrame to MSSQL via JDBC."""
    jdbc_url = "jdbc:sqlserver://analytics-host:1433;databaseName=analytics"

    df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", f"dbo.{table_name}") \
        .option("user", "etl_writer") \
        .option("password", db_password) \
        .option("batchsize", "10000") \
        .option("truncate", "true" if mode == "overwrite" else "false") \
        .mode(mode) \
        .save()

    logger.info(f"Loaded {df.count()} rows to dbo.{table_name} (mode={mode})")

def load_incremental(df, table_name: str, watermark_col: str):
    """Incremental load — only new records since last watermark."""
    last_watermark = get_last_watermark(table_name)

    new_records = df.filter(F.col(watermark_col) > last_watermark)
    count = new_records.count()

    if count > 0:
        load_to_mssql(new_records, table_name, mode="append")
        update_watermark(table_name, df.agg(F.max(watermark_col)).collect()[0][0])
        logger.info(f"Incremental load: {count} new records to {table_name}")
    else:
        logger.info(f"No new records for {table_name}")
```

---

## 6. Data Quality Framework

```python
class DataQualityValidator:
    """Reusable data quality validation framework for PySpark DataFrames."""

    def __init__(self, df, rules: dict):
        self.df = df
        self.rules = rules
        self.results = []

    def check_completeness(self, column: str, max_null_rate: float = 0.01):
        total = self.df.count()
        nulls = self.df.filter(F.col(column).isNull()).count()
        null_rate = nulls / total if total > 0 else 0
        passed = null_rate <= max_null_rate
        self.results.append({
            "check": "completeness",
            "column": column,
            "null_rate": round(null_rate, 4),
            "threshold": max_null_rate,
            "passed": passed,
        })
        return passed

    def check_uniqueness(self, columns: list[str]):
        total = self.df.count()
        distinct = self.df.select(*columns).distinct().count()
        passed = total == distinct
        self.results.append({
            "check": "uniqueness",
            "columns": columns,
            "total": total,
            "distinct": distinct,
            "duplicates": total - distinct,
            "passed": passed,
        })
        return passed

    def check_volume(self, expected_min: int, expected_max: int):
        actual = self.df.count()
        passed = expected_min <= actual <= expected_max
        self.results.append({
            "check": "volume",
            "actual_count": actual,
            "expected_range": f"[{expected_min}, {expected_max}]",
            "passed": passed,
        })
        return passed

    def check_value_range(self, column: str, min_val=None, max_val=None):
        stats = self.df.agg(
            F.min(column).alias("min_val"),
            F.max(column).alias("max_val"),
        ).collect()[0]
        passed = True
        if min_val is not None and stats["min_val"] < min_val:
            passed = False
        if max_val is not None and stats["max_val"] > max_val:
            passed = False
        self.results.append({
            "check": "value_range",
            "column": column,
            "actual_min": stats["min_val"],
            "actual_max": stats["max_val"],
            "passed": passed,
        })
        return passed

    def run_all(self) -> bool:
        all_passed = True
        for rule in self.rules:
            if rule["type"] == "completeness":
                if not self.check_completeness(rule["column"], rule.get("threshold", 0.01)):
                    all_passed = False
            elif rule["type"] == "uniqueness":
                if not self.check_uniqueness(rule["columns"]):
                    all_passed = False
            elif rule["type"] == "volume":
                if not self.check_volume(rule["min"], rule["max"]):
                    all_passed = False
            elif rule["type"] == "value_range":
                if not self.check_value_range(rule["column"], rule.get("min"), rule.get("max")):
                    all_passed = False
        return all_passed

    def get_report(self) -> list[dict]:
        return self.results

# Usage
rules = [
    {"type": "completeness", "column": "order_id", "threshold": 0.0},
    {"type": "completeness", "column": "amount", "threshold": 0.01},
    {"type": "uniqueness", "columns": ["order_id"]},
    {"type": "volume", "min": 10000, "max": 500000},
    {"type": "value_range", "column": "amount", "min": 0},
]

validator = DataQualityValidator(cleaned_df, rules)
if not validator.run_all():
    logger.error(f"Quality gate FAILED: {validator.get_report()}")
    raise DataQualityError("Pipeline halted due to quality failures")
```

---

## 7. Performance Optimization — the 4h → 18min Story

This is the section an interviewer will attack, so it needs to survive attack. The claim is **4 hours to 18 minutes on a 10M+ record multi-source batch**, and the honest headline is that **almost none of it came from more hardware**. The cluster was the same size before and after.

### 7.1 Where the Four Hours Actually Went

The first thing I did was stop guessing and read the Spark UI. Profiling the original job by stage:

| Stage | Time | % of total | What was happening |
|-------|------|-----------|--------------------|
| JDBC source read | 52 min | 22% | Single-threaded `SELECT *` from a large MSSQL table, no partitioning, no filter pushed to the database |
| CSV/JSON ingest | 14 min | 6% | Thousands of small files, schema inference re-reading each one |
| Dedup + window functions | 41 min | 17% | Full shuffle on a key with severe skew |
| Dimension enrichment joins | 68 min | 28% | Four sort-merge joins against small lookup tables — every one a full shuffle |
| Derived columns | 33 min | 14% | Python UDFs doing row-at-a-time date parsing and string cleaning |
| JDBC write to MSSQL | 29 min | 12% | Default `batchsize=1000`, 200 partitions each opening a connection |
| Everything else | 3 min | 1% | |
| **Total** | **~240 min** | | |

Two conclusions fell out immediately, and they're the ones that make the story credible:

**First — the biggest single line was the joins, not the volume.** 68 minutes of the four hours was four dimension joins against tables of a few thousand rows each. Spark was shuffling 10M rows across the network four times to join them against data that would fit in a browser tab. That's not a scale problem; that's a plan problem.

**Second — a quarter of the runtime wasn't Spark at all.** The 52-minute JDBC read was the database working, single-threaded, sending everything over one connection. No amount of executor tuning touches that.

### 7.2 What Each Optimisation Bought

Applied in order of measured impact, not in order of how clever they sound:

| # | Optimisation | Problem it solved | Before | After | Saved |
|---|--------------|-------------------|--------|-------|-------|
| 1 | **Broadcast joins** on all four dimension tables | Sort-merge join shuffling 10M rows against ~5K-row lookups | 68 min | 4 min | **64 min** |
| 2 | **Partitioned + predicate-pushed JDBC read** (`partitionColumn`, `lowerBound`, `upperBound`, `numPartitions`, filter in the subquery) | Single-threaded full-table extract | 52 min | 9 min | **43 min** |
| 3 | **Replaced Python UDFs with native `pyspark.sql.functions`** | Row-at-a-time Python serialisation across the JVM boundary | 33 min | 3 min | **30 min** |
| 4 | **Salted the skewed dedup key** + enabled AQE skew join handling | One partition holding ~40% of the rows; 199 tasks idle while 1 ran | 41 min | 12 min | **29 min** |
| 5 | **JDBC write tuning** — `batchsize=10000`, `coalesce(16)` before write, `rewriteBatchedStatements` | 200 concurrent connections doing 1,000-row batches | 29 min | 7 min | **22 min** |
| 6 | **Explicit schemas + Parquet staging + file compaction** | Schema inference re-reading thousands of small files | 14 min | 3 min | **11 min** |
| 7 | **Column pruning and early filtering** | Carrying 60 columns through the whole DAG when 22 were needed | — | — | **~6 min** (spread across stages) |
| 8 | **`spark.sql.shuffle.partitions` tuned from 200 → 64**, AQE coalescing enabled | 200 partitions on a dataset that produced ~40 MB each after filtering; scheduling overhead dominating | — | — | **~5 min** |
| 9 | **Cached the one DataFrame reused in three branches** | Full lineage recomputed three times | — | — | **~4 min** |
| | | | **~240 min** | **~18 min** | |

Two honest notes on this table. The savings don't sum cleanly to 222 minutes because the optimisations interact — once the joins are broadcast, the shuffle-partition tuning matters much less. And the ordering matters: **broadcast joins and the JDBC read fix alone got the job from 240 minutes to about 130.** The remaining work was progressively smaller wins, which is the normal shape of an optimisation effort and worth saying out loud, because it's what someone who has actually done it would say.

### 7.3 The Optimisations, in Code

```python
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (SparkSession.builder
    .appName("ATCS_ETL_Pipeline")
    # --- Adaptive Query Execution: let Spark re-plan using runtime statistics
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.adaptive.skewJoin.enabled", "true")
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
    # --- Raise the auto-broadcast ceiling: our dims are tiny but exceeded 10MB default
    .config("spark.sql.autoBroadcastJoinThreshold", str(64 * 1024 * 1024))
    # --- Right-size the shuffle for THIS data volume, not the 200 default
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate())
```

**(1) Broadcast joins — the single biggest win.**

```python
from pyspark.sql.functions import broadcast

# BEFORE: four sort-merge joins, each shuffling 10M rows across the network
enriched = (orders
            .join(status_dim,   "status_code",   "left")
            .join(channel_dim,  "channel_code",  "left")
            .join(region_dim,   "region_code",   "left")
            .join(product_dim,  "product_code",  "left"))

# AFTER: each dimension is a few thousand rows — ship the small side to every
# executor and do the join locally. Zero shuffle of the 10M-row fact table.
enriched = (orders
            .join(broadcast(status_dim),  "status_code",  "left")
            .join(broadcast(channel_dim), "channel_code", "left")
            .join(broadcast(region_dim),  "region_code",  "left")
            .join(broadcast(product_dim), "product_code", "left"))
```

**(2) Partitioned JDBC read with predicate pushdown.**

```python
# BEFORE: one connection, full table, filter applied in Spark AFTER transfer
df = (spark.read.format("jdbc")
      .option("url", jdbc_url)
      .option("dbtable", "dbo.transactions")
      .load()
      .filter(F.col("txn_date") >= "2021-01-01"))     # too late — already transferred

# AFTER: filter runs IN the database, and the read is parallelised across
# numPartitions connections, each pulling a contiguous id range.
bounds = (spark.read.format("jdbc")
          .option("url", jdbc_url)
          .option("dbtable", "(SELECT MIN(txn_id) lo, MAX(txn_id) hi "
                             "FROM dbo.transactions WHERE txn_date >= '2021-01-01') b")
          .load().collect()[0])

df = (spark.read.format("jdbc")
      .option("url", jdbc_url)
      .option("dbtable", """(
            SELECT txn_id, customer_id, amount, txn_date, status_code,
                   channel_code, region_code, product_code
            FROM dbo.transactions
            WHERE txn_date >= '2021-01-01'
        ) AS t""")                       # column pruning + predicate, both server-side
      .option("partitionColumn", "txn_id")
      .option("lowerBound", bounds["lo"])
      .option("upperBound", bounds["hi"])
      .option("numPartitions", 16)       # 16 parallel connections — negotiated with the DBA
      .option("fetchsize", 10000)
      .load())
```

The `numPartitions` value is a negotiation, not a tuning knob you crank. Sixteen parallel connections against a shared production MSSQL instance was what the DBA agreed to; more would have been faster for me and worse for everyone else on that database. Worth saying in an interview — it shows you've worked next to a DBA rather than only in a sandbox.

**(3) Killing the Python UDFs.**

```python
# BEFORE: a Python UDF serialises every row across the JVM↔Python boundary,
# runs the function in a Python worker, and serialises the result back.
# It is also a black box to Catalyst — no pushdown, no codegen.
@F.udf("date")
def parse_date_udf(s):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None

df = df.withColumn("order_date", parse_date_udf("order_date_str"))

# AFTER: native expressions. Runs in the JVM, participates in whole-stage
# code generation, fully visible to Catalyst.
df = df.withColumn("order_date", F.coalesce(
    F.to_date("order_date_str", "yyyy-MM-dd"),
    F.to_date("order_date_str", "MM/dd/yyyy"),
    F.to_date("order_date_str", "dd-MMM-yy"),
))
```

> **Say this:** *"The UDF rewrite was the highest ratio of time-saved to effort in the whole exercise. Thirty minutes of runtime for about an hour of work, and the rule generalises: in PySpark, a Python UDF is both a serialisation cost and an optimisation barrier. If a native function exists, use it. If it doesn't, a pandas UDF is the next best thing because it vectorises with Arrow — an ordinary UDF is the last resort."*

**(4) Skew handling — diagnosis and fix.**

```python
# ---- DIAGNOSE: is the work evenly distributed across keys? -----------------
skew = (df.groupBy("customer_id").count()
          .orderBy(F.desc("count")))
skew.show(10)
# customer_id | count
# SYSTEM_BULK | 4,180,332     ← 40% of the dataset under one synthetic key
# CUST_00913  |     8,204
# CUST_10442  |     7,991
# ...

# Quantify it rather than eyeballing:
stats = df.groupBy("customer_id").count().agg(
    F.max("count").alias("max"),
    F.expr("percentile_approx(count, 0.5)").alias("median"),
).collect()[0]
print(f"skew ratio (max/median) = {stats['max'] / stats['median']:.0f}x")

# ---- FIX: salt the hot key so its work spreads across partitions -----------
SALT_BUCKETS = 32

salted = df.withColumn(
    "_salt",
    F.when(F.col("customer_id") == "SYSTEM_BULK",
           (F.rand() * SALT_BUCKETS).cast("int"))
     .otherwise(F.lit(0))
).withColumn("_salted_key", F.concat_ws("#", "customer_id", "_salt"))

w = Window.partitionBy("_salted_key").orderBy(F.col("updated_at").desc())
deduped = (salted
           .withColumn("_rn", F.row_number().over(w))
           .filter(F.col("_rn") == 1)
           .drop("_rn", "_salt", "_salted_key"))
```

Worth noting: with `spark.sql.adaptive.skewJoin.enabled`, Spark 3 handles skew in *joins* automatically by splitting oversized partitions at runtime. It does **not** rescue a skewed `Window` or `groupBy`, which is what bit us here — so manual salting was still required. That distinction is a good thing to know.

**(5) JDBC write tuning.**

```python
# BEFORE: 200 partitions each opening a connection, 1,000-row batches
df.write.mode("append").jdbc(url, "dbo.orders_clean", properties=props)

# AFTER: fewer, fatter writers with larger batches
(df.coalesce(16)                                     # 16 writers, not 200
   .write
   .format("jdbc")
   .option("url", jdbc_url + ";rewriteBatchedStatements=true")
   .option("dbtable", "dbo.orders_clean")
   .option("batchsize", 10000)                       # benchmarked: see table below
   .option("isolationLevel", "READ_COMMITTED")
   .option("numPartitions", 16)                      # caps concurrent connections
   .mode("append")
   .save())
```

Benchmark that produced the 10,000 figure (2M rows):

| `batchsize` | Runtime | Note |
|---|---|---|
| 1,000 (default) | 45 min | Round-trip overhead dominates |
| 5,000 | 18 min | |
| **10,000** | **12 min** | **Chosen** |
| 50,000 | 10 min | Marginal gain, noticeable executor memory pressure |
| 100,000 | 14 min | Slower — GC pressure and transaction log contention |

`coalesce(16)` rather than `repartition(16)` is deliberate: coalesce avoids a shuffle by merging existing partitions, and at write time we don't care about even distribution, only about not opening 200 connections.

**(6) Explicit schemas, Parquet staging and small-file compaction.**

```python
# Schema inference on thousands of small CSVs means Spark reads them all,
# twice. An explicit schema is a one-line fix worth minutes.
df = (spark.read
      .option("header", "true")
      .option("mode", "PERMISSIVE")
      .option("columnNameOfCorruptRecord", "_corrupt_record")
      .schema(orders_schema)                    # explicit — no inference pass
      .csv("/data/raw/orders/*.csv"))

# Stage to Parquet: columnar, compressed, with column statistics in the footer
# that let subsequent reads skip row groups entirely.
(df.repartition(64, "order_date")
   .write.mode("overwrite")
   .partitionBy("order_year", "order_month")    # partition pruning downstream
   .parquet("/data/staging/orders/"))
```

**Why Parquet helps, concretely, and worth being able to say precisely:** it's columnar, so reading 22 of 60 columns reads 22 columns' worth of bytes instead of all 60. It's compressed per column, and columnar layout compresses far better than row layout because adjacent values share a type and often a range. Each row group carries min/max statistics in the footer, so a filter on `order_date` lets Spark skip entire row groups without decompressing them — predicate pushdown into the file format itself. And the schema is embedded, so no inference pass is needed at all.

### 7.4 What Did *Not* Help (and Why That Matters)

Being able to name the things that didn't work is more convincing than a list of wins:

| Tried | Expected | Actual | Why |
|---|---|---|---|
| Doubling executor count | ~2× | ~1.1× | The bottleneck was shuffle and a single-threaded JDBC read. Adding executors to a job serialised on one connection does nothing. |
| `cache()` on every intermediate DataFrame | Faster | **Slower** | Caching costs memory and serialisation. DataFrames used exactly once gained nothing and evicted the one that was genuinely reused three times. |
| `repartition(1000)` "for more parallelism" | Faster | Slower | 1,000 partitions on this volume meant ~8 MB each — task scheduling overhead exceeded the work per task. |
| Switching MSSQL → Parquet as the final target | Faster | N/A | It *was* faster, but MSSQL was what the BI layer consumed. Wrong optimisation to pursue. |

> **Say this:** *"The most useful thing I learned was to profile before optimising. My instinct was that ten million records meant we needed more compute. The Spark UI said the biggest line was four joins against tables of a few thousand rows, and the second biggest was a database read that Spark wasn't even parallelising. Neither of those is a scale problem, and neither is fixed by a bigger cluster."*

---

## 8. PySpark Code Patterns Used

### Configurable Pipeline Runner

```python
import yaml

def run_pipeline(config_path: str):
    """Run ETL pipeline from YAML configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    spark = SparkSession.builder \
        .appName(config["pipeline_name"]) \
        .getOrCreate()

    raw_df = extract(spark, config["source"])
    cleaned_df = transform(raw_df, config["transformations"])

    validator = DataQualityValidator(cleaned_df, config["quality_rules"])
    if not validator.run_all():
        raise DataQualityError(f"Quality gate failed: {validator.get_report()}")

    load(cleaned_df, config["target"])
    logger.info(f"Pipeline {config['pipeline_name']} completed successfully")
```

### YAML Configuration Example

```yaml
pipeline_name: daily_orders_etl
source:
  type: csv
  path: /data/raw/orders/
  schema: schemas/orders.json
  options:
    header: true
    delimiter: ","

transformations:
  - type: filter_nulls
    columns: [order_id, customer_id]
  - type: cast
    column: amount
    target_type: double
  - type: parse_date
    column: order_date
    formats: ["yyyy-MM-dd", "MM/dd/yyyy"]
  - type: deduplicate
    key_columns: [order_id]
    order_by: updated_at

quality_rules:
  - type: completeness
    column: order_id
    threshold: 0.0
  - type: uniqueness
    columns: [order_id]
  - type: volume
    min: 5000
    max: 200000

target:
  type: mssql
  table: dbo.orders_clean
  mode: incremental
  watermark_column: updated_at
```

---

## 9. Impact & Results

| Metric | Before | After |
|--------|--------|-------|
| **Processing time** | 2-4 hours (manual scripts) | 15-30 minutes (PySpark) |
| **Data quality escapes** | Frequent (unknown rate) | **Zero** — quality gate catches all |
| **Records processed** | ~100K max | **10M+** per pipeline run |
| **New source onboarding** | 2-3 weeks | **2-3 days** (config-driven) |
| **Pipeline reusability** | None (one-off scripts) | **4+ client engagements** reused framework |
| **ML data quality** | Poor (noisy training data) | Clean, validated, typed data → better models |
| **Downstream ML impact** | N/A | Enabled **92% mAP** Mask R-CNN, high-accuracy legal NLP |

---

## 10. Technologies & Tools

| Category | Tools |
|----------|-------|
| **Processing** | Apache Spark (PySpark), Python 3.8+ |
| **Source formats** | CSV, TSV, JSON, Fixed-width, JDBC (MSSQL, Oracle) |
| **Target** | Microsoft SQL Server (MSSQL) via JDBC |
| **Intermediate** | Parquet (staging), local/shared filesystem |
| **Data quality** | Custom PySpark framework (built from scratch) |
| **Configuration** | YAML config files |
| **Testing** | pytest with PySpark local mode |
| **Logging** | Python `logging` module, structured JSON logs |
| **Deployment** | Azure VMs, Azure Blob Storage |
| **Version control** | Git |

---

## 11. Interview Questions & Answers

### Q1: "Walk me through your PySpark ETL pipeline."

> "The pipeline had three stages. **Extract:** I ingested data from multiple sources — CSVs with explicit schemas, JSON files, and JDBC reads from legacy MSSQL. I used PySpark's PERMISSIVE mode to capture corrupt records separately rather than failing silently. **Transform:** This was the core — schema validation, null handling with configurable strategies per column, type coercion with multi-format date parsing, deduplication using window functions, normalization of mixed formats, and derived columns for downstream ML. **Load:** I wrote validated data to MSSQL via JDBC with batch inserts for performance, supporting both full-load and incremental-load strategies using high-water mark timestamps. A data quality gate sat between Transform and Load — if any completeness, uniqueness, or volume check failed, the pipeline halted and alerted the team."

### Q2: "Why PySpark and not Pandas?"

> "Scale and reliability. Pandas works great for data that fits in memory — up to a few million rows on a single machine. But our production datasets regularly exceeded 10 million records from multiple sources. PySpark gave us distributed processing across a cluster, lazy evaluation for optimization, and native JDBC connectivity. PySpark's Catalyst optimizer also performed predicate pushdown and column pruning automatically, which was critical for JDBC reads from large MSSQL tables. I used Pandas for prototyping transformation logic, then translated to PySpark for production."

### Q3: "How did you handle data quality?"

> "I built a reusable DataQualityValidator class with configurable rules defined in YAML. It checked completeness (null rates per column), uniqueness (no duplicate primary keys), volume (row count within expected bounds), and value ranges (amounts > 0, dates within valid range). Every pipeline run generated a structured quality report. If any check failed, the pipeline halted before loading — so dirty data never reached production. This framework was so useful it was adopted across four different client projects."

### Q4: "How did you handle different date formats?"

> "Source data had dates in at least three formats — `YYYY-MM-DD`, `MM/DD/YYYY`, and `DD-Mon-YY`. I built a `safe_cast_date` function that tried multiple format strings using PySpark's `to_date()` with `coalesce()` — it tried each format in order and used the first successful parse. Unparseable dates were logged and reported in the quality check, not silently dropped."

### Q5: "How did you handle duplicates?"

> "Two types. **Exact duplicates** I handled with `dropDuplicates()` on all columns. **Business key duplicates** — same order_id but different timestamps — I used PySpark window functions: `ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY updated_at DESC)`, keeping only the latest record per business key. This is the same pattern as SQL deduplication."

### Q6: "How did this ETL enable downstream ML?"

> "The ML models — Mask R-CNN for document detection and a BERT classifier for legal documents — needed clean, structured training data. Before my pipeline, training data had nulls, duplicates, and type issues that degraded model accuracy. After, the ML team got validated, correctly typed data with metadata about completeness and distributions. This directly contributed to the Mask R-CNN achieving 92% mAP — because the training annotations were properly aligned with clean, deduplicated document records."

### Q7: "What was the hardest challenge?"

> "The hardest part was handling heterogeneous date formats from legacy systems. Some MSSQL extracts had dates as strings in mixed formats within the same column. I couldn't just pick one format because roughly 30% of records would fail. The solution was the multi-format parser with error capture. I also had to handle timezone-naive vs timezone-aware timestamps when joining data from different source systems — I standardized everything to UTC early in the pipeline."

### Q8: "How did you test your PySpark pipeline?"

> "I used pytest with PySpark's local mode (`spark.master=local[*]`). Each transformation function had unit tests with small DataFrames — I'd create test data with known edge cases (nulls, duplicates, mixed formats), run the transformation, and assert the output. For integration tests, I ran the full pipeline on a sample of production data and validated the quality report. This caught regressions when I modified transformation logic."

### Q9: "What performance optimizations did you apply?"

> "Five main ones: (1) **Predicate pushdown** — filtered data in the JDBC query subselect rather than reading the full table. (2) **Broadcast joins** — small lookup tables (< 10MB) broadcast to all executors to avoid shuffle. (3) **Column pruning** — selected only needed columns immediately after read. (4) **Coalesce** — reduced partition count before writing to prevent small file proliferation. (5) **JDBC batching** — set batchsize to 10,000 for bulk inserts, which was 5x faster than default."

### Q10: "What would you do differently today?"

> "Three things: (1) Use **Delta Lake** or **Iceberg** for the staging layer instead of raw Parquet — gives ACID transactions, time travel, and schema evolution. (2) Use **Apache Airflow** for orchestration instead of cron-based scheduling — better monitoring, retry logic, and dependency management. (3) Adopt **dbt** for SQL-based transformations inside the warehouse (ELT pattern) for simpler transformations, keeping PySpark only for complex custom logic."

### Q11: "ETL vs ELT — which would you use at FusionSpan?"

> "It depends on the transformation complexity. For FusionSpan's Salesforce data integration into Snowflake, I'd lean toward **ELT** — extract Salesforce data via APIs, load raw JSON into Snowflake's VARIANT columns, then transform using SQL or dbt inside Snowflake. Snowflake's compute handles SQL transformations efficiently, and ELT preserves raw data for evolving requirements. I'd use **ETL** (Python/PySpark) only if the data needs heavy custom preprocessing — parsing unstructured documents, complex NLP cleaning, or ML feature engineering — that can't be expressed in SQL."

### Q12: "How would you approach a new data source you've never worked with?"

> "Systematic exploration: (1) **Profile** the data — row count, column types, null rates, cardinality, distributions, sample records. (2) **Document** the schema — column definitions, business keys, relationships to other tables. (3) **Identify quality issues** — encoding problems, mixed formats, nulls, duplicates. (4) **Build a minimal pipeline** — extract a sample, apply basic transforms, validate output. (5) **Iterate** — add more transformation rules as I discover edge cases. (6) **Automate** — move from notebook to production pipeline with quality gates."

---

## 12. Key Talking Points

### 30-Second Pitch

> "At ATCS, I engineered PySpark ETL pipelines that processed millions of production records from CSV, JSON, and MSSQL sources into a clean analytics layer. I built a reusable data quality framework with configurable completeness, uniqueness, and volume checks — achieving zero quality escapes into production. The pipeline reduced processing from hours to minutes and was adopted across four client engagements. The clean data directly enabled downstream ML models to achieve 92% mAP on document detection."

### Three Things That Set This Apart

1. **Data quality as a first-class citizen** — not an afterthought. Quality gate between Transform and Load with structured reporting.
2. **Config-driven architecture** — YAML configs meant new data sources could be onboarded in days, not weeks. Reusable across clients.
3. **ML-aware ETL** — the pipeline was specifically designed to produce clean training data for downstream ML workflows, not just analytics.

### How It Maps to FusionSpan

| ATCS Experience | FusionSpan Application |
|----------------|----------------------|
| PySpark ETL from heterogeneous sources | Data integration from Salesforce, CRM, member systems |
| MSSQL as target warehouse | Snowflake as target warehouse |
| Data quality validation framework | Ensuring clean data for client analytics |
| Config-driven pipeline design | Reusable integration patterns across association clients |
| Enabling downstream ML/analytics | Powering data-driven insights for membership organizations |

---

*Prepared for Rahul Sharma — ATCS PySpark ETL Pipeline Project Description for FusionSpan Interview.*
