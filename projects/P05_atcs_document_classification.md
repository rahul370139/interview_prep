# P05 — Mask R-CNN Document Detection (92% mAP), PySpark ETL & Legal Document Classification | ATCS

> **Rahul Sharma** | Associate Data Scientist | Advanced Technology Consulting Service (ATCS), Jaipur, India
> **Duration:** November 2020 – September 2021
> **Primary resume bullets covered here:** Mask R-CNN / ResNet-101-FPN / Azure / 92% mAP / 63% review reduction · Docker + CI/CD
> **Companion document:** `P05b_atcs_pyspark_etl_pipeline.md` covers the ETL and data-quality bullets in full depth

---

## Table of Contents

1. [Project Overview (All Three Projects)](#1-project-overview)
2. [Project A: PySpark ETL Pipeline with Data Quality Validation](#2-project-a-pyspark-etl-pipeline-with-data-quality-validation)
3. [Project B: Mask R-CNN Document Detection (92% mAP)](#3-project-b-mask-r-cnn-document-detection-92-map)
4. [Project C: Legal Document Classification (NLP)](#4-project-c-legal-document-classification-nlp)
5. [Topics You Must Know (Comprehensive Study Guide)](#5-topics-you-must-know-comprehensive-study-guide)
6. [Interview Questions & Answers (35+)](#6-interview-questions--answers)
7. [Red Flags & How to Handle](#7-red-flags--how-to-handle)
8. [Key Takeaways & Talking Points](#8-key-takeaways--talking-points)

---

## 0. Resume Bullet ↔ Proof Map

Two of the four ATCS resume bullets are covered in depth here (the CV model and the containerisation/CI-CD work). The two ETL bullets have their own document, `P05b`. Every hard number below must be traceable to a section.

| # | Resume bullet | Hard metric | Where the proof lives | Say this (one sentence) |
|---|---------------|-------------|----------------------|-------------------------|
| 1 | "Deployed a fine-tuned **Mask R-CNN** document-detection model with **ResNet-101-FPN** on **Azure**, achieving **92% mAP** and reducing **manual document-region review by 63%**." | 92% mAP (mAP@0.5, 5 classes) | §3.2 Architecture, §3.4 Transfer Learning, §3.5 mAP Explained | *"I fine-tuned a Mask R-CNN with a ResNet-101-FPN backbone from COCO weights onto five document-region classes, hit 92% mAP at IoU 0.5, and served it from Azure — and because the confident detections stopped going to a human, manual region review dropped 63%."* |
| 1b | — same bullet, business half | 63% reduction in manual document-region review | §3.7 The 63% Review Reduction | *"63% is the share of document regions that no longer needed a human to look at them: we routed only the low-confidence and high-stakes regions to review, and measured the before-and-after review queue over the same document mix."* |
| 2 | "Containerized Python and ML workloads using **Docker** and automated testing and deployment through **CI/CD pipelines**, standardizing model releases." | Reproducible image per release; automated test + deploy | §3.8 Docker & CI/CD | *"Training and inference both ran from pinned Docker images built in CI, so a model release was a tagged image plus a versioned weights blob rather than someone's laptop environment."* |
| 3 | "Engineered distributed **PySpark ETL pipelines** processing **10M+ multi-source records**, reducing production batch runtime **from 4 hours to 18 minutes**." | 10M+ records · 4 h → 18 min | §2 (summary here) · **full depth in `P05b`** | *"That's the ETL workstream — I've got the full optimisation breakdown, but the short version is 10M+ records across CSV, JSON and JDBC sources, and a 4-hour batch cut to 18 minutes."* |
| 4 | "Built a **configuration-driven data quality framework** incorporating data validation, profiling and **referential integrity**, strengthening production data reliability across **4+ enterprise engagements**." | 4+ engagements | §2.3.2 (summary here) · **full depth in `P05b`** | *"The quality rules lived in YAML, not code, which is exactly why the same framework was reused on four-plus client engagements instead of being rewritten each time."* |

> **The one number to be careful with: 92% mAP is at IoU 0.5.** Under the stricter COCO protocol — mAP averaged over IoU 0.50 to 0.95 — the same model is around 0.70. Both numbers are real; they answer different questions. Lead with 92% mAP@0.5, and volunteer the COCO number *before* an interviewer asks, because that's the question a strong CV interviewer will ask second. See §3.5.

---

## 1. Project Overview

### 1.1 STAR Summary (Interview-Ready — Unified Narrative)

**Situation**
At ATCS (Advanced Technology Consulting Service), a technology consulting firm in Jaipur, India, clients across industries needed to digitize, classify, and process large volumes of heterogeneous documents — from scanned contracts and legal filings to structured transactional data. The existing workflows relied on manual sorting, ad-hoc scripting, and fragile batch jobs that couldn't scale. Data quality issues routinely propagated into downstream analytics, document classification was labor-intensive, and there was no automated detection pipeline for extracting structured regions from scanned documents.

**Task**
As an Associate Data Scientist, I was responsible for three interconnected workstreams:
1. **Building a production-grade PySpark ETL pipeline** that ingested raw data from multiple sources, applied rigorous data quality validation, and loaded clean data into MSSQL for downstream analytics.
2. **Training and deploying a Mask R-CNN model** (ResNet-101-FPN backbone) using transfer learning in TensorFlow for detecting and segmenting regions of interest (stamps, signatures, tables, headers) in scanned documents, achieving **92% mAP** and deploying to **Azure** for inference — which cut **manual document-region review by 63%**.
3. **Developing an NLP-based legal document classification system** using transfer learning and modular preprocessing pipelines to automate legal document tagging and labeling accuracy validation.

**Approach & Action**

| Phase | What I Did |
|-------|-----------|
| **ETL Pipeline Design** | Designed and implemented a PySpark-based ETL pipeline reading from flat files, CSVs, and database extracts; applied schema validation, null handling, type coercion, and deduplication before loading to MSSQL via JDBC. |
| **Data Quality Framework** | Built a reusable data quality validation layer with configurable rules — completeness checks, referential integrity, statistical profiling, and anomaly flagging — that ran as a pre-analytics gate. |
| **Mask R-CNN Training** | Fine-tuned a Mask R-CNN model (**ResNet-101 + FPN** backbone) pretrained on COCO using TensorFlow/Keras on a custom-annotated document dataset. Implemented data augmentation, anchor tuning, and learning rate scheduling. |
| **Model Deployment** | Packaged the trained Mask R-CNN model and deployed on **Azure** (weights in Blob Storage, inference container on Azure VM/Container Instances), building a pipeline that pulled the versioned model, ran detection on incoming documents, and returned bounding boxes + masks. |
| **Containerisation & CI/CD** | Containerised the Python training and inference workloads with **Docker** (pinned CUDA/TensorFlow base image, no "works on my machine"), and automated linting, unit tests, a smoke-inference test and image publish/deploy through a **CI/CD pipeline**, so a model release was a tagged image plus a versioned weights blob. |
| **Human-in-the-Loop Routing** | Designed the confidence-based routing that sends only low-confidence and high-stakes regions to a human reviewer — the mechanism behind the **63% reduction in manual document-region review**. |
| **Legal NLP Classification** | Built a classification model for legal document tagging using transfer learning (BERT-based) with NLP preprocessing (tokenization, stopword removal, legal entity extraction). Created modular pipelines for automated document ingestion. |
| **Labeling Validation** | Developed labeling accuracy validation logic — comparing model predictions against human annotations using confusion matrices, per-class F1, and inter-annotator agreement (Cohen's Kappa). |

**Result**
- PySpark ETL pipelines processing **10M+ multi-source records** into MSSQL with **zero data quality escapes**, batch runtime cut **from 4 hours to 18 minutes** (full detail in `P05b`)
- Mask R-CNN (ResNet-101-FPN) achieving **92% mAP** on document region detection, deployed on **Azure**, cutting **manual document-region review by 63%**
- **Docker + CI/CD** standardising model releases — reproducible images, automated tests, versioned weights
- Legal document classification model with high accuracy on multi-label tagging, reducing manual labeling effort by ~70%
- Config-driven data quality framework reused across **4+ enterprise engagements**

---

### 1.2 Combined Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ATCS DOCUMENT INTELLIGENCE PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗   │
│  ║                   PROJECT A: PySpark ETL Pipeline                        ║   │
│  ║                                                                         ║   │
│  ║  ┌──────────┐  ┌──────────┐  ┌──────────┐                              ║   │
│  ║  │ CSV/Flat │  │ Database │  │ External │                              ║   │
│  ║  │ Files    │  │ Extracts │  │ APIs     │                              ║   │
│  ║  └────┬─────┘  └────┬─────┘  └────┬─────┘                              ║   │
│  ║       │              │              │                                    ║   │
│  ║       ▼              ▼              ▼                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │          PySpark Ingestion Layer             │                        ║   │
│  ║  │  • spark.read.csv / .parquet / .jdbc         │                        ║   │
│  ║  │  • Schema inference + explicit casting       │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │       Data Quality Validation Layer          │                        ║   │
│  ║  │  • Schema checks (column names, types)       │                        ║   │
│  ║  │  • Null/missing value analysis               │                        ║   │
│  ║  │  • Type validation & coercion                │                        ║   │
│  ║  │  • Deduplication (exact + fuzzy)             │                        ║   │
│  ║  │  • Statistical profiling & anomaly flags     │                        ║   │
│  ║  │  • Referential integrity checks              │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │       Transformation & Enrichment            │                        ║   │
│  ║  │  • Column renaming, type casting             │                        ║   │
│  ║  │  • Derived features, aggregations            │                        ║   │
│  ║  │  • Partitioning for load optimization        │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │        MSSQL Load (JDBC)                     │                        ║   │
│  ║  │  • Batch writes via JDBC connector           │                        ║   │
│  ║  │  • Upsert logic (merge/overwrite)            │                        ║   │
│  ║  │  • Transaction management                    │                        ║   │
│  ║  └─────────────────────────────────────────────┘                        ║   │
│  ╚═══════════════════════════════════════════════════════════════════════════╝   │
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗   │
│  ║              PROJECT B: Mask R-CNN Document Detection                    ║   │
│  ║                                                                         ║   │
│  ║  ┌──────────────┐   ┌──────────────────┐                                ║   │
│  ║  │  Scanned     │   │  Annotation Tool │                                ║   │
│  ║  │  Documents   │   │  (VIA/LabelMe)   │                                ║   │
│  ║  │  (PDF/TIFF)  │   │  COCO-format     │                                ║   │
│  ║  └──────┬───────┘   └────────┬─────────┘                                ║   │
│  ║         │                     │                                          ║   │
│  ║         ▼                     ▼                                          ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │        Data Preprocessing                    │                        ║   │
│  ║  │  • Image resizing (1024×1024)                │                        ║   │
│  ║  │  • Augmentation (flip, rotate, brightness)   │                        ║   │
│  ║  │  • Train/Val/Test split (70/15/15)           │                        ║   │
│  ║  │  • Mask generation from polygon annotations  │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │    Mask R-CNN (Transfer Learning)            │                        ║   │
│  ║  │    Backbone: ResNet-101 + FPN                │                        ║   │
│  ║  │    Pretrained: COCO weights                  │                        ║   │
│  ║  │                                              │                        ║   │
│  ║  │  ┌─────────┐ ┌─────┐ ┌─────────┐ ┌──────┐  │                        ║   │
│  ║  │  │ResNet-  │→│ FPN │→│  RPN    │→│ ROI  │  │                        ║   │
│  ║  │  │101      │ │     │ │(Region  │ │Align │  │                        ║   │
│  ║  │  │Backbone │ │     │ │Proposal)│ │      │  │                        ║   │
│  ║  │  └─────────┘ └─────┘ └─────────┘ └──┬───┘  │                        ║   │
│  ║  │                                      │      │                        ║   │
│  ║  │                    ┌─────────────────┼──────────────┐                 ║   │
│  ║  │                    ▼                 ▼              ▼                 ║   │
│  ║  │              ┌──────────┐   ┌──────────┐   ┌──────────┐             ║   │
│  ║  │              │  Class   │   │  BBox    │   │  Mask    │             ║   │
│  ║  │              │  Head    │   │  Head    │   │  Head    │             ║   │
│  ║  │              │(Softmax) │   │(Regress) │   │(FCN)    │             ║   │
│  ║  │              └──────────┘   └──────────┘   └──────────┘             ║   │
│  ║  │                                                                      │   │
│  ║  │    Result: 92% mAP @ IoU 0.5  (≈0.70 mAP@[.5:.95])                  │   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │      Azure Deployment (Docker + CI/CD)       │                        ║   │
│  ║  │  • Weights + config in Azure Blob (versioned)│                        ║   │
│  ║  │  • Inference container pulls model on startup│                        ║   │
│  ║  │  • Batch + real-time inference support       │                        ║   │
│  ║  │  • CI: lint → unit → smoke-infer → publish   │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │   Confidence-Based Human-in-the-Loop Routing │                        ║   │
│  ║  │  • High confidence → auto-accept region      │                        ║   │
│  ║  │  • Low conf / high stakes → review queue     │                        ║   │
│  ║  │  → 63% fewer regions needing manual review   │                        ║   │
│  ║  └─────────────────────────────────────────────┘                        ║   │
│  ╚═══════════════════════════════════════════════════════════════════════════╝   │
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗   │
│  ║           PROJECT C: Legal Document Classification (NLP)                ║   │
│  ║                                                                         ║   │
│  ║  ┌──────────────┐   ┌──────────────────┐                                ║   │
│  ║  │  Legal Docs  │   │  Human Labels    │                                ║   │
│  ║  │  (Contracts, │   │  (Category tags, │                                ║   │
│  ║  │   Filings,   │   │   multi-label)   │                                ║   │
│  ║  │   Briefs)    │   │                  │                                ║   │
│  ║  └──────┬───────┘   └────────┬─────────┘                                ║   │
│  ║         │                     │                                          ║   │
│  ║         ▼                     ▼                                          ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │         NLP Preprocessing Pipeline           │                        ║   │
│  ║  │  • Text extraction (OCR if scanned)          │                        ║   │
│  ║  │  • Legal tokenization (domain-aware)         │                        ║   │
│  ║  │  • Stopword removal + legal stopword list    │                        ║   │
│  ║  │  • Named Entity Recognition (legal entities) │                        ║   │
│  ║  │  • Section segmentation                      │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │      Transfer Learning Classification        │                        ║   │
│  ║  │  • BERT / Legal-BERT fine-tuning             │                        ║   │
│  ║  │  • Multi-label classification head            │                        ║   │
│  ║  │  • Class-weighted loss for imbalance          │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │       Labeling Accuracy Validation           │                        ║   │
│  ║  │  • Confusion matrix per class                │                        ║   │
│  ║  │  • Per-class precision / recall / F1         │                        ║   │
│  ║  │  • Cohen's Kappa (inter-annotator)           │                        ║   │
│  ║  │  • Active learning feedback loop             │                        ║   │
│  ║  └─────────────────────────────────────────────┘                        ║   │
│  ╚═══════════════════════════════════════════════════════════════════════════╝   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 Tech Stack

| Category | Tools / Technologies |
|----------|---------------------|
| **Languages** | Python, SQL, PySpark |
| **ETL / Data** | Apache Spark (PySpark), MSSQL, JDBC, Pandas |
| **Computer Vision** | TensorFlow, Keras, Mask R-CNN (ResNet-101-FPN), OpenCV, `imgaug` |
| **NLP** | Hugging Face Transformers, BERT, spaCy, NLTK |
| **Cloud** | Azure Blob Storage, Azure VMs, Azure Container Registry / Container Instances |
| **Containerisation** | Docker (multi-stage builds, pinned CUDA/TF base images), docker-compose for local parity |
| **CI/CD** | Azure DevOps Pipelines (lint → unit tests → smoke inference → image build → push → deploy) |
| **Annotation** | VGG Image Annotator (VIA), LabelMe, COCO format |
| **Monitoring** | Custom data quality validators, logging frameworks, review-queue telemetry |
| **Version Control** | Git, Azure DevOps |

---

## 2. Project A: PySpark ETL Pipeline with Data Quality Validation

### 2.1 Problem Context

The client required a robust, scalable pipeline to:
- Ingest data from heterogeneous sources (flat files, CSVs, database extracts)
- Apply rigorous data quality validation before any analytics or reporting
- Load validated data into MSSQL for downstream BI tools and analytics teams
- Handle daily batch loads with growing data volumes (GBs per day)

**Why PySpark over plain Python/Pandas?**
- Data volumes exceeded single-node memory capacity
- PySpark's distributed computation enabled horizontal scaling
- Native support for reading multiple file formats
- Catalyst optimizer for efficient query execution plans
- Seamless integration with JDBC for MSSQL writes

---

### 2.2 ETL vs. ELT — Why ETL?

| Aspect | ETL (Extract-Transform-Load) | ELT (Extract-Load-Transform) |
|--------|------------------------------|------------------------------|
| **Transform location** | In the processing engine (Spark) | In the target database |
| **When to use** | When data quality must be validated before loading | When target DB has strong compute (Snowflake, BigQuery) |
| **Data quality** | Validated before it enters the target DB | Loaded raw, then cleaned in-place |
| **Our choice** | **ETL** — we needed to guarantee no dirty data entered MSSQL | — |

**Key decision rationale:** MSSQL was used primarily for reporting and analytics. Loading unvalidated data would have corrupted downstream dashboards and reports. ETL ensured that only clean, validated data reached MSSQL.

---

### 2.3 PySpark Pipeline — Deep Dive

#### 2.3.1 Extraction Phase

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

spark = SparkSession.builder \
    .appName("ATCS_ETL_Pipeline") \
    .config("spark.jars", "/path/to/mssql-jdbc-9.4.0.jre11.jar") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Define explicit schema (never rely solely on inference in production)
document_schema = StructType([
    StructField("doc_id", StringType(), nullable=False),
    StructField("doc_type", StringType(), nullable=True),
    StructField("client_name", StringType(), nullable=True),
    StructField("received_date", DateType(), nullable=True),
    StructField("page_count", IntegerType(), nullable=True),
    StructField("file_path", StringType(), nullable=False),
    StructField("status", StringType(), nullable=True),
    StructField("confidence_score", DoubleType(), nullable=True),
])

# Read with explicit schema
raw_df = spark.read \
    .option("header", "true") \
    .option("mode", "PERMISSIVE") \
    .option("columnNameOfCorruptRecord", "_corrupt_record") \
    .schema(document_schema) \
    .csv("/data/incoming/documents/*.csv")

print(f"Records read: {raw_df.count()}")
print(f"Corrupt records: {raw_df.filter(raw_df['_corrupt_record'].isNotNull()).count()}")
```

#### 2.3.2 Data Quality Validation Layer

This was the critical differentiator — a reusable, configurable validation framework.

```python
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from functools import reduce

class DataQualityValidator:
    """Reusable data quality validation framework for PySpark DataFrames."""
    
    def __init__(self, df, table_name):
        self.df = df
        self.table_name = table_name
        self.issues = []
        self.total_rows = df.count()
    
    def check_schema(self, expected_schema: StructType):
        """Validate that DataFrame schema matches expected schema."""
        actual_cols = set(self.df.columns)
        expected_cols = set([f.name for f in expected_schema.fields])
        
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        
        if missing:
            self.issues.append(f"SCHEMA_ERROR: Missing columns: {missing}")
        if extra:
            self.issues.append(f"SCHEMA_WARNING: Unexpected columns: {extra}")
        
        # Type validation
        for field in expected_schema.fields:
            if field.name in actual_cols:
                actual_type = dict(self.df.dtypes).get(field.name)
                expected_type = field.dataType.simpleString()
                if actual_type != expected_type:
                    self.issues.append(
                        f"TYPE_ERROR: {field.name} expected {expected_type}, got {actual_type}"
                    )
        return self
    
    def check_nulls(self, critical_columns, threshold=0.05):
        """Flag columns where null ratio exceeds threshold."""
        for col_name in critical_columns:
            null_count = self.df.filter(F.col(col_name).isNull()).count()
            null_ratio = null_count / self.total_rows if self.total_rows > 0 else 0
            
            if null_ratio > threshold:
                self.issues.append(
                    f"NULL_ERROR: {col_name} has {null_ratio:.2%} nulls "
                    f"(threshold: {threshold:.2%})"
                )
        return self
    
    def check_duplicates(self, key_columns):
        """Detect duplicate rows based on key columns."""
        dup_count = self.total_rows - self.df.dropDuplicates(key_columns).count()
        if dup_count > 0:
            self.issues.append(
                f"DUPLICATE_ERROR: {dup_count} duplicate rows on keys {key_columns}"
            )
        return self
    
    def check_value_ranges(self, column, min_val=None, max_val=None):
        """Validate numeric columns fall within expected ranges."""
        if min_val is not None:
            below = self.df.filter(F.col(column) < min_val).count()
            if below > 0:
                self.issues.append(
                    f"RANGE_ERROR: {column} has {below} values below {min_val}"
                )
        if max_val is not None:
            above = self.df.filter(F.col(column) > max_val).count()
            if above > 0:
                self.issues.append(
                    f"RANGE_ERROR: {column} has {above} values above {max_val}"
                )
        return self
    
    def check_referential_integrity(self, column, reference_df, ref_column):
        """Ensure foreign key values exist in reference table."""
        orphan_count = self.df.join(
            reference_df, self.df[column] == reference_df[ref_column], "left_anti"
        ).count()
        if orphan_count > 0:
            self.issues.append(
                f"REF_INTEGRITY_ERROR: {orphan_count} orphan records in {column}"
            )
        return self
    
    def validate(self, fail_on_error=True):
        """Execute all checks and return results."""
        errors = [i for i in self.issues if "ERROR" in i]
        warnings = [i for i in self.issues if "WARNING" in i]
        
        report = {
            "table": self.table_name,
            "total_rows": self.total_rows,
            "errors": len(errors),
            "warnings": len(warnings),
            "issues": self.issues,
            "passed": len(errors) == 0
        }
        
        if fail_on_error and not report["passed"]:
            raise DataQualityError(
                f"Validation failed for {self.table_name}: {errors}"
            )
        
        return report

class DataQualityError(Exception):
    pass
```

**Usage in the pipeline:**

```python
# Run validation before loading
validator = DataQualityValidator(raw_df, "document_metadata")
report = validator \
    .check_schema(document_schema) \
    .check_nulls(critical_columns=["doc_id", "file_path"], threshold=0.0) \
    .check_nulls(critical_columns=["doc_type", "client_name"], threshold=0.05) \
    .check_duplicates(key_columns=["doc_id"]) \
    .check_value_ranges("page_count", min_val=1, max_val=10000) \
    .check_value_ranges("confidence_score", min_val=0.0, max_val=1.0) \
    .validate(fail_on_error=True)

print(f"Validation passed: {report['passed']}")
```

#### 2.3.3 Transformation Phase

```python
# Clean and transform validated data
transformed_df = raw_df \
    .withColumn("doc_type", F.upper(F.trim(F.col("doc_type")))) \
    .withColumn("client_name", F.initcap(F.trim(F.col("client_name")))) \
    .withColumn("received_date", F.to_date(F.col("received_date"), "yyyy-MM-dd")) \
    .withColumn("ingestion_timestamp", F.current_timestamp()) \
    .withColumn("status", F.coalesce(F.col("status"), F.lit("PENDING"))) \
    .withColumn("confidence_score",
                F.when(F.col("confidence_score").isNull(), F.lit(0.0))
                 .otherwise(F.col("confidence_score"))) \
    .dropDuplicates(["doc_id"])

# Add derived columns
transformed_df = transformed_df \
    .withColumn("is_high_confidence", F.col("confidence_score") >= 0.85) \
    .withColumn("processing_priority",
                F.when(F.col("doc_type") == "LEGAL", 1)
                 .when(F.col("doc_type") == "FINANCIAL", 2)
                 .otherwise(3))
```

#### 2.3.4 Loading to MSSQL via JDBC

```python
# MSSQL JDBC connection properties
jdbc_url = "jdbc:sqlserver://atcs-prod-db.database.windows.net:1433;databaseName=DocumentDB"
connection_properties = {
    "user": "etl_service_account",
    "password": "${DB_PASSWORD}",  # injected from environment / secrets manager
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
    "batchsize": "10000",
    "isolationLevel": "READ_COMMITTED"
}

# Write to MSSQL — overwrite for full refresh, append for incremental
transformed_df.write \
    .mode("append") \
    .jdbc(url=jdbc_url, table="dbo.document_metadata", properties=connection_properties)

# Verify write
verify_count = spark.read \
    .jdbc(url=jdbc_url, table="dbo.document_metadata", properties=connection_properties) \
    .count()

print(f"Records in MSSQL after load: {verify_count}")
```

---

### 2.4 PySpark Core Concepts (Must-Know)

#### RDDs vs. DataFrames vs. Datasets

| Feature | RDD | DataFrame | Dataset (Scala/Java) |
|---------|-----|-----------|----------------------|
| **Abstraction** | Low-level, distributed collection | Distributed table with named columns | Typed DataFrame |
| **Optimization** | No Catalyst optimization | Catalyst + Tungsten optimized | Catalyst + Tungsten optimized |
| **Schema** | No schema | Has schema (StructType) | Strongly typed schema |
| **API** | Functional (map, filter, reduce) | SQL-like (select, filter, groupBy) | Combines both |
| **Use case** | Unstructured data, custom logic | Structured/semi-structured data | Type-safe operations |
| **Performance** | Slowest (no optimization) | Fastest (optimizer + code gen) | Fast (with type safety) |

**Interview talking point:** "We used DataFrames exclusively because our data was structured (CSVs, database tables), and DataFrames benefit from Catalyst optimization which gave us 2-5x better performance over equivalent RDD operations."

#### Transformations vs. Actions

```
TRANSFORMATIONS (Lazy — build execution plan)          ACTIONS (Eager — trigger computation)
─────────────────────────────────────────────          ────────────────────────────────────
select(), filter(), groupBy(), join()                  count(), collect(), show(), write()
withColumn(), drop(), distinct()                       first(), take(), foreach()
union(), repartition(), coalesce()                     toPandas(), saveAsTable()

Key insight: Spark builds a DAG of transformations. Nothing executes until an action triggers it.
This enables Catalyst to optimize the entire plan.
```

#### Narrow vs. Wide Transformations

```
NARROW (no shuffle — fast)         WIDE (shuffle required — expensive)
────────────────────────────       ──────────────────────────────────
map(), filter(), select()          groupBy(), join(), distinct()
withColumn(), union()              repartition(), orderBy()

Why it matters: Wide transformations cause data shuffle across the network,
which is the #1 performance bottleneck in Spark.
```

#### Spark Execution Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SPARK EXECUTION FLOW                             │
│                                                                      │
│  User Code (PySpark)                                                 │
│       │                                                              │
│       ▼                                                              │
│  Logical Plan                                                        │
│       │                                                              │
│       ▼                                                              │
│  Catalyst Optimizer ──→ Optimized Logical Plan                       │
│       │                  • Predicate pushdown                        │
│       │                  • Column pruning                            │
│       │                  • Constant folding                          │
│       │                  • Join reordering                           │
│       ▼                                                              │
│  Physical Plan (chosen by cost-based optimizer)                      │
│       │                                                              │
│       ▼                                                              │
│  Tungsten Engine (Code Generation)                                   │
│       │    • Whole-stage code generation                             │
│       │    • Off-heap memory management                              │
│       ▼                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                              │
│  │  Stage 1 │  │  Stage 2 │  │  Stage 3 │  (separated by shuffles)   │
│  │ Tasks=N  │→ │ Tasks=M  │→ │ Tasks=K  │                            │
│  └─────────┘  └─────────┘  └─────────┘                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 2.5 MSSQL Integration — Key Patterns

**JDBC Batch Size Tuning:**
```python
# Default batchsize is 1000 — too small for large loads
# We tuned to 10,000 based on benchmarking
#   1,000 → 45 min for 2M rows
#  10,000 → 12 min for 2M rows
#  50,000 → 10 min (diminishing returns, memory pressure)
```

**Handling Upsert (Merge) Logic:**
```python
# PySpark doesn't natively support UPSERT
# Strategy: Write to staging table, then MERGE in MSSQL

# Step 1: Write to staging
transformed_df.write \
    .mode("overwrite") \
    .jdbc(url=jdbc_url, table="dbo.staging_document_metadata", properties=connection_properties)

# Step 2: Execute MERGE via pyodbc
import pyodbc
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()
cursor.execute("""
    MERGE dbo.document_metadata AS target
    USING dbo.staging_document_metadata AS source
    ON target.doc_id = source.doc_id
    WHEN MATCHED THEN
        UPDATE SET target.status = source.status,
                   target.confidence_score = source.confidence_score,
                   target.ingestion_timestamp = source.ingestion_timestamp
    WHEN NOT MATCHED THEN
        INSERT (doc_id, doc_type, client_name, received_date, 
                page_count, file_path, status, confidence_score, ingestion_timestamp)
        VALUES (source.doc_id, source.doc_type, source.client_name, source.received_date,
                source.page_count, source.file_path, source.status, 
                source.confidence_score, source.ingestion_timestamp);
""")
conn.commit()
```

---

## 3. Project B: Mask R-CNN Document Detection (92% mAP)

### 3.1 Problem Context

Scanned documents contained regions of interest — stamps, signatures, tables, headers, handwritten notes — that needed to be detected and segmented for downstream processing (extraction, classification, archival). Manual identification was slow and error-prone.

**Why Mask R-CNN (not YOLO, Faster R-CNN, etc.)?**

| Model | Detection | Segmentation | Speed | Our Need |
|-------|-----------|-------------|-------|----------|
| **YOLO** | Bounding boxes | No masks | Very fast | Not enough — we needed pixel-level masks |
| **Faster R-CNN** | Bounding boxes | No masks | Moderate | No segmentation capability |
| **Mask R-CNN** | Bounding boxes + classes | **Pixel-level masks** | Moderate | **Chosen** — we needed both detection and segmentation |
| **U-Net** | No detection | Semantic segmentation | Fast | No instance-level distinction |

**Key requirement:** We needed to distinguish between overlapping elements (e.g., a signature overlapping a stamp). Only instance segmentation (Mask R-CNN) provides per-instance masks.

---

### 3.2 Mask R-CNN Architecture — Complete Breakdown

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       MASK R-CNN ARCHITECTURE                                 │
│                                                                               │
│  Input Image (1024 × 1024 × 3)                                               │
│       │                                                                       │
│       ▼                                                                       │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 1: BACKBONE — ResNet-101                                         ║ │
│  ║                                                                         ║ │
│  ║  Conv1 ──→ Pool ──→ Res2 (C2) ──→ Res3 (C3) ──→ Res4 (C4) ──→ Res5 (C5) ║
│  ║   │                  256ch          512ch          1024ch         2048ch ║ │
│  ║   │                    │              │               │              │   ║ │
│  ║   │                    │              │               │              │   ║ │
│  ╚═══│════════════════════│══════════════│═══════════════│══════════════│═══╝ │
│       │                    │              │               │              │     │
│       │                    ▼              ▼               ▼              ▼     │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 2: FEATURE PYRAMID NETWORK (FPN)                                 ║ │
│  ║                                                                         ║ │
│  ║  Top-down pathway + lateral connections:                                ║ │
│  ║                                                                         ║ │
│  ║  C5 ──→ P5 (1/32 scale) ──→ upsample ──→ + ──→ P4 (1/16 scale)       ║ │
│  ║                                            │                            ║ │
│  ║  C4 ────────────────────────→ 1×1 conv ────┘    upsample               ║ │
│  ║                                                   │                     ║ │
│  ║  C3 ────────────────────────→ 1×1 conv ──→ + ────→ P3 (1/8 scale)     ║ │
│  ║                                                                         ║ │
│  ║  C2 ────────────────────────→ 1×1 conv ──→ + ────→ P2 (1/4 scale)     ║ │
│  ║                                                                         ║ │
│  ║  All Pn have 256 channels                                               ║ │
│  ║  Purpose: Detect objects at multiple scales                             ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│       │                                                                       │
│       ▼                                                                       │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 3: REGION PROPOSAL NETWORK (RPN)                                 ║ │
│  ║                                                                         ║ │
│  ║  For each location on each feature map (P2–P5):                         ║ │
│  ║    • Generate k anchor boxes (k=15: 5 scales × 3 aspect ratios)        ║ │
│  ║    • Two outputs per anchor:                                            ║ │
│  ║      1. Objectness score (object vs. background)   [2k scores]         ║ │
│  ║      2. Bounding box refinement (dx, dy, dw, dh)   [4k offsets]       ║ │
│  ║                                                                         ║ │
│  ║  Pipeline:                                                              ║ │
│  ║    Anchors → IoU with GT → Binary classification + BBox regression      ║ │
│  ║           → NMS (Non-Maximum Suppression) → Top-N proposals (~2000)    ║ │
│  ║                                                                         ║ │
│  ║  Anchor scales: [32, 64, 128, 256, 512]                                ║ │
│  ║  Anchor ratios: [0.5, 1, 2]                                            ║ │
│  ╚══════════════════════════════════════════════╤════════════════════════════╝ │
│                                                  │                             │
│                                                  ▼                             │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 4: ROI ALIGN                                                     ║ │
│  ║                                                                         ║ │
│  ║  Problem: ROI Pooling (Faster R-CNN) uses quantization → misalignment  ║ │
│  ║  Solution: ROI Align uses bilinear interpolation → no quantization     ║ │
│  ║                                                                         ║ │
│  ║  For each proposal:                                                     ║ │
│  ║    1. Map ROI to feature map coordinates (no rounding!)                 ║ │
│  ║    2. Divide into grid (e.g., 7×7 for class/bbox, 14×14 for mask)     ║ │
│  ║    3. Sample 4 points per bin using bilinear interpolation              ║ │
│  ║    4. Max-pool or avg-pool within each bin                              ║ │
│  ║                                                                         ║ │
│  ║  Why this matters: Mask prediction requires pixel-level alignment.      ║ │
│  ║  Even 1-pixel misalignment degrades mask quality significantly.         ║ │
│  ╚══════════════════════════════════════════════╤════════════════════════════╝ │
│                                                  │                             │
│                        ┌─────────────────────────┼──────────────────┐          │
│                        ▼                         ▼                  ▼          │
│  ╔══════════════╗  ╔══════════════╗  ╔═══════════════════════════╗             │
│  ║  CLASS HEAD  ║  ║  BBOX HEAD   ║  ║      MASK HEAD            ║             │
│  ║              ║  ║              ║  ║                           ║             │
│  ║  FC layers   ║  ║  FC layers   ║  ║  4 conv layers (3×3)     ║             │
│  ║  → Softmax   ║  ║  → Regression║  ║  → deconv (2×2, s=2)    ║             │
│  ║              ║  ║  (dx,dy,dw,dh)  ║  → 1×1 conv per class   ║             │
│  ║  Output:     ║  ║  Output:     ║  ║  → Sigmoid              ║             │
│  ║  N+1 classes ║  ║  4 coords    ║  ║                           ║             │
│  ║  per ROI     ║  ║  per ROI     ║  ║  Output: 28×28 binary    ║             │
│  ║              ║  ║              ║  ║  mask per class per ROI  ║             │
│  ╚══════════════╝  ╚══════════════╝  ╚═══════════════════════════╝             │
│                                                                               │
│  LOSS FUNCTION:                                                               │
│  L = L_cls + L_bbox + L_mask                                                  │
│    = CrossEntropy + SmoothL1 + BinaryCrossEntropy (per-class, only on GT class)│
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.3 Key Architectural Components Explained

#### 3.3.1 ResNet-101 Backbone

ResNet-101 consists of 101 layers organized into 4 residual stages (conv2_x through conv5_x). Each stage uses **residual (skip) connections** that solve the vanishing gradient problem:

```
Residual Block:
x ──→ [Conv → BN → ReLU → Conv → BN] ──→ (+) ──→ ReLU ──→ output
 │                                          ↑
 └──────────── identity shortcut ───────────┘

output = F(x) + x    (where F(x) is the learned residual)
```

**Why ResNet-101 (not ResNet-50)?** Deeper backbone extracts richer features for complex document layouts. ResNet-101 gave ~2% mAP improvement over ResNet-50 in our experiments.

#### 3.3.2 Feature Pyramid Network (FPN)

FPN addresses the **multi-scale detection problem**: small objects (text) need high-resolution features, while large objects (tables) need high-level semantic features.

```
Without FPN:                           With FPN:
Only detect at one scale               Detect at all scales simultaneously

C5 (low-res, high-semantic) → detect   P5 → large objects (tables, full pages)
                                        P4 → medium objects (paragraphs, figures)
                                        P3 → small-medium objects (signatures, stamps)
                                        P2 → small objects (text lines, icons)
```

**FPN assignment rule:** An ROI of width w and height h is assigned to pyramid level:

```
k = floor(k₀ + log₂(√(wh) / 224))

where k₀ = 4 (P4 is the canonical level for 224×224 ROIs)
```

#### 3.3.3 ROI Align vs. ROI Pooling

```
ROI Pooling (Faster R-CNN):                ROI Align (Mask R-CNN):
┌─────────────────────┐                    ┌─────────────────────┐
│ ROI on feature map   │                    │ ROI on feature map   │
│ x=3.75 → quantize→4 │                    │ x=3.75 → keep 3.75  │
│ y=2.3  → quantize→2 │                    │ y=2.3  → keep 2.3   │
│                      │                    │                      │
│ Division: quantize   │                    │ Division: exact      │
│ each bin boundary    │                    │ Bilinear interpolate │
│                      │                    │ at sample points     │
│ → Misaligned by up   │                    │ → Pixel-perfect      │
│   to 0.5 pixels      │                    │   alignment          │
└─────────────────────┘                    └─────────────────────┘

Impact: ROI Align improved mask AP by ~3% (from the original Mask R-CNN paper)
```

#### 3.3.4 Mask Head — Decoupled Prediction

**Critical design choice:** The mask head predicts a binary mask **per class**, independently of classification. This decouples mask prediction from class prediction.

```
Traditional approach: Predict one multi-class mask → classes compete for pixels
Mask R-CNN approach: Predict K binary masks (one per class) → use class head to select

This avoids inter-class competition and improves mask quality.
Loss is computed only on the mask corresponding to the ground-truth class.
```

#### 3.3.5 The Multi-Task Loss, Formally

Mask R-CNN is trained end-to-end on a sum of three losses, computed per sampled RoI:

$$L = L_{cls} + L_{box} + L_{mask}$$

**Classification loss** — multinomial cross-entropy over $K+1$ classes (the $+1$ is background):

$$L_{cls}(p, u) = -\log p_u$$

where $p$ is the softmax output over classes and $u$ is the ground-truth class index.

**Box regression loss** — smooth $L_1$ over the four parameterised offsets, applied **only when $u \ge 1$** (i.e. the RoI is not background; there is no box to regress for background):

$$L_{box}(t^u, v) = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L_1}\!\left(t_i^u - v_i\right)$$

$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

Smooth $L_1$ is used rather than plain $L_2$ because it is **less sensitive to outliers** — a badly-placed proposal produces a gradient bounded at 1 instead of growing linearly, which prevents a handful of terrible proposals from dominating the update. It's differentiable at the origin, unlike plain $L_1$.

The offsets themselves are the standard scale-invariant parameterisation, which matters because it makes the regression target independent of object size:

$$t_x = \frac{x - x_a}{w_a},\quad t_y = \frac{y - y_a}{h_a},\quad t_w = \log\frac{w}{w_a},\quad t_h = \log\frac{h}{h_a}$$

**Mask loss** — average binary cross-entropy over the $m \times m$ mask (28×28 in our config), computed **only on the mask channel corresponding to the ground-truth class $u$**:

$$L_{mask} = -\frac{1}{m^2}\sum_{i,j}\Big[ y_{ij}\log \hat{y}^u_{ij} + (1 - y_{ij})\log(1 - \hat{y}^u_{ij}) \Big]$$

**This per-class, sigmoid-based formulation is the paper's key insight, and it's the detail worth being able to explain.** The obvious alternative is a single mask with a softmax over classes at each pixel — but that makes classes *compete* for pixels, so the mask branch is forced to implicitly solve the classification problem again. By emitting $K$ independent binary masks with sigmoid activation and supervising only the ground-truth channel, mask prediction is **decoupled** from classification: the class head decides *what*, the mask head decides *which pixels*, and neither has to do the other's job. The paper measures this as a substantial mask AP gain over the softmax formulation.

**The RPN has its own two-term loss**, trained jointly:

$$L_{RPN} = \frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}}\sum_i p_i^* \, L_{box}(t_i, t_i^*)$$

The $p_i^*$ multiplier on the regression term is the same idea as before — only positive anchors contribute a box loss. Anchors are labelled positive at IoU ≥ 0.7 with any ground truth (or as the highest-IoU anchor for a ground truth that would otherwise be unmatched), negative below 0.3, and **ignored in between** — those ambiguous anchors contribute no gradient at all, which is a cleaner solution than forcing a hard label onto a genuinely borderline case.

---

### 3.4 Transfer Learning Strategy

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING PIPELINE                                │
│                                                                              │
│  Phase 1: Start with COCO-pretrained weights                                 │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │  COCO Dataset: 330K images, 80 object classes            │                │
│  │  Pretrained Mask R-CNN: Strong general feature extraction │                │
│  │  Backbone (ResNet-101) + FPN + RPN already trained       │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                              │
│  Phase 2: Freeze backbone, train heads (10-20 epochs)                        │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │  Frozen: ResNet-101 backbone + FPN                       │                │
│  │  Trainable: RPN + Classification head + BBox head + Mask │                │
│  │  Learning rate: 0.001                                    │                │
│  │  Purpose: Adapt detection heads to document domain       │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                              │
│  Phase 3: Unfreeze all, fine-tune end-to-end (30-50 epochs)                  │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │  All layers trainable                                    │                │
│  │  Learning rate: 0.0001 (10× lower to avoid catastrophic  │                │
│  │                         forgetting)                       │                │
│  │  LR schedule: Step decay at epochs 30, 40                │                │
│  │  Data augmentation: horizontal flip, rotation (±15°),    │                │
│  │                     brightness/contrast, random crop      │                │
│  │  Result: 92% mAP @ IoU 0.5  (≈0.70 mAP@[.5:.95])         │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Why Layer Freezing in This Order — the Reasoning, Not Just the Recipe

The two-phase schedule isn't arbitrary. When you attach randomly-initialised heads to a pretrained backbone and immediately train end-to-end, the large gradients from the untrained heads flow backwards and **destroy** the pretrained features before they can be useful. That's catastrophic forgetting, and on a few-hundred-image dataset it's unrecoverable. Freezing the backbone in Phase 1 lets the heads reach a sensible state first; only then is it safe to let gradients through.

The 10× lower learning rate in Phase 2 is the same logic applied continuously: you want the backbone to *adapt*, not to be *rewritten*.

| Layer group | Phase 1 (epochs 1–20) | Phase 2 (epochs 21–50) | Why |
|---|---|---|---|
| ResNet-101 stages C1–C2 | Frozen | **Still frozen** | Edges, corners, stroke texture — universal. Documents need exactly the same low-level features as COCO. Unfreezing these only adds variance. |
| ResNet-101 stages C3–C5 | Frozen | Unfrozen @ LR/10 | Mid/high-level semantics. COCO's "object-ness" priors are useful but need adaptation — a table is not any COCO class. |
| FPN lateral/output convs | Frozen | Unfrozen @ LR/10 | Needs to re-weight which pyramid levels matter, since document objects skew smaller and flatter than COCO objects. |
| RPN | **Trainable** | Trainable | Document aspect ratios (wide, flat headers and tables) are unlike COCO's roughly-square object prior. The RPN needs retraining from the start. |
| Class / bbox / mask heads | **Trainable** (re-initialised) | Trainable | 5 classes, not 80 — these layers are new by definition. |

Note the detail in the loading code below: `exclude=[...]` drops the COCO classification and mask output layers, because their shapes are tied to 80 classes. Forgetting that `exclude` is a classic first-run error — the weights simply won't load.

**Also worth stating:** BatchNorm layers stay frozen throughout. With `IMAGES_PER_GPU = 2`, the effective batch size is 2, and BatchNorm statistics estimated from batches of 2 are pure noise. Every Mask R-CNN implementation defaults to frozen BN for exactly this reason, and interviewers who've trained detectors will check whether you know why.

#### Augmentation Choices Specific to Documents

Generic ImageNet-style augmentation is actively wrong for scanned documents. What you keep and what you throw away is a good signal of whether you've actually trained a detector on this domain.

| Augmentation | Used? | Reasoning |
|---|---|---|
| **Small rotation (±5–15°)** | Yes, heavily | Directly simulates skew from a physical scanner or a photographed page. The single highest-value document augmentation. |
| **Brightness / contrast / gamma jitter** | Yes, heavily | Scan quality varies enormously across clients, devices and paper age. |
| **Gaussian noise + JPEG compression artefacts** | Yes | Fax-quality and re-scanned-photocopy inputs are real; training on clean scans only produces a model that collapses on them. |
| **Slight blur / sharpen** | Yes | Focus variation from phone-camera capture. |
| **Random crop / scale jitter** | Yes | Page margins and DPI differ; scale jitter also exercises all the FPN levels. |
| **Horizontal flip** | Limited | Mirrors text into nonsense. Harmless for *stamp* and *signature* (blob-like), damaging for *header* and *table* where left-alignment and reading order are the cue. Applied per-class rather than globally. |
| **Vertical flip / 180° rotation** | No | An upside-down page is a different problem — solved by an orientation-detection pre-step, not by teaching the detector that inverted headers are normal. |
| **Colour channel shuffle / strong hue jitter** | No | Most inputs are greyscale or near-greyscale; hue augmentation invents variation that never occurs in production. |
| **Cutout / random erasing** | Cautiously | Can erase a whole small object — a stamp is small — so it was applied with a size cap. |
| **Synthetic stamp/signature compositing** | Yes | The most effective single intervention for the rare classes: paste real cropped stamps and signatures onto real blank document backgrounds at plausible positions and scales, carrying the mask across. |

#### Class Imbalance in the Detection Setting

Class imbalance in object detection is two separate problems, and conflating them is a common interview slip.

**(1) Foreground/background imbalance — intrinsic to every detector.** The RPN generates thousands of anchors per image and almost all are background. Mask R-CNN handles this architecturally, and you should be able to name how: the RPN samples a **fixed minibatch of 256 anchors per image at a target 1:1 positive:negative ratio** (padding with negatives when positives are scarce), and the detection head samples **512 RoIs at 1:3 positive:negative**. That's why a two-stage detector doesn't need focal loss the way a dense one-stage detector like RetinaNet does — the sampling *is* the rebalancing.

**(2) Inter-class imbalance — specific to this dataset.** The five classes were very unevenly represented:

| Class | Approx. share of annotated instances | Difficulty |
|---|---|---|
| Header | ~31% | Easy — consistent position and shape |
| Table | ~27% | Easy — strong ruled-line structure |
| Signature | ~21% | Medium — variable, but a consistent stroke texture |
| Stamp | ~14% | Medium — small, but distinctive colour and shape |
| Handwriting | ~7% | **Hard** — unbounded visual variability, fuzzy boundaries |

What I did about it, ordered by how much it actually helped:

1. **Targeted annotation** — spent the marginal labelling budget almost entirely on handwriting and stamps rather than uniformly. More real examples of the rare class beats every algorithmic trick.
2. **Synthetic compositing** for stamps and signatures, which have clean extractable masks.
3. **Repeat-factor sampling of images containing rare classes**, rather than reweighting the classification loss. In detection, image-level oversampling is better behaved than class-weighting the head, because it also gives the RPN and the mask head more rare-class supervision.
4. **Reported per-class AP, never only the mean.** The mean hides the weak class entirely — which is exactly what the per-class table in §3.5 exists to prevent.

> **Say this:** *"Handwriting was 7% of instances and by far the hardest class, so it dominated my error budget. I spent the extra annotation there, oversampled the images containing it, and always reported per-class AP so the mean couldn't hide it."*

**Training Code (simplified):**

```python
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib

class DocumentConfig(Config):
    NAME = "document_detection"
    NUM_CLASSES = 1 + 5  # background + (stamp, signature, table, header, handwriting)
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    
    # Backbone
    BACKBONE = "resnet101"
    
    # RPN anchors - tuned for document elements
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    # Training
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    # Image sizing
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    
    # Detection
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    
    # Training proposals
    TRAIN_ROIS_PER_IMAGE = 200
    
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20

config = DocumentConfig()

# Load pretrained COCO weights
model = modellib.MaskRCNN(mode="training", config=config, model_dir="./logs")
model.load_weights("mask_rcnn_coco.h5", by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                            "mrcnn_bbox", "mrcnn_mask"])

# Phase 2: Train heads only
model.train(train_dataset, val_dataset,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            layers="heads")

# Phase 3: Fine-tune all layers
model.train(train_dataset, val_dataset,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=50,
            layers="all")
```

---

### 3.5 Mean Average Precision (mAP) — 92% Explained Properly

#### What is mAP?

mAP is the primary metric for object detection. It measures both **localization accuracy** (are bounding boxes correct?) and **classification accuracy** (are labels correct?).

**Step-by-step mAP computation:**

```
Step 1: For each detection, compute IoU with ground truth

         Area of Overlap
  IoU = ─────────────────────
         Area of Union

         |A ∩ B|
       = ─────────
         |A ∪ B|

Step 2: At a given IoU threshold (e.g., 0.5), classify detections:
  - True Positive (TP): IoU ≥ threshold AND correct class
  - False Positive (FP): IoU < threshold OR wrong class
  - False Negative (FN): Ground truth with no matching detection

Step 3: Compute Precision and Recall at each detection (sorted by confidence):
  
           TP                    TP
  P = ──────────       R = ──────────
       TP + FP              TP + FN

Step 4: Plot Precision-Recall curve, compute AP (area under PR curve):
  
  AP = ∫₀¹ p(r) dr    (using 11-point or all-point interpolation)

Step 5: Average AP across all classes:

         1   C
  mAP = ─── Σ  APᵢ
         C  i=1

  where C = number of classes
```

#### The Formal Definitions

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

$$P = \frac{TP}{TP + FP} \qquad R = \frac{TP}{TP + FN}$$

$$\text{AP}_c = \int_0^1 p_{\text{interp}}(r)\,dr \qquad\text{where}\qquad p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$$

$$\text{mAP} = \frac{1}{C}\sum_{c=1}^{C} \text{AP}_c \qquad\qquad \text{mAP@}[.5{:}.95] = \frac{1}{10}\sum_{t \in \{0.50, 0.55, \ldots, 0.95\}} \text{mAP@}t$$

#### Precision-Recall Curve Interpolation — the Part People Get Wrong

The raw precision-recall curve is **saw-toothed**: as you walk down the confidence-sorted detection list, each true positive bumps precision up and each false positive drags it down. Integrating that jagged curve directly would make AP hypersensitive to the exact ordering of near-tied detections.

So AP is always computed on an **interpolated** curve — at each recall level you take the *maximum precision achieved at that recall or any higher recall*, which produces a monotonically non-increasing step function. Three conventions exist and they give different numbers on the same predictions:

| Convention | How | Used by |
|---|---|---|
| **11-point interpolation** | Average interpolated precision at recall = 0.0, 0.1, …, 1.0 | PASCAL VOC 2007. Coarse; largely historical. |
| **All-point (area-under-step) interpolation** | Sum precision × recall-width over every recall level where recall changes | PASCAL VOC 2010+, and the standard everywhere now |
| **101-point interpolation** | Average interpolated precision at recall = 0.00, 0.01, …, 1.00 | COCO. Effectively all-point with a fixed sampling grid. |

Worked micro-example — 5 ground-truth objects, 6 detections sorted by descending confidence:

| Rank | Conf | TP/FP | Cum TP | Cum FP | Precision | Recall | Interpolated precision |
|---|---|---|---|---|---|---|---|
| 1 | 0.98 | TP | 1 | 0 | 1.000 | 0.20 | 1.000 |
| 2 | 0.95 | TP | 2 | 0 | 1.000 | 0.40 | 1.000 |
| 3 | 0.91 | FP | 2 | 1 | 0.667 | 0.40 | 0.750 |
| 4 | 0.88 | TP | 3 | 1 | 0.750 | 0.60 | 0.750 |
| 5 | 0.72 | FP | 3 | 2 | 0.600 | 0.60 | 0.714 |
| 6 | 0.65 | TP | 4 | 2 | 0.667 | 0.80 | 0.667 |

All-point AP = $(0.20 - 0.00)(1.000) + (0.40 - 0.20)(1.000) + (0.60 - 0.40)(0.750) + (0.80 - 0.60)(0.667) = 0.683$

Note the fifth ground-truth object was never detected — recall tops out at 0.80, and the missing 0.20 of recall contributes zero. **Missed objects silently cap your AP**, which is why recall failures hurt mAP more than most people expect.

Three matching rules that materially change the number and are worth being able to recite:
1. Detections are matched **greedily in descending confidence order**. The highest-confidence detection claims the best-overlapping unmatched ground truth.
2. A ground truth can only be matched **once**. A second detection on the same object is a false positive, not a duplicate true positive — this is what makes NMS quality part of your mAP.
3. **Every detection you emit counts.** Lowering your confidence threshold to catch more objects also injects false positives at the tail of the ranking. It usually *helps* AP slightly (extra recall at low precision adds a little area) but it destroys the *operating-point* precision your users actually experience — which is why AP and deployed precision are different conversations.

#### mAP@0.5 vs mAP@[.5:.95] — and Which One 92% Refers To

**Our headline 92% is mAP@0.5** — mean average precision at a single IoU threshold of 0.5, the PASCAL VOC convention (and what COCO calls AP<sub>50</sub>). Under the stricter COCO primary metric, averaging over IoU 0.50 to 0.95, the same model scores roughly 0.70.

Both numbers describe the same predictions. They differ because they ask different questions:

| | mAP@0.5 | mAP@[.5:.95] |
|---|---|---|
| **Question it answers** | "Did you find the right thing, roughly in the right place?" | "Did you find the right thing *and* trace its boundary tightly?" |
| **Sensitive to** | Detection and classification | Detection, classification, **and** localisation precision |
| **Typical value on the same model** | Higher | Substantially lower — a 20-point gap is normal |
| **Right choice when** | Downstream consumes the region as a crop with margin | Downstream needs pixel-accurate boundaries |

**Per-class breakdown of the 92%:**

| Class | AP@0.5 | AP@[.5:.95] | Comment |
|-------|--------|-------------|---------|
| Header | 0.96 | 0.76 | Consistent position and rectangular shape — easiest class |
| Table | 0.95 | 0.75 | Ruled lines give crisp, unambiguous boundaries |
| Stamp | 0.94 | 0.72 | Distinctive colour and shape; small size costs some localisation precision |
| Signature | 0.92 | 0.68 | Wispy strokes make the "true" boundary genuinely ambiguous |
| Handwriting | 0.83 | 0.59 | Unbounded variability; annotators themselves disagree on extent |
| **Mean** | **0.92** | **0.70** | |

#### Why 92% mAP@0.5 Is Meaningful *Here* — and Why It Isn't Comparable to COCO

The reflexive objection is "state of the art on COCO is about 50% mAP, so 92% must be wrong." That comparison is invalid, and being able to say precisely *why* is the point:

| Factor | COCO | Our task |
|---|---|---|
| **Classes** | 80, many semantically confusable (cow/horse/sheep) | 5, visually distinct from each other |
| **Metric quoted** | mAP@[.5:.95] | mAP@0.5 (COCO's own AP<sub>50</sub> is ~65–70% for strong models, not 50%) |
| **Scene variability** | Arbitrary natural scenes, occlusion, extreme scale range, lighting | Rectangular white pages, consistent orientation after deskew, bounded scale range |
| **Object appearance** | Enormous intra-class variation | A stamp looks like a stamp; a ruled table looks like a ruled table |
| **Background** | Unconstrained | Almost always white paper |

The apples-to-apples statement: a strong COCO model scores roughly **65–70% AP<sub>50</sub> across 80 hard classes in unconstrained scenes**; we scored **92% AP<sub>50</sub> across 5 visually-distinct classes on deskewed white pages**. That's the comparison that holds up, and it makes 92% look like a reasonable domain-specific result rather than a miracle.

> **Say this when they push:** *"92% is mAP at IoU 0.5. Under COCO's stricter 0.5-to-0.95 averaging the same model is around 0.70, and I'd quote that number if boundary precision were the requirement. It wasn't — downstream cropped each region with a small margin before OCR, so IoU 0.5 was the threshold that actually mapped to 'is this usable', and optimising for tighter masks would have been optimising for a metric nobody consumed."*

#### What I Did *Not* Measure, and Should Have

Worth volunteering, because it's the honest gap:
- **Mask AP separately from box AP.** Mask R-CNN produces both, and I reported the box metric. For a model whose selling point is instance masks, mask AP is arguably the more relevant number.
- **AP by object size** (COCO's AP<sub>S</sub>/AP<sub>M</sub>/AP<sub>L</sub>). Small-object performance is where FPN earns its keep, and I never isolated it.
- **Confidence calibration.** The routing logic in §3.7 depends on the confidence score meaning something. I checked it informally by bucketing; I didn't produce a reliability diagram.

---

### 3.6 Error Analysis — What the Model Actually Got Wrong

A per-class AP table tells you *which* class is weak. It doesn't tell you *why*, and "what did the model fail on?" is a question that separates people who trained a model from people who ran a training script.

**Failure taxonomy, by observed frequency:**

| Failure mode | What it looked like | Root cause | What I did |
|---|---|---|---|
| **Handwriting boundary disagreement** | Detection found the handwriting but IoU landed at 0.4–0.6, scoring as a miss at higher thresholds | Annotators genuinely disagreed on where a handwritten note "ends" — tight around ink vs. the whole annotated region | Tightened the annotation guideline with worked examples; this was as much a labelling problem as a model problem |
| **Signature ↔ handwriting confusion** | Class swap on cursive signatures inside a handwritten block | The two classes are visually continuous, not discrete — a signature *is* handwriting with a specific function | Accepted it as an inherent class-definition weakness; downstream treated both as "needs human eyes" so the confusion was mostly harmless |
| **Stamps overlapping signatures** | One merged detection instead of two instances, or the mask of one bleeding into the other | Overlapping instances are the hardest case for NMS — a high-IoU pair of *different* objects looks identical to a duplicate | Lowered the NMS IoU threshold and evaluated Soft-NMS; this is precisely the case where instance segmentation earns its keep over semantic segmentation |
| **Multi-column / landscape pages** | Wide tables split into two detections, or a header detected per column | Under-represented in training data; the anchor aspect ratios `[0.5, 1, 2]` don't cover very wide, flat objects well | Flagged for anchor re-tuning via k-means on ground-truth boxes — a known, unimplemented improvement |
| **Very low-quality scans (fax, third-generation photocopy)** | Confidence collapsed across all classes | Training set skewed toward good scans | Added noise/JPEG-artefact augmentation, which helped but didn't close the gap |
| **Pre-printed letterhead logos** | Detected as stamps | Visually similar — a coloured, roughly circular graphic in a corner | The single most common false positive. A "logo" negative class would have fixed it; I ran out of annotation budget |
| **Multi-page documents where page 1 differs** | Fine — but throughput cost, since every page was scored | Not a model failure, a pipeline design point | Noted for batching optimisation |

**The two numbers that matter for the downstream system:** false positives cost a *wasted human review*, false negatives cost a *missed region* — and in a document pipeline a missed region is far more expensive, because the information silently never reaches OCR. So I tuned `DETECTION_MIN_CONFIDENCE` toward recall (0.7 rather than 0.9) and let the review queue absorb the extra false positives. That trade-off is the direct link between §3.5 and §3.7.

---

### 3.7 The 63% Manual-Review Reduction — How It Was Measured

This is the business half of resume bullet #1, and it's the number most likely to be probed, because business metrics are where people inflate.

#### What the Baseline Actually Was

Before the model, **every document region that fed the extraction pipeline was visually confirmed by a human operator.** An operator opened the scan, identified the regions of interest, and confirmed or corrected the boundaries before extraction ran. That is the denominator: **100% of processed regions touched a human.**

#### What Changed

The model didn't remove humans; it **triaged** them. Detections were routed by a confidence-and-stakes rule:

```
for each detected region:
    if confidence >= AUTO_ACCEPT_THRESHOLD and class not in HIGH_STAKES:
        → auto-accept, no human review
    elif confidence < REVIEW_THRESHOLD:
        → route to human review queue
    elif class in HIGH_STAKES:            # signature, stamp — legal significance
        → route to human review queue regardless of confidence
    else:
        → auto-accept

    # Safety net, independent of the above:
    if page produced zero detections but was expected to contain regions:
        → route the whole page to review
```

Two design points worth stating unprompted:

1. **High-stakes classes were never auto-accepted.** Signatures and stamps carry legal weight; the client's risk appetite for a missed or misattributed signature was zero. So the 63% comes *entirely* from the easier classes, which is a more conservative result than a naive threshold sweep would produce.
2. **A zero-detection page always went to review.** A model that detects nothing is indistinguishable from a blank page, and silently dropping a page is the worst possible failure. This deliberately gave back some of the automation.

#### The Measurement

```
Review reduction = 1 − (regions routed to human review ÷ regions that previously required review)
```

Measured over a **fixed, matched sample of production documents** — the same client mix, the same document-type mix, the same month-over-month volume band — so the comparison wasn't confounded by an easier batch. Both numerator and denominator counted **regions**, not documents or pages; mixing units here is the classic way this kind of metric gets inflated.

| | Before | After |
|---|---|---|
| Regions requiring human confirmation | 100% (all of them) | 37% |
| — of which: below confidence threshold | — | ~22% |
| — of which: high-stakes class (auto-routed) | — | ~13% |
| — of which: zero-detection page safety net | — | ~2% |
| **Regions auto-accepted** | 0% | **63%** |

#### The Honest Caveats

- **63% is a reduction in review *volume*, not in review *headcount*.** The team wasn't cut by 63%; capacity was redirected to a backlog and to harder documents. I'd never claim a cost saving I didn't measure.
- **It depends on the threshold**, and the threshold was a business decision, not an optimisation output. Raising the auto-accept threshold would have pushed the number below 63% while reducing risk; lowering it would have pushed above 63% while letting more errors through unseen. The client chose the operating point after seeing the precision-at-threshold curve.
- **The measurement period was finite.** It was measured over a matched production sample, not tracked for a year. Drift in document mix would move it.
- **The 63% and the 92% are linked.** If the model's confidence weren't well-ordered, the auto-accepted 63% would contain errors. The safeguard was an audit: a random sample of auto-accepted regions was periodically re-reviewed by a human to confirm the auto-accept error rate stayed inside the agreed tolerance. Without that audit loop the 63% would be a claim rather than a measurement.

> **Say this:** *"The baseline was that every region was confirmed by a human. After deployment, 37% of regions still went to a human — low-confidence ones, plus signatures and stamps which we never auto-accepted because of their legal significance, plus a safety net that routed any page with zero detections. So 63% of regions stopped needing human eyes. It's a reduction in review volume measured on a matched document sample, not a headcount saving, and it was backed by a periodic audit of the auto-accepted regions to confirm we weren't quietly shipping errors."*

---

### 3.8 Docker & CI/CD — Standardising Model Releases

Resume bullet #4. The framing that matters: containerisation on an ML project is not about deployment convenience, it's about **making "the model" a well-defined object**. A model is not a `.h5` file — it's weights plus preprocessing plus a specific CUDA/cuDNN/TensorFlow combination, and if any of those drift, inference silently changes.

#### Why Containers Specifically Mattered Here

| Problem before | How the container fixed it |
|---|---|
| Training ran on a GPU VM with hand-installed CUDA; inference ran elsewhere. Version skew produced *different predictions from the same weights* — the worst class of bug because nothing errors. | One pinned base image with a fixed CUDA/cuDNN/TF triple, used by both training and inference. |
| Matterport Mask R-CNN pins to older TF/Keras APIs; a stray `pip install -U` broke the environment repeatedly. | Fully pinned `requirements.txt` with hashes, installed at build time, never at runtime. |
| Onboarding a new engineer meant a day of environment setup. | `docker compose up` and you're running. |
| "It worked in my notebook" was an untestable claim. | CI ran the same image the deployment used. |

#### The Image Layout

```dockerfile
# ---------- Stage 1: build wheels once, in a fat image -------------------
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3-pip python3.8-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Pinned + hash-checked: the environment is part of the model artifact
RUN pip3 wheel --require-hashes -r requirements.txt -w /wheels

# ---------- Stage 2: slim runtime ---------------------------------------
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip3 install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

WORKDIR /app
COPY src/ /app/src/
COPY configs/ /app/configs/

# Weights are NOT baked in — they are pulled at startup by version tag,
# so the same image can serve any model version and rollback is a tag change.
ENV MODEL_BLOB_CONTAINER=model-artifacts \
    MODEL_VERSION=v3 \
    DETECTION_MIN_CONFIDENCE=0.7

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python3 -c "import requests,sys; \
        sys.exit(0 if requests.get('http://localhost:8000/health', timeout=5).ok else 1)"

EXPOSE 8000
CMD ["python3", "-m", "src.serve"]
```

**The design decision worth defending:** *weights are not baked into the image.* They're pulled from Azure Blob at startup by version tag. This decouples code releases from model releases — a bug fix in the preprocessing doesn't require re-uploading a 250 MB weights file, and a model rollback is an environment-variable change rather than an image rebuild. The cost is a startup dependency on Blob availability, mitigated by a local cache volume.

Multi-stage build matters here too: the `devel` CUDA image with build toolchain is several gigabytes; the `runtime` image without it is far smaller, which matters when you're pulling it onto autoscaled inference nodes.

#### The CI/CD Pipeline

```yaml
# azure-pipelines.yml — model release pipeline
trigger:
  branches: { include: [main] }
  paths:    { include: [src/*, configs/*, requirements.txt, Dockerfile] }

variables:
  IMAGE: atcsregistry.azurecr.io/docdetect-inference
  TAG: $(Build.BuildNumber)

stages:
  - stage: Quality
    jobs:
      - job: StaticChecks
        steps:
          - script: pip install -r requirements-dev.txt
          - script: black --check src/ && isort --check src/
            displayName: Formatting
          - script: flake8 src/ --max-line-length=100
            displayName: Lint
          - script: mypy src/ --ignore-missing-imports
            displayName: Types

      - job: UnitTests
        steps:
          - script: pytest tests/unit -v --cov=src --cov-fail-under=75
            displayName: Unit tests (no GPU, no model weights)

  - stage: Build
    dependsOn: Quality
    jobs:
      - job: BuildImage
        steps:
          - task: Docker@2
            inputs:
              command: build
              repository: $(IMAGE)
              tags: |
                $(TAG)
                latest
          - script: |
              trivy image --exit-code 1 --severity HIGH,CRITICAL $(IMAGE):$(TAG)
            displayName: Vulnerability scan

  - stage: SmokeInference
    dependsOn: Build
    jobs:
      - job: GoldenSetRegression
        steps:
          # The gate that actually protects the model: run the built image
          # against a frozen 30-image golden set and assert mAP has not regressed.
          - script: |
              docker run --rm \
                -v $(pwd)/tests/golden:/data \
                -e MODEL_VERSION=$(MODEL_VERSION) \
                $(IMAGE):$(TAG) \
                python3 -m src.evaluate --data /data --out /data/result.json
              python3 scripts/assert_no_regression.py \
                --result tests/golden/result.json \
                --baseline tests/golden/baseline_metrics.json \
                --tolerance 0.01
            displayName: Golden-set mAP regression gate

  - stage: Publish
    dependsOn: SmokeInference
    jobs:
      - job: PushAndDeploy
        steps:
          - task: Docker@2
            inputs: { command: push, repository: $(IMAGE), tags: '$(TAG)' }
          - script: |
              az container create \
                --resource-group ml-inference-rg \
                --name docdetect-$(TAG) \
                --image $(IMAGE):$(TAG) \
                --environment-variables MODEL_VERSION=$(MODEL_VERSION)
            displayName: Deploy to Azure Container Instances
```

#### The Testing Strategy — Three Layers

The interesting question in ML CI is *what can you actually assert*, given that model outputs aren't deterministic-by-inspection.

| Layer | What it tests | Runs where | Why it's the right scope |
|---|---|---|---|
| **Unit tests** (fast, no GPU, no weights) | Pure functions: image resizing preserves aspect ratio; polygon→mask conversion is correct; NMS removes the right boxes on synthetic input; confidence-routing logic sends the right region to the right queue; the COCO annotation parser handles malformed input | Every commit, seconds | These are ordinary software bugs and they're where most real defects live. No model needed. |
| **Contract / schema tests** | Inference output conforms to the agreed JSON schema — required keys, value ranges, mask dimensions matching the box, confidence in [0,1] | Every commit | Protects every downstream consumer from a silent format change. |
| **Golden-set regression test** | The built image, running real weights on a frozen 30-image set, produces mAP within 1 point of the recorded baseline | Every build, minutes | **The only test that catches a genuine model regression.** A refactor of the preprocessing that subtly changes normalisation will pass every unit test and quietly cost you 5 points of mAP. This is the gate that catches it. |

The golden set was deliberately chosen to include the hard cases from §3.6 — a multi-column page, a fax-quality scan, an overlapping stamp-and-signature — rather than 30 easy pages. A regression test made of easy examples is a regression test that never fires.

> **Say this:** *"The container made the model a well-defined object — pinned CUDA, TensorFlow and dependency versions, so training and inference couldn't drift apart and produce different predictions from the same weights. And the test that actually mattered wasn't a unit test; it was a golden-set regression gate that ran the built image against thirty frozen hard cases and failed the build if mAP moved more than a point. That's the one that catches a preprocessing refactor silently costing you accuracy."*

---

### 3.9 Azure Deployment — Model Artifacts & Inference Serving

```
┌────────────────────────────────────────────────────────────────────┐
│                  INFERENCE DEPLOYMENT ARCHITECTURE                  │
│                                                                    │
│  ┌──────────────────────────────────┐                              │
│  │    Azure Blob Storage             │                              │
│  │    Container: model-artifacts     │                              │
│  │                                   │                              │
│  │    /models/                       │                              │
│  │      mask_rcnn_document_v1.h5     │  ← Model weights            │
│  │      config.json                  │  ← Hyperparameters          │
│  │      class_names.json             │  ← Label mapping            │
│  │    /inference-results/            │  ← Detection outputs        │
│  └──────────────┬───────────────────┘                              │
│                  │                                                  │
│                  ▼                                                  │
│  ┌──────────────────────────────────┐                              │
│  │    Inference Service              │                              │
│  │    (Azure VM / Container)         │                              │
│  │                                   │                              │
│  │    1. Pull model from Blob        │                              │
│  │    2. Load into TF Serving        │                              │
│  │    3. Accept document images      │                              │
│  │    4. Run Mask R-CNN inference     │                              │
│  │    5. Return:                     │                              │
│  │       - Bounding boxes            │                              │
│  │       - Class labels + confidence │                              │
│  │       - Pixel-level masks         │                              │
│  │    6. Write results back to Blob  │                              │
│  └──────────────────────────────────┘                              │
│                                                                    │
│  Azure Blob SDK (Python):                                          │
│  from azure.storage.blob import BlobServiceClient                  │
│  blob_client = BlobServiceClient.from_connection_string(conn_str)  │
│  container = blob_client.get_container_client("model-artifacts")   │
│  blob = container.get_blob_client("models/mask_rcnn_document_v1.h5")│
│  with open("local_model.h5", "wb") as f:                           │
│      f.write(blob.download_blob().readall())                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Why Azure Blob Storage (not Azure ML Registry, not S3)?**
- Client was on Azure ecosystem — Blob was already provisioned
- Simple versioning via blob naming (v1, v2, etc.) or blob snapshots
- Cost-effective for storing large model files (~250 MB per model)
- Direct integration with Azure VMs and containers running inference
- No additional service overhead (vs. spinning up Azure ML)

**How this composes with the container from §3.8:** the image holds the *code and environment*; Blob holds the *weights and config*, tagged by version. At startup the container reads `MODEL_VERSION` and pulls the matching artefacts. That's what makes a rollback an environment-variable change instead of an image rebuild, and it's why a code fix and a model refresh can ship independently.

```python
import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient

CACHE = Path("/var/cache/models")


def fetch_model_artifacts(version: str = None) -> Path:
    """Pull weights + config for a given model version, with a local cache.

    The container is version-agnostic; the version is injected at runtime,
    so rollback is a config change rather than an image rebuild.
    """
    version = version or os.environ["MODEL_VERSION"]
    local = CACHE / version
    if (local / "weights.h5").exists():        # already warm on this node
        return local

    local.mkdir(parents=True, exist_ok=True)
    container = (BlobServiceClient
                 .from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
                 .get_container_client(os.environ["MODEL_BLOB_CONTAINER"]))

    for name in ("weights.h5", "config.json", "class_names.json"):
        blob = container.get_blob_client(f"models/{version}/{name}")
        with open(local / name, "wb") as f:
            f.write(blob.download_blob().readall())

    return local
```

**Inference throughput realities worth mentioning:** Mask R-CNN with a ResNet-101-FPN backbone at 1024×1024 is not cheap — roughly a few hundred milliseconds per page on a single GPU, and multiple seconds on CPU. Three things that mattered in practice: batching pages rather than scoring one at a time, keeping the model resident in memory rather than reloading per request (which is why the container has a long `start-period` on its healthcheck), and running the whole document batch asynchronously with results written back to Blob rather than pretending this was a synchronous request/response API.

---

## 4. Project C: Legal Document Classification (NLP)

### 4.1 Problem Context

Legal teams across ATCS clients needed to classify incoming documents (contracts, court filings, briefs, memos, compliance documents) into predefined categories for routing, archival, and compliance tracking. Manual classification was:
- **Slow:** Paralegal teams spending hours per batch
- **Inconsistent:** Different annotators applied labels differently
- **Costly:** Senior legal staff reviewing edge cases

**Goal:** Build an automated classification pipeline with transfer learning and validate labeling accuracy.

---

### 4.2 NLP Preprocessing Pipeline

Legal text has unique characteristics that require domain-specific preprocessing:

```python
import spacy
import re
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_lg")

class LegalTextPreprocessor:
    """Domain-aware preprocessing for legal documents."""
    
    # Legal-specific stopwords (common but non-discriminative in legal text)
    LEGAL_STOPWORDS = {
        "herein", "hereinafter", "thereof", "thereby", "whereas",
        "pursuant", "notwithstanding", "aforementioned", "undersigned",
        "witnesseth", "hereunder", "heretofore"
    }
    
    # Legal entity patterns
    LEGAL_ENTITY_PATTERNS = [
        r"(?:Section|§)\s*\d+[\.\d]*",           # Section references
        r"(?:Article|Art\.)\s+[IVXLC]+",           # Article references  
        r"\d+\s+U\.?S\.?C\.?\s+§?\s*\d+",         # USC citations
        r"\d+\s+F\.\s*(?:2d|3d|4th)\s+\d+",       # Federal Reporter
        r"[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+",       # Case names
    ]
    
    def __init__(self, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.max_length = max_length
    
    def clean_text(self, text):
        """Remove formatting artifacts from legal documents."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'_{3,}', '', text)
        text = re.sub(r'-{3,}', '', text)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_legal_entities(self, text):
        """Extract legal citations and references."""
        entities = []
        for pattern in self.LEGAL_ENTITY_PATTERNS:
            entities.extend(re.findall(pattern, text))
        
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PERSON", "LAW", "DATE"):
                entities.append(f"{ent.label_}:{ent.text}")
        
        return entities
    
    def segment_sections(self, text):
        """Split legal document into logical sections."""
        section_pattern = r'(?:^|\n)(?:ARTICLE|SECTION|CLAUSE|WHEREAS|NOW\s+THEREFORE)\s+'
        sections = re.split(section_pattern, text, flags=re.IGNORECASE)
        return [s.strip() for s in sections if s.strip()]
    
    def preprocess(self, text):
        """Full preprocessing pipeline."""
        text = self.clean_text(text)
        entities = self.extract_legal_entities(text)
        sections = self.segment_sections(text)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "entities": entities,
            "num_sections": len(sections)
        }
```

---

### 4.3 Transfer Learning for Legal Document Classification

#### Why Transfer Learning for NLP?

```
From-scratch training:                 Transfer learning:
─────────────────────                  ───────────────────
• Need millions of labeled samples     • Need hundreds to thousands
• Weeks of training on GPU clusters    • Hours of fine-tuning on single GPU
• No language understanding            • Pre-learned syntax, semantics, reasoning
• Risk: poor generalization            • Strong baseline from pretraining
```

#### Model Selection

| Model | Parameters | Pretraining | Legal Domain? | Our Choice |
|-------|-----------|-------------|---------------|------------|
| BERT-base | 110M | General English (BooksCorpus, Wikipedia) | No | Baseline |
| **Legal-BERT** | 110M | Legal corpora (case law, legislation, contracts) | **Yes** | **Selected** |
| RoBERTa | 125M | General English (larger corpus) | No | Considered |
| Longformer | 149M | Long documents (up to 4096 tokens) | No | For long docs |

**Legal-BERT was selected** because it was pretrained on 12 GB of legal text (EU legislation, UK/US case law, contracts), giving it superior understanding of legal terminology, citation patterns, and clause structures.

#### Fine-Tuning Architecture

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LegalDocumentClassifier(nn.Module):
    """Multi-label legal document classifier using Legal-BERT."""
    
    def __init__(self, num_labels, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        hidden_size = self.bert.config.hidden_size  # 768
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)
        return logits

# Document categories
LEGAL_CATEGORIES = [
    "contract", "court_filing", "legal_brief", "memo",
    "compliance", "patent", "regulatory", "correspondence"
]

model = LegalDocumentClassifier(num_labels=len(LEGAL_CATEGORIES))

# Class-weighted loss for imbalanced data
class_weights = torch.tensor([1.0, 2.5, 1.8, 1.2, 3.0, 2.0, 2.2, 1.5])
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=500  # warmup
)
```

#### Training Loop

```python
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels.float())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= threshold).int()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    report = classification_report(
        all_labels, all_preds,
        target_names=LEGAL_CATEGORIES,
        zero_division=0
    )
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    return macro_f1, report
```

---

### 4.4 Document Ingestion Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   DOCUMENT INGESTION PIPELINE                           │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │ Scanned PDFs│  │ Digital PDFs│  │ Word Docs   │                     │
│  │ (images)    │  │ (text-based)│  │ (.docx)     │                     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                     │
│         │                │                │                             │
│         ▼                ▼                ▼                             │
│  ┌──────────────────────────────────────────────────┐                  │
│  │          Text Extraction Module                   │                  │
│  │  • Scanned → OCR (Tesseract / Azure Form Recog.) │                  │
│  │  • Digital PDF → PyMuPDF / pdfplumber             │                  │
│  │  • DOCX → python-docx                             │                  │
│  └──────────────────────┬───────────────────────────┘                  │
│                          ▼                                              │
│  ┌──────────────────────────────────────────────────┐                  │
│  │          NLP Preprocessing                        │                  │
│  │  • Legal tokenization                             │                  │
│  │  • Entity extraction                              │                  │
│  │  • Section segmentation                           │                  │
│  │  • Feature extraction                             │                  │
│  └──────────────────────┬───────────────────────────┘                  │
│                          ▼                                              │
│  ┌──────────────────────────────────────────────────┐                  │
│  │          Classification Model                     │                  │
│  │  • Legal-BERT inference                           │                  │
│  │  • Multi-label prediction                         │                  │
│  │  • Confidence scoring per label                   │                  │
│  └──────────────────────┬───────────────────────────┘                  │
│                          ▼                                              │
│  ┌──────────────────────────────────────────────────┐                  │
│  │          Validation & Routing                     │                  │
│  │  • Confidence thresholding                        │                  │
│  │  • Low-confidence → human review queue            │                  │
│  │  • High-confidence → auto-tagged and routed       │                  │
│  │  • Results logged for labeling accuracy tracking  │                  │
│  └──────────────────────────────────────────────────┘                  │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### 4.5 Labeling Accuracy Validation

A core deliverable was validating that model labels matched human annotations reliably.

```python
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    cohen_kappa_score, f1_score
)

class LabelingAccuracyValidator:
    """Validate model predictions against human annotations."""
    
    def __init__(self, class_names):
        self.class_names = class_names
    
    def compute_confusion_matrix(self, y_true, y_pred):
        """Per-class confusion matrix for multi-label."""
        results = {}
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            tn, fp, fn, tp = cm.ravel()
            results[class_name] = {
                "TP": int(tp), "FP": int(fp), 
                "FN": int(fn), "TN": int(tn),
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            }
            p, r = results[class_name]["precision"], results[class_name]["recall"]
            results[class_name]["f1"] = 2*p*r / (p+r) if (p+r) > 0 else 0
        return results
    
    def compute_cohens_kappa(self, y_true, y_pred):
        """
        Cohen's Kappa measures inter-annotator agreement 
        (model vs. human), correcting for chance agreement.
        
        κ = (p_o - p_e) / (1 - p_e)
        
        where:
          p_o = observed agreement
          p_e = expected agreement by chance
        
        Interpretation:
          κ < 0.20  → Poor agreement
          0.20-0.40 → Fair
          0.41-0.60 → Moderate  
          0.61-0.80 → Substantial
          0.81-1.00 → Almost perfect
        """
        kappas = {}
        for i, class_name in enumerate(self.class_names):
            kappas[class_name] = cohen_kappa_score(y_true[:, i], y_pred[:, i])
        kappas["macro_avg"] = np.mean(list(kappas.values()))
        return kappas
    
    def full_validation_report(self, y_true, y_pred):
        """Generate comprehensive validation report."""
        cm_results = self.compute_confusion_matrix(y_true, y_pred)
        kappas = self.compute_cohens_kappa(y_true, y_pred)
        
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        
        report = {
            "per_class_metrics": cm_results,
            "cohens_kappa": kappas,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.class_names, zero_division=0
            )
        }
        return report
```

**Cohen's Kappa Formula:**

```
            p_o - p_e
    κ = ──────────────
            1 - p_e

where:
    p_o = (TP + TN) / N                          (observed agreement)
    p_e = (TP+FP)(TP+FN)/N² + (FN+TN)(FP+TN)/N²  (expected by chance)

Example:
    100 documents, model agrees with human on 85
    By chance, agreement would be 52%
    
    κ = (0.85 - 0.52) / (1 - 0.52) = 0.33 / 0.48 = 0.69 (Substantial agreement)
```

---

## 5. Topics You Must Know (Comprehensive Study Guide)

### 5.1 PySpark & Data Engineering

| Topic | Key Concepts | Why It Matters |
|-------|-------------|----------------|
| **RDD vs DataFrame** | RDD is low-level untyped; DataFrame has schema + Catalyst optimization | Explain performance difference, when to use each |
| **Transformations vs Actions** | Lazy vs eager; DAG construction; `explain()` to see plan | Shows understanding of Spark execution model |
| **Catalyst Optimizer** | Logical plan → optimized logical plan → physical plan → code gen | Predicate pushdown, column pruning, join reordering |
| **Tungsten Engine** | Off-heap memory, whole-stage code generation, cache-aware computation | Explains why Spark is fast |
| **Partitioning** | `repartition()` vs `coalesce()`; hash vs range partitioning | Shuffle performance, write optimization |
| **Broadcast Joins** | Small table broadcast to all executors; avoids shuffle | Critical for ETL joins with dimension tables |
| **Data Skew** | Uneven key distribution → some tasks take much longer | Salting, repartitioning, adaptive query execution |
| **JDBC Integration** | Connection pooling, batch size tuning, predicate pushdown | Production loading patterns |
| **Checkpointing** | Truncate lineage for long DAGs; `persist()` vs `checkpoint()` | Fault tolerance in long pipelines |
| **ETL Design Patterns** | Slowly Changing Dimensions, CDC, idempotency, deduplication | Production data engineering fundamentals |

### 5.2 Object Detection & Computer Vision

| Topic | Key Concepts | Why It Matters |
|-------|-------------|----------------|
| **R-CNN Family Evolution** | R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN | Shows understanding of progression and why each improvement was needed |
| **Anchor Boxes** | Predefined boxes at each feature map location; scales + ratios | Core to RPN; tuning anchors affects detection quality |
| **Feature Pyramid Network** | Top-down + lateral connections; multi-scale detection | Detecting small and large objects simultaneously |
| **ROI Align vs ROI Pooling** | Bilinear interpolation vs quantization; alignment matters for masks | Key innovation in Mask R-CNN |
| **Non-Maximum Suppression** | Remove redundant detections; IoU threshold selection | Post-processing step; soft-NMS as alternative |
| **IoU (Intersection over Union)** | Overlap metric; used for matching detections to ground truth | Foundation of all detection metrics |
| **mAP Computation** | PR curve → AP → average across classes → average across IoU thresholds | Know PASCAL VOC vs COCO evaluation protocols |
| **Transfer Learning** | Pretrained backbone → freeze → fine-tune; learning rate scheduling | Standard practice; explain why it works |
| **Data Augmentation** | Geometric (flip, rotate, crop), photometric (brightness, contrast) | Critical for small datasets |
| **Instance vs Semantic Segmentation** | Instance: per-object masks; Semantic: per-pixel class labels | Explains why Mask R-CNN, not U-Net |

### 5.3 NLP & Text Classification

| Topic | Key Concepts | Why It Matters |
|-------|-------------|----------------|
| **BERT Architecture** | Bidirectional encoder; [CLS] token; MLM + NSP pretraining | Foundation of transfer learning for NLP |
| **Tokenization** | WordPiece (BERT), BPE (GPT), SentencePiece (T5) | Subword tokenization handles OOV words |
| **Fine-Tuning Strategy** | Freeze/unfreeze layers; learning rate warmup; discriminative LR | Practical transfer learning knowledge |
| **Multi-Label Classification** | BCE loss per label; sigmoid (not softmax); threshold tuning | Different from multi-class; explain the distinction |
| **Class Imbalance** | Weighted loss, oversampling (SMOTE), focal loss, stratified splits | Common in real-world classification tasks |
| **Legal NLP Challenges** | Long documents, domain terminology, citation parsing, section structure | Domain-specific preprocessing knowledge |
| **Evaluation Metrics** | Precision, Recall, F1 (micro/macro/weighted), Cohen's Kappa | Know formulas and when to use each |
| **Active Learning** | Model identifies uncertain samples → human labels → retrain | Efficient use of annotation budget |
| **Named Entity Recognition** | spaCy, Hugging Face token classification; legal entity types | Preprocessing for document understanding |
| **Document Embeddings** | [CLS] token, mean pooling, max pooling; Sentence-BERT | Representing documents as vectors |

### 5.4 Azure & Deployment

| Topic | Key Concepts | Why It Matters |
|-------|-------------|----------------|
| **Azure Blob Storage** | Containers, blobs, access tiers (hot/cool/archive), SAS tokens | Model artifact storage and versioning |
| **Azure Blob SDK** | BlobServiceClient, upload/download, streaming, metadata | Practical code knowledge |
| **Model Versioning** | Naming conventions, blob snapshots, metadata tags | Production model management |
| **TF Serving** | SavedModel format, REST/gRPC endpoints, batching | Serving TensorFlow models at scale |
| **Inference Pipeline** | Model loading → preprocessing → inference → postprocessing → output | End-to-end deployment understanding |
| **Cost Optimization** | Right-sizing VMs, spot instances, access tier selection | Real-world deployment considerations |

### 5.5 Core Formulas You Must Know

```
PRECISION:      P = TP / (TP + FP)     "Of all positive predictions, how many were correct?"

RECALL:         R = TP / (TP + FN)     "Of all actual positives, how many did we find?"

F1 SCORE:       F1 = 2PR / (P + R)     "Harmonic mean of precision and recall"

IoU:            IoU = |A ∩ B| / |A ∪ B|   "Overlap between predicted and ground truth boxes"

mAP:            mAP = (1/C) Σ APᵢ         "Average of per-class Average Precision"

AP:             AP = ∫₀¹ p(r) dr           "Area under the Precision-Recall curve"

COHEN'S KAPPA:  κ = (p_o - p_e) / (1 - p_e)   "Agreement correcting for chance"

BCE LOSS:       L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]   "Binary cross-entropy per label"

SMOOTH L1:      L = { 0.5x²      if |x| < 1        "BBox regression loss"
                    { |x| - 0.5   otherwise

CROSS-ENTROPY:  L = -Σ yᵢ·log(ŷᵢ)                  "Classification loss"
```

---

## 6. Interview Questions & Answers

### Project A: PySpark ETL Pipeline

---

**Q1: Walk me through your PySpark ETL pipeline at ATCS.**

**A:** "At ATCS, I built a PySpark-based ETL pipeline that ingested data from multiple heterogeneous sources — CSVs, flat files, and database extracts — validated data quality rigorously, and loaded clean data into MSSQL for downstream analytics.

The pipeline had four stages:
1. **Extraction**: PySpark read from various formats using explicit schemas (never relying solely on inference in production) with corrupt record tracking.
2. **Validation**: A reusable DataQualityValidator class ran schema checks, null analysis, duplicate detection, range validation, and referential integrity checks. If critical checks failed, the pipeline halted before any data reached MSSQL.
3. **Transformation**: Standardization (trimming, type casting), derived column creation, and deduplication.
4. **Loading**: Batch writes to MSSQL via JDBC with tuned batch sizes (10,000 rows per batch, which we benchmarked as optimal). For upsert scenarios, we used a staging table approach with MSSQL's MERGE statement."

---

**Q2: Why PySpark instead of plain Python/Pandas?**

**A:** "Three reasons. First, data volumes exceeded single-machine memory — we were processing gigabytes per day, and Pandas would have required chunking and manual parallelism. Second, PySpark's Catalyst optimizer automatically optimized our query plans — things like predicate pushdown when reading from JDBC, column pruning, and join reordering. We got 2-5x better performance over equivalent Pandas operations without manual tuning. Third, PySpark gave us a clear path to scale horizontally. When data volumes grew, we could add executors without rewriting code."

---

**Q3: What data quality checks did you implement, and why?**

**A:** "I built a configurable DataQualityValidator framework with five categories of checks:

1. **Schema validation**: Verified column names, data types, and nullable constraints matched expected schemas. This caught upstream changes early.
2. **Null analysis**: Flagged columns exceeding configurable null thresholds — 0% for critical fields like primary keys, 5% for non-critical fields.
3. **Duplicate detection**: Identified duplicate rows based on composite keys using `dropDuplicates()`.
4. **Range validation**: Ensured numeric fields fell within business-valid ranges (e.g., confidence scores between 0 and 1, page counts between 1 and 10,000).
5. **Referential integrity**: Verified foreign keys existed in reference tables using left-anti joins.

The key design decision was making this a pre-load gate. If critical checks failed, the pipeline raised a `DataQualityError` and halted, ensuring no dirty data ever reached MSSQL. Warnings (like unexpected extra columns) were logged but didn't block the load."

---

**Q4: How did you handle the PySpark-to-MSSQL JDBC connection? Any performance issues?**

**A:** "JDBC writing was the main bottleneck initially. Three optimizations made a major difference:

1. **Batch size tuning**: Default JDBC batch size is 1,000 rows. We benchmarked different sizes and found 10,000 to be optimal — it reduced our 2M-row load from 45 minutes to 12 minutes. Going higher (50,000) gave diminishing returns and increased memory pressure.

2. **Partition-aware writing**: We repartitioned the DataFrame before writing to match the MSSQL table's natural partitioning, which improved parallelism.

3. **Staging + MERGE pattern**: For incremental loads, we wrote to a staging table first (overwrite mode), then used MSSQL's MERGE statement to handle upsert logic. This was more efficient than trying to do row-level upsert through JDBC."

---

**Q5: Explain the difference between ETL and ELT. Why did you choose ETL?**

**A:** "ETL transforms data in the processing engine before loading to the target. ELT loads raw data first and transforms inside the target database.

We chose ETL because MSSQL was used primarily for reporting and analytics — not as a data lake with powerful compute. Loading unvalidated data would have corrupted dashboards. With ETL, PySpark acted as a quality gate: data was cleaned, validated, and transformed before it ever entered MSSQL. If we'd been loading into Snowflake or BigQuery, ELT might have made sense because those systems have strong compute engines that can handle transformation efficiently."

---

**Q6: What are narrow vs. wide transformations in Spark? Why does it matter?**

**A:** "Narrow transformations — like `map`, `filter`, `select`, `withColumn` — can be computed on each partition independently without data movement. Wide transformations — like `groupBy`, `join`, `distinct`, `orderBy` — require shuffling data across the network to co-locate matching keys.

It matters because shuffles are the number one performance bottleneck in Spark. Each shuffle means serializing data, writing to disk, transferring over the network, and deserializing. In our pipeline, I minimized shuffles by filtering early (predicate pushdown), using broadcast joins for small dimension tables, and coalescing partitions instead of repartitioning when reducing partition count."

---

### Project B: Mask R-CNN

---

**Q7: Explain the Mask R-CNN architecture and why you chose it.**

**A:** "Mask R-CNN extends Faster R-CNN by adding a parallel mask prediction branch. The architecture has four key stages:

1. **Backbone (ResNet-101 + FPN)**: Extracts multi-scale feature maps. ResNet-101 provides deep feature extraction with residual connections that solve vanishing gradients. FPN creates a feature pyramid with top-down connections, enabling detection of objects at different scales.

2. **Region Proposal Network (RPN)**: Slides over feature maps and proposes ~2000 candidate regions using anchor boxes. Each anchor gets an objectness score and bounding box refinement.

3. **ROI Align**: This is a key innovation over Faster R-CNN's ROI Pooling. Instead of quantizing ROI coordinates (which causes misalignment), ROI Align uses bilinear interpolation to sample features at exact floating-point positions. This is critical for mask quality — even 1-pixel misalignment degrades masks significantly.

4. **Three parallel heads**: Classification head (what class), bounding box head (refined coordinates), and mask head (28x28 binary mask per class).

I chose Mask R-CNN because we needed instance segmentation — not just bounding boxes. When a signature overlaps a stamp, we needed separate masks for each. YOLO and Faster R-CNN only give bounding boxes; U-Net gives semantic segmentation without distinguishing instances."

---

**Q8: Explain your transfer learning approach for Mask R-CNN.**

**A:** "I used a two-phase transfer learning strategy:

**Phase 1 — Head training (20 epochs)**: I loaded COCO-pretrained weights for the entire network but froze the ResNet-101 backbone and FPN. Only the RPN, classification head, bbox head, and mask head were trainable. I excluded the final classification and bbox layers from weight loading since our class count (5 document classes) differed from COCO (80 classes). Learning rate was 0.001.

**Phase 2 — Full fine-tuning (30 more epochs)**: I unfroze all layers and trained end-to-end with a 10x lower learning rate (0.0001) to avoid catastrophic forgetting. I used step decay at epochs 30 and 40.

The rationale is that early layers learn general features (edges, textures, shapes) that transfer well across domains. By fine-tuning with a lower learning rate, we preserve these general features while adapting higher layers to document-specific patterns."

---

**Q9: What does 92% mAP mean, and how is it calculated?**

**A:** "mAP — Mean Average Precision — is the standard metric for object detection, and I'll be precise about which variant I'm quoting, because it matters.

The mechanics: for each class you sort all detections by confidence, walk down the list matching each detection to a ground-truth box by IoU, and record precision and recall at each step. That gives a saw-toothed precision-recall curve, which is then *interpolated* — at each recall level you take the maximum precision at that recall or higher — and AP is the area under that interpolated step function. mAP is the mean of AP across classes.

**Our 92% is mAP at IoU 0.5** — the PASCAL VOC convention, which COCO calls AP<sub>50</sub>. Under COCO's stricter primary metric, averaging over IoU 0.50 to 0.95 in steps of 0.05, the same model scores about 0.70. Both are real; they answer different questions. IoU 0.5 asks 'did you find the right thing roughly in the right place'; the 0.5-to-0.95 average also grades how tightly you traced the boundary.

I quote the IoU 0.5 number because it's the one that mapped to the business requirement — downstream cropped each region with a small margin before OCR, so a slightly loose box was harmless. Optimising for tighter masks would have been optimising for a metric nobody consumed.

On the inevitable 'but COCO SOTA is 50%' objection: that compares a different metric on a different problem. A strong COCO model scores roughly 65–70% AP<sub>50</sub> across 80 semantically-confusable classes in unconstrained natural scenes. We scored 92% AP<sub>50</sub> across 5 visually-distinct classes on deskewed white pages with a bounded scale range. That's the apples-to-apples comparison, and it makes 92% a reasonable domain-specific result rather than a suspicious one."

---

**Q10: Explain ROI Align and why it's better than ROI Pooling.**

**A:** "ROI Pooling in Faster R-CNN quantizes floating-point ROI coordinates to integer pixel positions. For example, if the ROI starts at x=3.75, it rounds to 4. This quantization happens twice — once when mapping the ROI to the feature map, and once when dividing into bins. The cumulative error can be several pixels, which doesn't matter much for classification but significantly degrades mask quality.

ROI Align avoids all quantization. It keeps ROI coordinates as floating-point, divides the ROI into bins at exact positions, and uses bilinear interpolation to sample 4 points per bin. This gives pixel-perfect alignment between the ROI and the feature map.

The original Mask R-CNN paper showed that ROI Align improved mask AP by about 3 percentage points over ROI Pooling — a substantial gain for a single architectural change."

---

**Q11: How did you deploy the model to Azure Blob Storage?**

**A:** "The deployment had three parts:

1. **Model packaging**: After training, I exported the Mask R-CNN model as a TensorFlow SavedModel format, along with a config file (hyperparameters, class names, preprocessing settings).

2. **Upload to Azure Blob**: Used the Azure Blob Storage SDK to upload model artifacts to a `model-artifacts` container with versioned naming. Each model version was a separate blob with metadata tags (training date, mAP, dataset version).

3. **Inference service**: An inference service running on an Azure VM pulled the latest model from Blob on startup, loaded it into memory, and exposed an API for document processing. For each incoming document image, it ran preprocessing, inference, NMS, and returned bounding boxes, class labels, confidence scores, and pixel-level masks.

Azure Blob was chosen over Azure ML Registry because the client's infrastructure was already Blob-centric, and the simplicity of blob storage fit our versioning needs without the overhead of a full MLOps platform."

---

### Project C: Legal Document Classification

---

**Q12: How did you approach legal document classification using NLP?**

**A:** "I built a multi-label classification pipeline with three main components:

1. **Domain-aware preprocessing**: Legal text is unique — full of citations, cross-references, archaic terms, and long, complex sentences. I built a preprocessing pipeline that handled legal-specific tokenization, extracted legal entities (case citations, statute references, party names), segmented documents into logical sections, and removed formatting artifacts.

2. **Transfer learning with Legal-BERT**: Rather than general BERT, I used Legal-BERT, which was pretrained on 12GB of legal text. This gave the model a strong understanding of legal terminology, citation patterns, and clause structures. I added a classification head (two FC layers with dropout) on top of the [CLS] token output and fine-tuned the entire model.

3. **Labeling accuracy validation**: I built a validation framework that compared model predictions against human annotations using per-class confusion matrices, F1 scores, and Cohen's Kappa to measure inter-rater agreement. Documents with low confidence were routed to a human review queue."

---

**Q13: Why Legal-BERT over standard BERT?**

**A:** "Standard BERT was pretrained on BooksCorpus and Wikipedia — general English text. Legal text has a fundamentally different distribution: different vocabulary (tort, estoppel, indemnification), different sentence structures (extremely long, nested clauses), different conventions (citation formats, section numbering).

Legal-BERT was pretrained on 12GB of legal text including EU legislation, UK case law, US contracts, and regulatory filings. In our experiments, Legal-BERT outperformed standard BERT by ~4-5% macro F1 on our classification task — a significant improvement that came 'for free' just by choosing the right pretrained model. The domain-specific vocabulary coverage meant fewer [UNK] tokens and better subword representations of legal terms."

---

**Q14: How does multi-label differ from multi-class classification?**

**A:** "In multi-class, each document belongs to exactly one class — you use softmax activation (probabilities sum to 1) and categorical cross-entropy loss. In multi-label, a document can belong to multiple classes simultaneously — a contract might also be tagged as 'compliance' and 'regulatory.'

For multi-label, I used sigmoid activation (independent probability per class, each between 0 and 1) and binary cross-entropy loss computed independently for each label. At inference, I applied a threshold (tuned via validation) to each sigmoid output to determine which labels to assign. This meant optimizing the threshold was important — too low gives false positives, too high gives false negatives."

---

**Q15: Explain Cohen's Kappa and why you used it.**

**A:** "Cohen's Kappa measures agreement between two raters (in our case, model vs. human annotator), correcting for the probability of agreement by chance.

The formula is κ = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is expected agreement by chance. If the model agrees with humans 85% of the time, but we'd expect 52% agreement by chance (due to class distribution), then κ = (0.85 - 0.52) / (1 - 0.52) = 0.69, which indicates substantial agreement.

I used it because accuracy alone is misleading with imbalanced labels. If 90% of documents are 'contracts', a model that predicts 'contract' for everything gets 90% accuracy but κ ≈ 0. Kappa exposes this. In our system, documents with low per-class Kappa were flagged for annotation review and model retraining."

---

### Cross-Project Questions

---

**Q16: How did the three projects at ATCS relate to each other?**

**A:** "They formed a cohesive document intelligence platform. The PySpark ETL pipeline handled structured data — metadata, client records, processing logs — bringing clean data into MSSQL for analytics. The Mask R-CNN model handled the visual layer — detecting and segmenting elements within scanned documents (stamps, signatures, tables). The NLP classification handled the textual layer — reading document content and classifying by type.

In practice, a scanned document would first go through Mask R-CNN to identify regions, then text extraction (OCR on detected text regions), then NLP classification to tag the document type, and the metadata would flow through the ETL pipeline into MSSQL for reporting and tracking."

---

**Q17: What was the most challenging part of working at ATCS?**

**A:** "The heterogeneity of client data. Each client had different document formats, different quality levels, different labeling conventions. Building reusable, configurable components was essential — the DataQualityValidator had configurable rules per client, the Mask R-CNN was trained on a combined dataset but with client-specific class mappings, and the NLP pipeline had modular preprocessing steps that could be toggled per document type.

The other challenge was annotation quality for the Mask R-CNN task. We had limited annotated data (a few hundred documents), which made transfer learning from COCO essential and data augmentation critical for preventing overfitting."

---

**Q18: How did you handle limited training data for Mask R-CNN?**

**A:** "Four strategies:

1. **Transfer learning**: Starting from COCO-pretrained weights meant the model already understood edges, textures, and spatial relationships. We only needed to teach it document-specific patterns.

2. **Aggressive data augmentation**: Horizontal flip, rotation (±15°), random brightness/contrast adjustment, and random cropping. This effectively multiplied our dataset.

3. **Two-phase training**: First training only the heads (with backbone frozen) meant we needed fewer samples to adapt the detection layers. Only then did we fine-tune the full network with a lower learning rate.

4. **Careful validation**: With limited data, I used stratified train/val/test splits and monitored for overfitting using validation mAP. Early stopping was applied when validation mAP plateaued."

---

**Q19: Walk me through a Spark optimization you performed.**

**A:** "The biggest optimization was addressing a data skew issue in one of our join operations. We were joining a large transaction table (~50M rows) with a client table (~5K rows) on client_id. The join was slow because certain clients had millions of transactions while others had tens.

The fix was a broadcast join. Since the client table was only ~5KB, I used `F.broadcast(client_df)` to send the entire client table to every executor. This eliminated the shuffle entirely — each executor could join its partition of the transaction table with the local copy of the client table. The join went from 15 minutes (with shuffle) to 45 seconds (with broadcast)."

---

**Q20: How would you improve the Mask R-CNN model beyond 92% mAP?**

**A:** "Several paths:

1. **More training data**: The biggest lever. More annotated documents would improve generalization, especially for underrepresented classes like handwriting.

2. **Better backbone**: Try ResNeXt-101 or a Swin Transformer backbone, which have shown improvements over ResNet in detection tasks.

3. **Anchor optimization**: Run K-means on ground truth bounding boxes to find optimal anchor scales and ratios for our specific document layout patterns.

4. **Multi-scale training**: Train with multiple input resolutions (800, 1024, 1333) to make the model more robust to document scanning variations.

5. **Test-time augmentation (TTA)**: Run inference on the image plus flipped/multi-scale versions and merge predictions. This typically adds 1-2% mAP.

6. **Soft-NMS**: Replace hard NMS with Soft-NMS, which decays confidence of overlapping detections instead of removing them entirely. Helps when document elements overlap."

---

**Q21: What's the difference between instance segmentation and semantic segmentation?**

**A:** "Semantic segmentation assigns a class label to every pixel, but doesn't distinguish between instances. If there are two signatures in a document, semantic segmentation labels all signature pixels identically — you can't tell them apart.

Instance segmentation assigns a class label AND a unique identity to each object. With Mask R-CNN, each signature gets its own bounding box, class label, and pixel-level mask. We needed this because downstream processing required extracting each document element individually — knowing that 'there are two signatures' and 'here is each one specifically' matters for verification workflows."

---

**Q22: Explain the multi-task loss function in Mask R-CNN.**

**A:** "Mask R-CNN uses a multi-task loss combining three components:

**L = L_cls + L_bbox + L_mask**

1. **L_cls (Classification loss)**: Cross-entropy loss over N+1 classes (N objects + background). Computed for each ROI.

2. **L_bbox (Bounding box regression loss)**: Smooth L1 loss on the 4 bounding box coordinates (dx, dy, dw, dh). Smooth L1 is less sensitive to outliers than L2:
   - If |x| < 1: loss = 0.5x²
   - Otherwise: loss = |x| - 0.5

3. **L_mask (Mask loss)**: Per-pixel binary cross-entropy, but only for the ground-truth class. This is the key insight — by decoupling mask prediction from classification, we avoid inter-class competition for pixels and improve mask quality.

The mask loss is computed on the predicted mask corresponding to the ground-truth class only, not on all K class masks. This is more efficient and avoids the need for the mask branch to implicitly re-learn classification."

---

**Q23: How did you ensure your ETL pipeline was idempotent?**

**A:** "Idempotency — running the pipeline twice produces the same result as running it once — was achieved through three mechanisms:

1. **Deduplication in the quality layer**: Before loading, we removed duplicates based on composite keys, so re-ingesting the same source data didn't create duplicate records.

2. **Staging + MERGE pattern**: For incremental loads, data went to a staging table first (overwrite mode clears it each run), then MERGE handled the upsert logic — updating existing records and inserting new ones. Running this twice on the same data just updates records to the same values.

3. **Watermarking**: For time-based incremental loads, we tracked a high-water mark (the latest timestamp successfully processed) and only ingested records newer than that mark. This prevented reprocessing and ensured monotonic progress."

---

**Q24: How did you evaluate model performance during the Mask R-CNN fine-tuning process?**

**A:** "I used a multi-metric evaluation approach during training:

1. **Primary metric — mAP@[0.5:0.95]**: The COCO-standard mAP computed on the validation set after each epoch. This was the metric I used for model selection and early stopping.

2. **Per-class AP breakdown**: I tracked AP for each class individually. This helped identify which classes were underperforming — for example, handwriting detection lagged behind stamps and signatures due to high visual variability.

3. **Loss curves**: I monitored training and validation losses for all three heads (classification, bbox, mask) separately. Divergence between training and validation loss indicated overfitting.

4. **Qualitative inspection**: For every validation epoch, I visualized predictions on a fixed set of 20 representative images. This caught issues that metrics alone might miss — like the model correctly detecting elements but with sloppy mask boundaries.

5. **Confidence calibration**: I checked that confidence scores correlated with actual precision. If the model assigned 0.9 confidence, roughly 90% of those predictions should be correct."

---

**Q25: How would you handle a scenario where the ETL pipeline fails mid-load?**

**A:** "Several safeguards:

1. **Transaction semantics**: For the MSSQL MERGE step, the entire operation runs within a transaction. If it fails mid-way, it rolls back — the target table is unchanged.

2. **Staging table isolation**: Data goes to a staging table first. If the MERGE fails, the staging table has the data and can be re-attempted without re-running the full Spark pipeline.

3. **Idempotent design**: Because of deduplication and the MERGE pattern, re-running the pipeline on the same data is safe and produces the correct result.

4. **Alerting and logging**: The pipeline logged each stage's completion with record counts. Any failure triggered an alert with the stage, error message, and a link to the full logs. This enabled quick diagnosis and re-run.

5. **Checkpointing**: For very long-running pipelines, I used Spark checkpointing to persist intermediate DataFrames to disk. If the pipeline crashed after the transform stage but before loading, we could resume from the checkpoint instead of re-reading and re-transforming all data."

---

### Mask R-CNN, Metrics, Deployment & CI/CD (New)

---

**Q26: Why Mask R-CNN and not YOLO, or just a layout parser?** *(trick)*

**A:** "Three candidate approaches, and the choice comes down to what the downstream system consumed.

**Why not a layout parser** — and this is the one worth taking seriously, because it's the cheapest option and I should be able to justify not taking it. Rule-based layout analysis (projection profiles, connected-component analysis, tools like Tesseract's page segmentation or LayoutParser's heuristics) works well on *clean, digitally-born, single-column documents*. Our inputs were scanned, skewed, multi-format across clients, and the targets weren't text blocks — they were stamps, handwritten annotations and signatures, which have no structural regularity at all. A projection profile will find you a column of text; it will not find you a rubber stamp overlapping a signature. I did evaluate a heuristic baseline and it fell over on exactly the cases we cared about.

**Why not YOLO** — YOLO would have been genuinely faster, and speed wasn't irrelevant. But YOLO (at the time, v3/v4) produced bounding boxes only. Two of our requirements needed pixel masks: extracting a signature *without* the surrounding text for downstream verification, and separating a stamp that physically overlaps a signature. Two overlapping bounding boxes are ambiguous about which pixels belong to which object; two masks are not. That's an instance-segmentation requirement, and boxes cannot satisfy it.

**Why not U-Net** — U-Net gives you semantic segmentation: every pixel labelled 'signature' or 'not signature'. If a page has three signatures, U-Net gives you one signature-coloured blob region and no notion that there are three distinct objects. The downstream workflow counted and extracted regions individually, so instance identity was mandatory.

So: masks required, instance identity required, moderate throughput acceptable. That's Mask R-CNN's exact niche.

**What I'd reconsider today:** if I were rebuilding this in 2026, I'd seriously evaluate a document-specific transformer — LayoutLMv3 or Donut for the understanding task, or DINO/Mask2Former for the detection task. And for the pure region-detection part, YOLOv8's segmentation variant closes most of the mask gap at much higher throughput. The 2020-era argument for Mask R-CNN was strong; it's weaker now, and I'd rather say that than defend a tool choice past its expiry."

---

**Q27: 92% mAP — at which IoU threshold?** *(trick)*

**A:** "IoU 0.5. That's the PASCAL VOC convention, and it's what COCO reports as AP<sub>50</sub>.

Under COCO's stricter primary metric — mAP averaged over IoU 0.50 to 0.95 in 0.05 steps — the same model is around 0.70. A twenty-point gap between AP<sub>50</sub> and AP<sub>[.5:.95]</sub> is completely normal; it's the price of grading boundary tightness rather than just detection.

I lead with the 0.5 number and I'll tell you why rather than making you extract it: the downstream consumer cropped each detected region with a small margin before running OCR, so a box that was slightly loose cost nothing. IoU 0.5 was the threshold that actually corresponded to 'is this region usable'. Spending model capacity on tightening masks from 0.75 to 0.85 IoU would have been optimising a metric that no consumer of the output could perceive.

Where that would flip: if the requirement had been redaction — blacking out signatures for a privacy workflow — then boundary precision is the entire job, a loose mask leaks PII, and I'd have reported AP<sub>[.5:.95]</sub> or even AP<sub>75</sub> as the headline and tuned for it."

---

**Q28: How did you measure the 63% review reduction? What's the denominator?** *(trick)*

**A:** "The denominator is the right thing to ask about, because that's where this kind of number gets inflated.

**Denominator:** document *regions* — not documents, not pages — that previously required human confirmation. Before the model, that was **all of them**. Every region feeding the extraction pipeline was visually confirmed by an operator, so the baseline was 100%.

**Numerator:** regions still routed to a human after deployment, which came to 37%. That breaks down as roughly 22% below the confidence threshold, 13% high-stakes classes that we routed to review *regardless* of confidence, and 2% from a safety net that sent any page with zero detections to review.

So 63% of regions stopped needing human eyes.

Three things I'd volunteer without being asked:

**The high-stakes carve-out makes this conservative, not generous.** Signatures and stamps carry legal weight, and the client's tolerance for a missed signature was zero, so those were never auto-accepted. The 63% comes entirely from the easier classes. A naive threshold sweep would have reported a higher number and been less defensible.

**It's a reduction in review volume, not headcount.** Nobody was cut by 63%; the capacity went to a backlog and to harder documents. I won't claim a cost saving I didn't measure.

**It was measured on a matched sample** — same client mix, same document-type mix, same volume band, month over month — specifically so the comparison wasn't confounded by an easier batch of documents.

And the safeguard that makes it a measurement rather than a claim: a random sample of the *auto-accepted* regions was periodically re-reviewed by a human, to confirm the auto-accept error rate stayed inside the agreed tolerance. Without that audit loop, 63% would just be a threshold setting with a percentage attached."

---

**Q29: What did the model actually fail on?** *(trick)*

**A:** "Six things, and the interesting part is that two of them weren't model problems.

**Handwriting boundaries** were the biggest single contributor to the AP gap — handwriting sat at 0.83 AP<sub>50</sub> against 0.96 for headers. But when I inspected the failures, most were detections that *found* the handwriting and landed at IoU 0.4 to 0.6. The underlying cause was that annotators genuinely disagreed on where a handwritten note ends — tight around the ink, or the whole annotated region? That's a labelling-guideline problem wearing a model-performance costume, and I fixed it by rewriting the guideline with worked examples rather than by changing the architecture.

**Signature versus handwriting confusion.** These two classes are visually continuous, not discrete — a signature *is* handwriting with a particular function. I never fully solved it, and I'd argue the class taxonomy was the problem. It was largely harmless in practice because downstream treated both as 'needs human eyes'.

**Overlapping stamps and signatures** producing one merged detection instead of two. This is the hardest case for NMS: a high-IoU pair of genuinely *different* objects is indistinguishable from a duplicate detection of one object. Lowering the NMS threshold helped; Soft-NMS helped more. Somewhat ironically, this is the exact scenario I chose Mask R-CNN *for*, and it remained the hardest case.

**Multi-column and landscape pages** — wide tables split into two detections. Root cause is anchor aspect ratios: the defaults of 0.5, 1 and 2 don't cover very wide, flat objects. The fix I identified but never shipped was k-means on the ground-truth box dimensions to derive document-specific anchors.

**Pre-printed letterhead logos detected as stamps.** The single most common false positive — a coloured, roughly circular graphic in a page corner is genuinely stamp-like. A dedicated 'logo' negative class would have fixed it; I ran out of annotation budget.

**Very low-quality scans** — fax and third-generation photocopies — where confidence collapsed across all classes. Noise and JPEG-artefact augmentation narrowed the gap but didn't close it.

The systematic point underneath all of this: I tuned the confidence threshold toward **recall** rather than precision, at 0.7 rather than 0.9, because in a document pipeline a false positive costs a wasted human review while a false negative means information silently never reaches OCR. The second failure is much more expensive and much harder to detect."

---

**Q30: Explain RoIAlign and why RoIPool wasn't good enough for masks.**

**A:** "The problem is quantisation, and it happens twice.

RoIPool takes a region proposal in image coordinates and maps it onto the feature map, which is downsampled — with a stride of 16, an x-coordinate of 3.75 in feature-map space gets rounded to 4. That's the first quantisation. Then it divides the RoI into a fixed grid, say 7×7, and each bin boundary gets rounded again. That's the second.

At stride 16, half a pixel of feature-map error is **eight pixels in the original image**. For classification that barely matters — you're pooling over a region and the semantic content survives a small shift. For a mask you're predicting a 28×28 binary grid that gets resized back onto the object, and an eight-pixel systematic offset visibly ruins the boundary.

RoIAlign removes both roundings. It keeps the RoI coordinates in floating point, divides the grid at exact fractional positions, samples four points inside each bin, and computes each sample by **bilinear interpolation** of the four nearest feature-map cells. Then it pools those four samples. No rounding anywhere.

The bilinear interpolation is also the reason it's differentiable with respect to the RoI coordinates, which matters for end-to-end training.

The paper reports roughly a 3-point mask AP improvement, and — the detail that shows you actually read it — **the gain is much larger under strict IoU thresholds than at IoU 0.5**, which is exactly what you'd predict if the mechanism is localisation precision rather than detection. That asymmetry is the evidence that the explanation is right."

---

**Q31: You froze the backbone in phase 1 and unfroze it in phase 2. Why not just train end-to-end from the start?** *(trick)*

**A:** "Because the randomly-initialised heads would destroy the pretrained backbone before they became useful.

Here's the mechanism. On the first forward pass the classification, box and mask heads are random, so their outputs are garbage and their loss is enormous. That loss backpropagates through the FPN and into the ResNet, and the resulting gradients are large enough to overwrite the COCO features. With a few hundred training images, those features are the only reason the model works at all — there isn't enough data to relearn edge and texture detectors from scratch. That's catastrophic forgetting, and on a small dataset it's unrecoverable.

Freezing the backbone in phase 1 lets the heads reach a sensible state while the features stay intact. Only then is it safe to let gradients through, and even then at a tenth of the learning rate, so the backbone *adapts* rather than gets *rewritten*.

Three related details worth adding:

I kept **C1 and C2 frozen permanently**, not just in phase 1. Those layers learn edges, corners and stroke texture, which are genuinely universal — documents need exactly the same low-level features as natural images. Unfreezing them adds parameters and variance for no benefit.

I trained the **RPN from the start**, not just the heads. Document objects are wide and flat — headers, tables — whereas COCO's anchor priors assume roughly square objects. The RPN needed adaptation immediately.

And **BatchNorm stayed frozen throughout**. With two images per GPU, batch statistics computed over a batch of two are noise. Every Mask R-CNN implementation freezes BN by default for that reason, and using a warm-up schedule with unfrozen BN at that batch size is a reliable way to get an unstable, non-reproducible training run.

An alternative I'd consider today: gradual unfreezing with discriminative learning rates — a different LR per layer group, higher toward the head — which is a smoother version of the same idea and tends to work slightly better than two hard phases."

---

**Q32: How do you know your train/test split didn't leak?** *(trick)*

**A:** "The specific leak that matters in document CV is **page-level splitting of multi-page documents**, and it's easy to do by accident.

If a 40-page contract is exploded into 40 images and you split randomly, pages 1–30 land in train and pages 31–40 in test — but they share the same letterhead, the same stamp design, the same signatory, the same scanner, the same paper. The model can recognise the *document* rather than the *class*, and your test mAP is inflated by an amount you can't easily bound.

So the split was made at the **source-document level**: every page of a given document lands entirely in train, or entirely in test. Never split.

Two further precautions, since document data has more than one identity axis:

**Client-level awareness.** Documents from the same client share templates. I kept a held-out client whose documents appeared nowhere in training, which is the closest thing to an honest generalisation test — it answers 'will this work for the next customer we onboard', which is the question the business actually had.

**Augmented-copy discipline.** Augmentation happens *after* the split, inside the training loader. Generating an augmented set and then splitting it would put a rotated copy of a training image into the test set, which is the same leak in a different coat.

What I'd add if I were doing it again: a near-duplicate check across the split boundary. Clients resubmit the same document, and two copies of one contract sitting on opposite sides of the split is a leak that document-level splitting won't catch, because they have different document IDs."

---

**Q33: Why does FPN help here? Isn't a deeper backbone enough?** *(trick)*

**A:** "They solve different problems, and conflating them is the trap in the question.

Depth buys **semantic strength**. ResNet-101's deepest stage, C5, has excellent semantic features — but it's downsampled 32×, so a 1024×1024 page becomes a 32×32 feature map. A stamp occupying 3% of the page is roughly two or three cells there. You cannot localise, let alone mask, an object represented by two feature-map cells, no matter how semantically rich those cells are.

The shallow stages have the opposite problem: C2 is at 1/4 resolution, so spatial detail is fine, but the features are edges and textures with no object-level semantics.

FPN resolves the trade-off rather than picking a side. The top-down pathway carries C5's semantics back down to higher-resolution levels, and the lateral 1×1 convolutions inject the spatially-precise information from C2, C3 and C4 at each level. The result is that P2 through P5 all carry **strong semantics at their native resolution**. Then each RoI is assigned to the pyramid level matching its scale — small objects to P2, large ones to P5.

Concretely for our data: table regions could occupy 60% of a page while stamps occupied 3%. That's roughly a 20× linear scale range within a single image, and often within the *same* image. Without FPN I'd have to pick a resolution that compromises one end. Every failure would be a small-object failure.

**The counter-argument you should expect and be ready for:** you could instead train at higher input resolution, or use dilated/atrous convolutions to keep resolution in the deep stages. Both work. Both are dramatically more expensive in memory and compute than FPN's few extra 1×1 and 3×3 convolutions. FPN is the cheap solution, which is why it became the default rather than because it's the only one."

---

**Q34: What's actually in your Docker image, and why isn't the model in it?**

**A:** "The image holds code and environment; the weights live in Azure Blob and get pulled at startup by version tag. That separation is deliberate and it's the design decision I'd defend hardest.

**What's in the image:** a pinned CUDA/cuDNN runtime base, Python, a fully pinned and hash-checked `requirements.txt`, the source, and config. Multi-stage build, so the compiler toolchain from the `devel` base doesn't ship in the runtime image — that's several gigabytes of difference, which matters when the image is being pulled onto autoscaled inference nodes.

**Why the weights aren't baked in:**

A code fix — say a preprocessing bug — shouldn't require rebuilding and re-pushing a 250 MB weights layer. Conversely, a model refresh shouldn't require rebuilding code that didn't change. Decoupling means code releases and model releases ship independently.

Rollback becomes an environment-variable change rather than an image rebuild, which turns 'revert the model' from a twenty-minute pipeline run into a restart.

And one image can serve any model version, which is what makes A/B or shadow evaluation practical.

**The cost, honestly:** a startup dependency on Blob availability, and a slow cold start on a fresh node. Mitigated with a cache volume and a generous healthcheck `start-period`, but it's a real trade-off, not a free win.

**The reason containerising mattered at all here** — beyond deployment convenience — is that a model is not a weights file. It's weights *plus* preprocessing *plus* a specific CUDA/cuDNN/TensorFlow combination. Before containers, training ran on a hand-configured GPU VM and inference ran elsewhere, and version skew produced *different predictions from the same weights*. That's the worst class of bug, because nothing throws an error. Pinning the environment into an image that both training and inference use is what eliminated it."

---

**Q35: How do you test an ML pipeline in CI when the output isn't deterministic-by-inspection?**

**A:** "You can't assert 'the model is correct', so you assert three narrower things at three different scopes.

**Layer one — unit tests on the deterministic parts.** Image resizing preserves aspect ratio. Polygon-to-mask conversion produces the right pixel count. NMS removes the right boxes given synthetic input. The confidence-routing logic sends the right region to the right queue. The COCO annotation parser handles malformed input without crashing. These need no GPU, no weights, and run in seconds — and they're where most *actual* defects live, because they're ordinary software.

**Layer two — contract tests on the output schema.** The inference output has the required keys, confidence is in [0,1], mask dimensions match the box dimensions, class IDs are in range. This protects every downstream consumer from a silent format change, which is a whole category of production incident.

**Layer three — the golden-set regression gate, which is the one that matters.** The *built image*, running *real weights*, is run against a frozen set of thirty images with known expected metrics, and the build fails if mAP moves more than a point in either direction.

That third layer exists because of a specific failure mode: a refactor of the preprocessing — someone changes a normalisation constant or an interpolation mode — will pass every unit test and every schema check, and quietly cost you five points of mAP. Nothing else catches that.

Two details about the golden set that make it work. It's built from the **hard cases**, deliberately — a multi-column page, a fax-quality scan, an overlapping stamp and signature. A regression suite made of easy examples never fires. And the tolerance is **two-sided**: a sudden *improvement* is as suspicious as a regression, because it usually means the evaluation data or the matching logic changed rather than the model getting better.

**What I'd add today:** data-validation tests on the training set itself, in the Great Expectations style — assert the class distribution hasn't shifted, assert no corrupt images, assert annotations parse — so a bad *dataset* fails the build rather than silently producing a worse model."

---

### Trick & Follow-Up Questions

---

**T1: "You used the Matterport implementation. So what did you actually build?"**

> "Fair, and I'd rather answer it directly than get defensive. I did not implement Mask R-CNN from scratch — I used the Matterport TensorFlow/Keras implementation as the base, which is what essentially everyone did in 2020.
>
> What I built on top: the custom `Dataset` subclass handling our COCO-format annotations and polygon-to-mask conversion; the anchor and configuration tuning for document-shaped objects; the two-phase freeze/unfreeze training schedule; the document-specific augmentation pipeline, including the synthetic stamp and signature compositing that was the single biggest win for the rare classes; the evaluation harness producing per-class AP at multiple IoU thresholds; the confidence-based human-in-the-loop routing that produced the 63%; the containerisation; and the CI pipeline with the golden-set gate.
>
> And I understand the architecture rather than just calling it — I can explain why RoIAlign's gain is concentrated at strict IoU thresholds, why the mask head predicts K binary masks rather than one K-way mask, why the RPN's 1:1 anchor sampling is what lets a two-stage detector skip focal loss, and why BatchNorm is frozen at batch size 2.
>
> The honest framing: using a well-tested reference implementation and spending the time on the domain problem was the right engineering decision. Reimplementing the architecture would have consumed the project's whole timeline and produced something worse."

---

**T2: "Your per-class AP ranges from 0.83 to 0.96. Isn't reporting the mean misleading?"**

> "Somewhat, yes — which is exactly why the per-class table exists in my documentation and why I never reported the mean alone internally.
>
> The mean is the right *summary* number for a resume line and for comparing model versions against each other, because you need one number to rank candidates. It's the wrong number for deciding whether the system is fit for purpose, because the classes weren't equally important. Signatures and stamps had legal significance; headers were nice to have. A mean weights them by nothing at all.
>
> What we actually operated on: per-class thresholds and per-class routing. Signatures went to human review regardless of confidence *because* their AP was lower and their cost of error was higher. So the operational system did account for the spread, even though the headline number doesn't show it.
>
> If I were reporting this to a technical stakeholder rather than on a resume, I'd lead with the per-class table and the weakest class, not the mean."

---

**T3: "A few hundred training images for a 100-plus-million-parameter model. Isn't that absurd?"**

> "It would be absurd if I were training from scratch. I wasn't — and that distinction is the whole answer.
>
> The COCO-pretrained backbone already encodes edges, textures, shapes and object-ness from 330,000 images. What I needed to learn was 'which of those patterns correspond to a stamp' — a much smaller problem in terms of effective parameters. In phase 1 the trainable parameter count is a small fraction of the total, because the backbone is frozen.
>
> Four things kept the variance manageable: transfer learning as the foundation; heavy but *domain-appropriate* augmentation, which multiplied the effective dataset several-fold; the frozen-then-unfrozen schedule so the small dataset never had to support learning general features; and early stopping on validation mAP rather than training to a fixed epoch count.
>
> But I won't over-claim. The evidence that data was still the binding constraint is right there in the per-class results: handwriting, the class with the fewest and most variable examples, is the weakest at 0.83, and the gap to the other classes tracks annotation count almost exactly. When I was asked how to improve the model, my first answer was 'more annotated data for the rare classes', not 'a better architecture'. That's what the error analysis pointed at."

---

**T4: "Azure Blob for model storage in 2021? Why not Azure ML, or MLflow?"**

> "It was the pragmatic call, and I'll give you both the justification and where it falls short.
>
> The justification: the client's infrastructure was already Blob-centric and provisioned, our versioning requirement was genuinely simple — a version tag and a metadata record — and Azure ML would have added a service, a cost line and an approval cycle to solve a problem we didn't have yet. For a handful of model versions on one project, Blob plus a naming convention plus metadata tags was proportionate.
>
> Where it falls short, and I'd say this unprompted: there's no lineage. Blob tells you *that* `v3` exists; it doesn't tell you which dataset, which commit, which hyperparameters or which evaluation produced it. We tracked that in a spreadsheet and in the CI build record, which works until it doesn't.
>
> What I'd build today: MLflow or Azure ML for the registry — experiment tracking, model versions with stages, lineage back to data and code — with Blob still underneath as the artefact store, because that part was never the problem. I did exactly this at Fibe afterwards, which is partly because I'd felt the gap here."

---

**T5: "Detection at 1024×1024 with ResNet-101 is slow. Did throughput ever become a problem?"**

> "Yes, and it shaped the architecture of the serving layer more than the model.
>
> Mask R-CNN with a ResNet-101-FPN backbone at that input size runs in the low hundreds of milliseconds per page on a single GPU, and multiple seconds on CPU. For a client processing thousands of pages in a batch window, that's a real constraint.
>
> Three things addressed it. **Batching** — pages were scored in batches rather than one request at a time, which is where most of the GPU utilisation gain came from. **Keeping the model resident** — loading Mask R-CNN weights takes tens of seconds, so reloading per request would have dominated the runtime entirely; that's why the container's healthcheck has a long start-period. And **asynchronous processing** — the whole thing ran as a batch job writing results back to Blob, rather than pretending it was a synchronous request-response API, which set the right latency expectation with consumers.
>
> **What I'd change:** ResNet-50 instead of ResNet-101 was the obvious lever. My experiments showed ResNet-101 was worth about two points of mAP over ResNet-50 — and if I'm honest about the cost-benefit at the operating point we actually used, two points of AP<sub>50</sub> for roughly 40% more inference cost is a trade I'd want the client to make explicitly rather than assume. I'd also look at mixed-precision inference and ONNX Runtime, neither of which I tried."

---

**T6: "You said recall matters more than precision. But false positives waste reviewer time, and reviewer time is the thing you're claiming to save. Isn't that contradictory?"**

> "It's a genuine tension and I don't think it's a contradiction, but the reason is worth spelling out.
>
> The asymmetry is in the *detectability* of the two errors. A false positive lands in the review queue, a human looks at it, rejects it in a couple of seconds, and moves on. It's visible, bounded and cheap. A false negative means a region was never detected — so it never entered the queue, no human ever saw it, and the information silently never reached OCR. Nobody finds out until a downstream process needs that field and it isn't there, possibly weeks later.
>
> So the costs aren't symmetric even though both consume attention: one costs seconds of visible time, the other costs an undetected data loss.
>
> The reconciliation with the 63% claim is that these operate at different points. The confidence threshold at 0.7 governs *what the model emits*, biased toward recall. The auto-accept threshold — a separate, higher threshold — governs *what skips review*. Between them sits a band of medium-confidence detections that get reviewed. Extra false positives from the low emission threshold land in the reviewed band, not in the auto-accepted band, so they don't corrupt the 63%; they slightly enlarge the 37%.
>
> And the zero-detection safety net exists for precisely the false-negative case the emission threshold can't fix: if a page produced no detections at all, the whole page went to a human regardless."

---

**T7: "Your CI pipeline has a golden-set gate with a 1-point tolerance. Where did 1 point come from?"**

> "From the run-to-run variance, measured rather than guessed — though I'll admit it was measured roughly.
>
> Inference on a fixed image set with fixed weights should be deterministic, but in practice it isn't quite: non-deterministic GPU kernel selection, cuDNN autotuning picking different algorithms, and floating-point non-associativity in reduction operations all introduce small variation. I ran the same evaluation repeatedly and the spread in mAP was well under a point on a thirty-image set. So one point is a threshold comfortably above the noise floor and comfortably below any change I'd care about.
>
> The weakness, honestly: thirty images is a small evaluation set, so the metric itself is noisy in a way that scales with set size. A larger golden set would let me tighten the tolerance, at the cost of a slower build. The right answer is probably a two-tier setup — a fast thirty-image gate on every build, and a larger nightly evaluation with a tighter tolerance.
>
> The tolerance is also **two-sided**, which people find counterintuitive. A sudden *improvement* fails the build too, because in my experience an unexplained jump usually means the evaluation data changed or the matching logic broke, not that the model spontaneously got better."

---

**T8: "If the 92% and the 63% both came from your work, which one would you drop from the resume and why?"**

> "I'd keep the 63% and drop the 92%, and I think that's the less obvious answer.
>
> The mAP number is a *model* metric. It requires context to interpret — which IoU threshold, how many classes, how hard is the domain — and without that context it's either meaningless or misleading, which is why I've written a whole section clarifying it. It also invites a comparison to COCO benchmarks that isn't valid.
>
> The 63% is an *outcome* metric. It says the work changed what a business does: two-thirds of a manual review step stopped happening. It needs no CV background to evaluate, it can't be inflated by choosing a favourable threshold on a benchmark, and it's the thing a hiring manager is actually trying to infer from the mAP anyway.
>
> The reason both are on the resume is that they serve different readers — a technical screener wants to know I can train a detector, a hiring manager wants to know it mattered. But if I could only defend one in a room, it's the 63%, because it's the one whose measurement I can walk through end to end including the caveats."

---

## 7. Red Flags & How to Handle

### Red Flag 1: "PySpark for small data?"

**What they're probing:** Did you use PySpark where Pandas would have sufficed?

**How to handle:** "We started with Pandas for prototyping, but daily data volumes quickly exceeded single-node memory. Even with chunking, Pandas required manual parallelism and couldn't benefit from Catalyst optimization. PySpark was a forward-looking choice — we knew data would grow, and having a distributed pipeline from the start avoided a painful migration later."

---

### Red Flag 2: "92% mAP sounds high for limited data."

**What they're probing:** Is this number inflated by overfitting, test-set leakage, a too-easy task, or a favourably-chosen metric?

**How to handle:** "Let me answer the metric question first, because it's the biggest part of the gap. 92% is mAP at IoU 0.5. Under COCO's stricter 0.5-to-0.95 averaging the same model is around 0.70 — I lead with the IoU 0.5 number because downstream cropped regions with a margin, so boundary tightness wasn't the requirement, but I'll always tell you both.

On why it's legitimately high for this task: five visually-distinct classes on deskewed white pages is a fundamentally easier problem than 80 confusable classes in unconstrained natural scenes. A strong COCO model gets 65–70% AP<sub>50</sub> on that harder problem, so 92% on ours isn't out of line.

On leakage and overfitting specifically: the test split was made **at the document level, not the page level** — pages from the same source document never straddled the train/test boundary, which is the mistake that would have inflated this most. I monitored train/validation loss divergence per head, and the honest weak spot shows in the per-class table: handwriting sits at 0.83 while headers are at 0.96. A model that had memorised the test set wouldn't show that spread."

---

### Red Flag 3: "Why not just use a pre-built document AI service?"

**What they're probing:** Cost-benefit analysis, build vs. buy.

**How to handle:** "We evaluated Azure Form Recognizer and AWS Textract. Two issues: first, these services offered general-purpose document extraction but didn't have our specific class taxonomy (stamps, signatures, handwriting regions in the specific document types our clients handled). Second, the per-page pricing at our volume was significantly more expensive than running our own model on an Azure VM. Building custom gave us full control over the detection classes, threshold tuning, and model iteration cycle."

---

### Red Flag 4: "Did you write the Mask R-CNN implementation yourself?"

**What they're probing:** Do you understand the architecture or just call an API?

**How to handle:** "I used the Matterport Mask R-CNN implementation as a base, which implements the architecture in TensorFlow/Keras. My work focused on adapting it to our domain — creating the custom dataset class, tuning anchor configurations, implementing the two-phase training strategy, adding document-specific augmentations, and building the deployment pipeline. I have a deep understanding of the architecture — I can explain ROI Align, FPN, the multi-task loss, and why each design choice matters."

---

### Red Flag 5: "How did you validate the NLP model on legal documents when labels might be subjective?"

**What they're probing:** Annotation quality, evaluation rigor.

**How to handle:** "Subjectivity in legal document labels was a real challenge. We addressed it three ways: First, we created detailed labeling guidelines with examples for each category, especially for borderline cases. Second, we had multiple annotators label a subset of documents and computed Cohen's Kappa to measure inter-annotator agreement — only categories with substantial agreement (κ > 0.61) were kept. Third, I built a validation pipeline that flagged low-confidence predictions for human review, creating an active learning feedback loop that improved both the model and the annotation guidelines over time."

---

### Red Flag 6: "What about MSSQL vs. a modern cloud data warehouse?"

**What they're probing:** Technology choice awareness.

**How to handle:** "MSSQL was the client's existing infrastructure, and switching to Snowflake or BigQuery wasn't in scope. Within that constraint, we maximized MSSQL's capabilities — using columnstore indexes for analytical queries, table partitioning for efficient date-range queries, and the MERGE statement for efficient upserts. If I were designing this today with no constraints, I'd likely choose a modern cloud warehouse for the analytics layer, but the PySpark processing and data quality patterns would be the same."

---

### Red Flag 7: "Isn't FPN overkill for document detection?"

**What they're probing:** Do you understand why FPN is needed vs. just using a deeper backbone?

**How to handle:** "FPN was essential because document elements span a wide range of scales. A table might occupy 60% of the page, while a small stamp might be 5%. Without FPN, detection at a single scale would either miss small elements (if using high-level features) or fail to understand context for large elements (if using low-level features). FPN gives us strong features at every scale, which was critical for our ~10x range of object sizes."

---

## 8. Key Takeaways & Talking Points

### For Behavioral Questions

| Theme | Talking Point |
|-------|--------------|
| **End-to-end ownership** | "At ATCS, I owned three interconnected workstreams — from ETL pipeline design to computer vision model training to NLP classification deployment. This taught me to think in systems, not just models." |
| **Production mindset** | "The data quality validation framework was as important as any ML model. In production, bad data causes more damage than a suboptimal model." |
| **Transfer learning expertise** | "I applied transfer learning in two very different domains — CV (COCO → documents with Mask R-CNN) and NLP (general text → legal domain with BERT). The principles are the same: pretrain on large general data, fine-tune on domain-specific data, manage the learning rate carefully." |
| **Scale thinking** | "I chose PySpark from the start because I anticipated data growth. Forward-looking infrastructure decisions saved us from a painful migration later." |

### Technical Differentiators

| What Makes This Stand Out | Why |
|---------------------------|-----|
| Mask R-CNN is rarely discussed in interviews | Shows deep CV knowledge beyond simple classification — RoIAlign, FPN scale assignment, the decoupled mask loss |
| Being precise about *which* mAP | 92% @ IoU 0.5 vs ~0.70 @ [.5:.95], volunteered rather than extracted — signals metric literacy |
| A business metric you can actually derive | The 63% has a stated denominator, a routing rule, and an audit loop |
| PySpark + config-driven data quality | Not just "model.fit()" — real pipeline thinking, reused across 4+ engagements |
| Transfer learning across both CV and NLP | Versatility and understanding of the common principle |
| Docker + CI/CD with a golden-set regression gate | The test that catches a silent preprocessing regression — most people don't have this |
| Azure deployment with weights decoupled from the image | Cloud deployment experience with a defensible design decision, not just notebook experiments |
| Multi-project narrative | Shows breadth and ability to work across the stack |

### 30-Second Elevator Pitch

"At ATCS, I was an Associate Data Scientist building a document intelligence platform. I engineered distributed PySpark ETL pipelines processing 10M-plus multi-source records, cutting the production batch from four hours to eighteen minutes, with a configuration-driven data quality framework — validation, profiling, referential integrity — that was reused across four-plus enterprise engagements. On the modelling side, I fine-tuned a Mask R-CNN with a ResNet-101-FPN backbone from COCO weights onto five document-region classes, hit 92% mAP, and deployed it on Azure — and because confident detections stopped going to a human, manual document-region review dropped 63%. I containerised both the Python and ML workloads with Docker and automated testing and deployment through CI/CD, including a golden-set regression gate, so model releases were standardised rather than hand-rolled. I also built an NLP classification system using Legal-BERT for automated legal document tagging with labeling accuracy validation."

### Questions to Ask the Interviewer (Show Depth)

1. "What does your data quality framework look like? Are validation rules defined declaratively or programmatically?"
2. "Are you using instance segmentation or just bounding box detection for your document processing pipeline?"
3. "How do you handle model versioning and rollback in production? Azure ML Registry, MLflow, or custom?"
4. "What's your approach to handling class imbalance in production NLP classifiers — weighted loss, oversampling, or something else?"

---

*Document prepared for Rahul Sharma — ATCS (Advanced Technology Consulting Service) interview preparation. Covers PySpark ETL, Mask R-CNN, and Legal Document Classification.*
