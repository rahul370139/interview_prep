# PROJECT: LLM-Based Automated Call Transcript Classification — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Role:** Data Scientist @ Jet2 and Jet2 Holidays, Leeds, UK (June 2022 – Aug 2024)
**Primary Resume Line:** *"Engineered multi-label intent classification across 1M+ call transcripts by fine-tuning LLMs and distilling a 70B teacher into a 7B model, achieving 94% recall on validated intent labels."*
**Supporting Resume Line:** *"Built transcript summarization pipelines using Snowflake Cortex and few-shot prompting, standardizing customer-conversation summaries across contact-center teams."*

---

## Table of Contents

0. [Resume Bullet ↔ Proof](#0-resume-bullet--proof)
1. [Project Overview (STAR)](#1-project-overview-star)
2. [Deep Technical Walkthrough](#2-deep-technical-walkthrough)
   - 2.8 [Knowledge Distillation Deep Dive (70B → 7B)](#28-knowledge-distillation-deep-dive-70b--7b)
3. [Key Metrics & Results](#3-key-metrics--results)
4. [Topics You Must Know](#4-topics-you-must-know)
5. [Interview Questions (30+) with Model Answers](#5-interview-questions-30-with-model-answers)
   - [Trick & Follow-Up Questions](#trick--follow-up-questions)
6. [Potential Red Flags & How to Handle](#6-potential-red-flags--how-to-handle)
7. [Key Takeaways](#7-key-takeaways)

**Companion learning notes:** [`../learning/08_knowledge_distillation.md`](../learning/08_knowledge_distillation.md) · [`../learning/07_fine_tuning_and_peft.md`](../learning/07_fine_tuning_and_peft.md) · [`../learning/14_evaluation_metrics.md`](../learning/14_evaluation_metrics.md)

---

# 0. Resume Bullet ↔ Proof

Use this as the pre-interview cheat sheet. Every number on the resume maps to a section of this doc and a one-sentence spoken answer.

| Resume Bullet | Exact Metric | Where the Proof Lives | 1-Sentence Spoken Answer |
|---|---|---|---|
| "Engineered multi-label intent classification across **1M+ call transcripts**" | 1,000,000+ transcripts processed; 2,000–5,000/day steady state | §1.4 Scale & Scope, §2.1 Data Pipeline | "We ran the full historic backlog of over a million transcripts through the pipeline and then kept it running on two to five thousand new calls a day." |
| "…by **fine-tuning LLMs**…" | LoRA/QLoRA supervised fine-tuning of a 7B student on ~40K curated examples | §2.5, §2.8.4 Student Training Setup | "I fine-tuned a 7B model with LoRA on teacher-generated labels rather than prompting a big model on every call." |
| "…and **distilling a 70B teacher into a 7B model**" | 70B → 7B, ~10x latency reduction, ~10–15x cost reduction | §2.8 Knowledge Distillation Deep Dive | "The 70B model was the labeller, not the product — I distilled its behaviour into a 7B student that runs the daily batch." |
| "…achieving **94% recall on validated intent labels**" | 94% recall (over the SME-validated label set) on the human-verified hold-out | §3.1 Performance Summary, §2.7 Metric Choice | "On the SME-validated hold-out set the student recovers 94% of the true intent labels, and recall is the metric that matters because a missed intent is a missed action." |
| "Built transcript summarization pipelines using **Snowflake Cortex** and **few-shot prompting**" | `CORTEX.SUMMARIZE` / `CORTEX.COMPLETE` over 1M+ transcripts; 5–8 shot prompts | §2.3 Snowflake Cortex, §2.4 Few-Shot Learning | "Summarization ran natively in Snowflake with Cortex so PII never left the warehouse, and few-shot prompts kept the summary format identical for every contact-centre team." |

> **If asked for the single number:** "94% recall on validated intent labels — that's the student model, on held-out transcripts labelled by our subject-matter experts, not just agreement with the teacher."

---

# 1. Project Overview (STAR)

---

## 1.1 STAR Summary

| STAR Element | Description |
|---|---|
| **Situation** | Jet2's customer service team handled millions of calls annually. Transcripts piled up with no automated way to extract insights or classify call intent reliably. Manual review was slow, expensive, and inconsistent — agents labelled the same call differently. Leadership needed structured data from unstructured conversations to drive operational decisions. |
| **Task** | Build an end-to-end system to automatically **summarize** and **classify** 1M+ call transcripts into a hierarchical, **multi-label** intent taxonomy (primary, secondary, tertiary intent), replacing manual review entirely. |
| **Approach** | (1) Collaborated with product/engineering to define and consolidate hundreds of raw intents into actionable groups. (2) Used Snowflake Cortex built-in functions for transcript summarization and normalization. (3) Implemented few-shot LLM classification with iterative prompt refinement based on SME feedback. (4) Applied knowledge distillation — 70B teacher → 7B student — for cost-effective production inference. (5) Automated the data pipeline with drift monitoring and operational dashboards. |
| **Result** | **94% recall on validated intent labels** for multi-label intent classification. Dramatically cut manual review time. Integrated into daily operational dashboards enabling real-time decision-making. Earned internal recognition as a key AI innovation at Jet2. |

---

## 1.2 End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                  CALL TRANSCRIPT CLASSIFICATION — SYSTEM ARCHITECTURE             │
│                                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐     │
│  │  CALL CENTER  │    │  TELEPHONY / IVR  │    │  ASR (Speech-to-Text)      │     │
│  │  Agent + Cust │───►│  System           │───►│  Transcription Engine      │     │
│  │  Live Calls   │    │  (Recording)      │    │  Raw Text Output           │     │
│  └──────────────┘    └──────────────────┘    └────────────┬────────────────┘     │
│                                                           │                      │
│                                                           ▼                      │
│  ┌────────────────────────────────────────────────────────────────────────┐      │
│  │                        SNOWFLAKE DATA PLATFORM                         │      │
│  │                                                                        │      │
│  │  ┌──────────────────┐   ┌──────────────────────────────────────┐      │      │
│  │  │  RAW TRANSCRIPTS  │   │  SNOWFLAKE CORTEX (Built-in LLM)    │      │      │
│  │  │  Staging Table    │──►│                                      │      │      │
│  │  │  1M+ records      │   │  CORTEX.SUMMARIZE() → Summaries     │      │      │
│  │  │  Partitioned by   │   │  CORTEX.COMPLETE()  → Classification│      │      │
│  │  │  date             │   │  CORTEX.SENTIMENT() → Sentiment     │      │      │
│  │  └──────────────────┘   └──────────────┬───────────────────────┘      │      │
│  │                                         │                              │      │
│  └─────────────────────────────────────────┼──────────────────────────────┘      │
│                                             │                                     │
│                                             ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐      │
│  │                     CLASSIFICATION PIPELINE                            │      │
│  │                                                                        │      │
│  │  Phase 1: TEACHER MODEL (70B LLM)                                     │      │
│  │  ┌────────────────────────────────────────────────────┐               │      │
│  │  │  • Few-shot prompting with curated examples        │               │      │
│  │  │  • Multi-intent classification (primary/sec/tert)  │               │      │
│  │  │  • Generates soft labels + confidence scores       │               │      │
│  │  │  • Iterative prompt refinement with SME feedback   │               │      │
│  │  └──────────────────────┬─────────────────────────────┘               │      │
│  │                          │ Soft labels / distillation data             │      │
│  │                          ▼                                             │      │
│  │  Phase 2: STUDENT MODEL (7B LLM)                                      │      │
│  │  ┌────────────────────────────────────────────────────┐               │      │
│  │  │  • Fine-tuned on teacher's outputs                 │               │      │
│  │  │  • Supervised fine-tuning (SFT) on labeled data    │               │      │
│  │  │  • 10x lower latency, 10x lower cost              │               │      │
│  │  │  • Validated on SME labels: 94% recall            │               │      │
│  │  └──────────────────────┬─────────────────────────────┘               │      │
│  │                          │                                             │      │
│  └──────────────────────────┼─────────────────────────────────────────────┘      │
│                              │                                                    │
│                              ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐      │
│  │                   PRODUCTION & MONITORING                              │      │
│  │                                                                        │      │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐      │      │
│  │  │  Classified   │  │  Drift Monitor   │  │  Operational       │      │      │
│  │  │  Transcripts  │  │  (Label dist.,   │  │  Dashboards        │      │      │
│  │  │  Table        │  │  embedding drift, │  │  (Tableau/BI)     │      │      │
│  │  │              │  │  confidence drop) │  │                    │      │      │
│  │  └──────────────┘  └──────────────────┘  └────────────────────┘      │      │
│  │                                                                        │      │
│  └────────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Platform** | Snowflake | Data warehouse, storage, compute |
| **Summarization** | Snowflake Cortex (`SUMMARIZE`, `COMPLETE`) | LLM-powered transcript summarization |
| **Teacher LLM** | 70B parameter model (Llama 2 70B / Mixtral class) | High-quality few-shot classification |
| **Student LLM** | 7B parameter model (Llama 2 7B / Mistral 7B class) | Production-grade fast inference |
| **Fine-tuning** | SFT (Supervised Fine-Tuning), LoRA/QLoRA | Efficient model adaptation |
| **Orchestration** | Snowflake Tasks, Stored Procedures | Automated daily pipeline |
| **Monitoring** | Custom drift detection, label distribution tracking | Model quality assurance |
| **Dashboards** | Tableau / Snowflake Dashboards | Operational reporting |
| **Languages** | Python, SQL | Implementation |

---

## 1.4 Scale & Scope

| Dimension | Value |
|---|---|
| Total transcripts processed | **1,000,000+** |
| Daily new transcripts | ~2,000–5,000 |
| Number of raw intent categories (before consolidation) | 200+ |
| Number of consolidated intent groups | ~30–50 (hierarchical: primary → secondary → tertiary) |
| Teacher model size | **70B parameters** |
| Student model size | **7B parameters** (~10x smaller) |
| Classification type | **Multi-label** (primary + secondary + tertiary intent, multiple intents per call) |
| Headline metric | **94% recall on validated intent labels** |
| How recall is computed | Per-sample label-set recall (Jaccard-style), micro-averaged across the hold-out set |
| Per-label decision rule | Sigmoid per label + per-label threshold (see §2.8.7) |
| Latency improvement (student vs teacher) | ~**8–10x faster** |
| Cost reduction | ~**10–15x cheaper** per inference |

---

# 2. Deep Technical Walkthrough

---

## 2.1 Data Pipeline: Raw Transcripts → Classification

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE FLOW                           │
│                                                                      │
│  Step 1: INGESTION                                                   │
│  ┌────────────────────────────────────────────────────┐             │
│  │  Raw call recordings → ASR transcription            │             │
│  │  Output: { call_id, timestamp, agent_id,            │             │
│  │           raw_transcript, duration, metadata }      │             │
│  │  Loaded into Snowflake staging table (daily batch)  │             │
│  └────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│  Step 2: CLEANING & NORMALIZATION                                    │
│  ┌────────────────────┴───────────────────────────────┐             │
│  │  • Remove PII (names, card numbers, booking refs)   │             │
│  │  • Normalize agent/customer speaker labels           │             │
│  │  • Handle ASR errors, filler words, truncation       │             │
│  │  • Filter very short calls (< 30 seconds)            │             │
│  └────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│  Step 3: SUMMARIZATION (Snowflake Cortex)                            │
│  ┌────────────────────┴───────────────────────────────┐             │
│  │  SELECT SNOWFLAKE.CORTEX.SUMMARIZE(                 │             │
│  │    clean_transcript                                  │             │
│  │  ) AS summary                                        │             │
│  │  FROM transcripts_clean;                             │             │
│  │                                                      │             │
│  │  5-page transcript → 2-3 sentence summary            │             │
│  │  Preserves key intent signals, removes noise         │             │
│  └────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│  Step 4: CLASSIFICATION (LLM — Teacher or Student)                   │
│  ┌────────────────────┴───────────────────────────────┐             │
│  │  Few-shot prompt + summary → structured output       │             │
│  │  { primary_intent, secondary_intent,                 │             │
│  │    tertiary_intent, confidence }                     │             │
│  └────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│  Step 5: STORAGE & DOWNSTREAM                                        │
│  ┌────────────────────┴───────────────────────────────┐             │
│  │  Classified results → production table               │             │
│  │  Feeds dashboards, reporting, operational tools      │             │
│  └────────────────────────────────────────────────────┘             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Why summarize before classifying?**
- Raw transcripts are long (often 2,000–10,000 tokens), exceeding efficient context window limits
- Summarization strips noise (pleasantries, repetitions, ASR errors) while preserving intent signals
- Dramatically reduces inference cost — classifying a 3-sentence summary is ~50x cheaper than a full transcript
- Cortex `SUMMARIZE()` runs natively inside Snowflake, avoiding data egress

---

## 2.2 Intent Taxonomy Design

### The Problem

Raw call logs from agents had **200+ free-text intent labels** — many were duplicates, misspellings, or overlapping categories. Examples:

```
Raw labels (sample):
  "flight change", "change flight", "flight amendment", "amend booking"
  "baggage complaint", "lost luggage", "missing bag", "bag not arrived"
  "refund request", "want money back", "refund", "cancellation refund"
  "hotel query", "accommodation question", "hotel booking issue"
```

### The Solution: Hierarchical Intent Mapping

Collaborated with product managers and contact centre leads to create a **three-level taxonomy**:

```
INTENT TAXONOMY (Hierarchical)
═══════════════════════════════════════════════════════════════
Level 1 (Primary)       Level 2 (Secondary)         Level 3 (Tertiary)
─────────────────       ───────────────────         ──────────────────
BOOKING_CHANGE    ───►  Flight Amendment      ───►  Date Change
                        Hotel Amendment               Passenger Name
                        Package Modification          Upgrade Request

CANCELLATION      ───►  Full Cancellation     ───►  Voluntary
                        Partial Cancellation          Involuntary (airline)
                        Refund Status                 Insurance Claim

COMPLAINT         ───►  Service Quality       ───►  Agent Behavior
                        Baggage Issues                Lost Luggage
                        Delay/Disruption              Flight Delay > 3hrs

INFORMATION       ───►  Booking Enquiry       ───►  Payment Status
                        Travel Requirements           Visa/Docs
                        Loyalty/Rewards               Points Balance

SALES             ───►  New Booking           ───►  Flight Only
                        Add-ons                       Hotel + Flight
                        Upgrades                      Extra Baggage
═══════════════════════════════════════════════════════════════
```

### Design Principles

1. **Mutually Exclusive at Each Level:** A call's primary intent falls into exactly one Level 1 category
2. **Multi-Intent Support:** A single call can have primary = `BOOKING_CHANGE`, secondary = `COMPLAINT` (e.g., customer wants to change a flight AND complains about service)
3. **Actionability:** Each leaf category maps to a concrete operational action or team
4. **SME Validation:** Domain experts reviewed every mapping; ambiguous cases resolved by majority vote
5. **Iterative Refinement:** Taxonomy was versioned; new intents discovered via low-confidence predictions were fed back into taxonomy updates

---

## 2.3 Snowflake Cortex: What It Is and How We Used It

### What is Snowflake Cortex?

Snowflake Cortex is Snowflake's **built-in AI/ML layer** that provides LLM-powered functions directly accessible via SQL. No need to move data outside Snowflake.

```
┌────────────────────────────────────────────────────────────┐
│                  SNOWFLAKE CORTEX OVERVIEW                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LLM Functions (SQL-callable)                         │   │
│  │                                                       │   │
│  │  CORTEX.COMPLETE(model, prompt)                       │   │
│  │    → General text generation, classification           │   │
│  │    → Models: llama3-70b, mistral-large, mixtral-8x7b  │   │
│  │                                                       │   │
│  │  CORTEX.SUMMARIZE(text)                               │   │
│  │    → Abstractive summarization of text                 │   │
│  │                                                       │   │
│  │  CORTEX.SENTIMENT(text)                               │   │
│  │    → Sentiment score [-1, 1]                          │   │
│  │                                                       │   │
│  │  CORTEX.TRANSLATE(text, from, to)                     │   │
│  │    → Machine translation                              │   │
│  │                                                       │   │
│  │  CORTEX.EXTRACT_ANSWER(text, question)                │   │
│  │    → Extractive QA                                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Key Benefits:                                               │
│  • Data never leaves Snowflake → compliance & security       │
│  • SQL interface → accessible to analysts, not just ML eng   │
│  • Serverless → no GPU provisioning                          │
│  • Scalable → processes millions of rows via warehouse       │
│  • Pay-per-query → no idle compute costs                     │
└────────────────────────────────────────────────────────────┘
```

### How We Used Cortex

**1. Summarization:**

```sql
-- Summarize each raw transcript using Cortex
SELECT
    call_id,
    SNOWFLAKE.CORTEX.SUMMARIZE(raw_transcript) AS call_summary
FROM call_transcripts_clean
WHERE call_date = CURRENT_DATE() - 1;
```

**2. Classification (Teacher - via COMPLETE):**

```sql
-- Classify summarized transcripts using few-shot prompting
SELECT
    call_id,
    SNOWFLAKE.CORTEX.COMPLETE(
        'llama3-70b',
        CONCAT(
            'You are a call intent classifier for a travel company. ',
            'Classify the following call summary into intents.\n\n',
            -- Few-shot examples
            'Example 1:\nSummary: "Customer called to change flight date...',
            'from March 15 to March 22 for 2 passengers."\n',
            'Output: {"primary": "BOOKING_CHANGE", "secondary": ',
            '"Flight Amendment", "tertiary": "Date Change"}\n\n',
            'Example 2:\nSummary: "Customer complained about lost luggage...',
            'on arrival at Malaga and requested compensation."\n',
            'Output: {"primary": "COMPLAINT", "secondary": ',
            '"Baggage Issues", "tertiary": "Lost Luggage"}\n\n',
            -- Actual input
            'Now classify:\nSummary: "', call_summary, '"\nOutput:'
        )
    ) AS classification_json
FROM call_summaries;
```

**Why Snowflake Cortex (not external API)?**
- **Data Governance:** Call transcripts contain PII — data never leaves Snowflake's security perimeter
- **No Infrastructure:** No GPU clusters to manage, no model hosting
- **SQL-Native:** Analysts and engineers both can use it; low barrier to adoption
- **Cost-Effective at Scale:** Serverless billing, scales with warehouse size
- **Compliance:** Jet2 operates under UK/EU data regulations (GDPR); keeping data in-platform simplifies compliance

---

## 2.4 Few-Shot Learning for Classification

### What is Few-Shot Learning?

Few-shot learning provides the LLM with **a small number of labeled examples** (typically 3–10) within the prompt to guide its classification behavior — no weight updates needed.

```
┌────────────────────────────────────────────────────────────────┐
│              LEARNING PARADIGMS COMPARISON                       │
│                                                                  │
│  ZERO-SHOT          FEW-SHOT              FINE-TUNING           │
│  ┌────────────┐    ┌────────────────┐    ┌─────────────────┐   │
│  │ No examples │    │ 3-10 examples  │    │ Thousands of    │   │
│  │ in prompt   │    │ in prompt      │    │ labeled samples │   │
│  │             │    │                │    │ + weight update  │   │
│  │ "Classify   │    │ "Example 1:    │    │                 │   │
│  │  this call" │    │  Summary: X    │    │ Training loop   │   │
│  │             │    │  Intent: Y     │    │ Loss + backprop │   │
│  │ Relies on   │    │ Example 2: ... │    │ Epochs of data  │   │
│  │ pretraining │    │ Now classify:" │    │                 │   │
│  │ knowledge   │    │                │    │ Model weights   │   │
│  │ alone       │    │ In-context     │    │ permanently     │   │
│  │             │    │ learning       │    │ changed         │   │
│  └────────────┘    └────────────────┘    └─────────────────┘   │
│                                                                  │
│  Accuracy:  Low ◄────────────────────────────────────► High     │
│  Cost:      Low ◄────────────────────────────────────► High     │
│  Speed:     Fast ◄───────────────────────────────────► Slow     │
│  Flex:      High ◄───────────────────────────────────► Low      │
└────────────────────────────────────────────────────────────────┘
```

### Our Few-Shot Strategy

**Example Selection Criteria:**
1. **Representative:** Chose 5–8 examples covering each primary intent category
2. **Edge Cases Included:** At least 1–2 ambiguous examples (e.g., a call that is both a complaint and a booking change)
3. **Diversity:** Examples varied in length, tone, complexity
4. **Hard Negatives:** Included examples that look similar but have different intents (e.g., "refund enquiry" vs "refund request")

**Prompt Structure:**

```
┌──────────────────────────────────────────────────────────────┐
│  SYSTEM: You are a call intent classifier for Jet2 Travel.   │
│  Classify each call summary into hierarchical intents.        │
│  Always return valid JSON with primary, secondary, tertiary.  │
│                                                                │
│  INTENT DEFINITIONS:                                           │
│  [List of all valid intent categories with brief descriptions] │
│                                                                │
│  EXAMPLES:                                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Summary: "Customer wants to move flight from Leeds       │ │
│  │ Bradford to Antalya from June 5 to June 12."            │ │
│  │ → {"primary":"BOOKING_CHANGE",                          │ │
│  │    "secondary":"Flight Amendment",                      │ │
│  │    "tertiary":"Date Change",                            │ │
│  │    "confidence":0.95}                                   │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Summary: "Caller asked about luggage allowance for      │ │
│  │ Tenerife trip, also mentioned wanting to add extra bag."│ │
│  │ → {"primary":"INFORMATION",                             │ │
│  │    "secondary":"Travel Requirements",                   │ │
│  │    "tertiary":"Extra Baggage",                          │ │
│  │    "confidence":0.88}                                   │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ [3-6 more examples covering edge cases]                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  INPUT: "<actual call summary>"                                │
│  OUTPUT (JSON):                                                │
└──────────────────────────────────────────────────────────────┘
```

### Iterative Prompt Refinement Process

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│ Draft   │     │ Run on 500   │     │ SME Review   │     │ Measure  │
│ Prompt  │────►│ sample calls │────►│ of errors    │────►│ Jaccard  │
│ v1      │     │              │     │              │     │ recall   │
└─────────┘     └──────────────┘     └──────────────┘     └────┬─────┘
                                                                │
     ┌──────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│  Error Analysis:                              │
│  • Which intents are confused most often?     │
│  • Are definitions clear enough?              │
│  • Do examples cover the failure cases?       │
│                                               │
│  Refinement Actions:                          │
│  • Add clarifying notes to intent definitions │
│  • Swap in better few-shot examples           │
│  • Add chain-of-thought reasoning step        │
│  • Adjust output format constraints           │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
               Prompt v2, v3, ... vN
               (repeat until metric plateau)
```

We went through **~8 prompt iterations** before finalizing, lifting the teacher's recall on validated intent labels from roughly 72% on the first draft to the level that became the ceiling for distillation.

---

## 2.5 Teacher-Student Distillation: Complete Walkthrough

### Why Distillation?

| Dimension | Teacher (70B) | Student (7B) |
|---|---|---|
| **Parameters** | ~70 billion | ~7 billion |
| **Inference latency** | ~8–15 sec/call | ~0.8–1.5 sec/call |
| **Cost per 1M calls** | $$$$$ (very expensive) | $ (10–15x cheaper) |
| **Quality** | Best-in-class accuracy | ~97% of teacher quality |
| **Production viability** | Too slow/expensive for daily batch | Suitable for daily batch on 5K calls |
| **GPU requirement** | 4x A100 (80GB) or large Cortex warehouse | 1x A100 or medium Cortex warehouse |

**Bottom line:** The 70B teacher gives gold-standard labels but is too expensive to run on every transcript daily. The 7B student replicates the teacher's behavior at a fraction of the cost.

### How Distillation Worked

```
┌──────────────────────────────────────────────────────────────────────┐
│                  TEACHER-STUDENT DISTILLATION PIPELINE                │
│                                                                      │
│  STEP 1: Teacher Generates Training Data                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  70B Teacher + few-shot prompt                                 │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  Run on ~50,000 diverse transcripts                            │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  Output: (summary, intent_labels, confidence)                  │  │
│  │  Filter: Keep only high-confidence predictions (conf > 0.85)   │  │
│  │  Result: ~40,000 high-quality labeled examples                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  STEP 2: Student Training (Supervised Fine-Tuning)                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  7B Base Model (e.g., Mistral-7B / Llama-2-7B)                │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  Fine-tune on teacher's (input, output) pairs                  │  │
│  │  Using SFT (Supervised Fine-Tuning)                            │  │
│  │  Format: input = call summary, output = JSON intent labels     │  │
│  │                                                                │  │
│  │  Training Config:                                              │  │
│  │  • LoRA rank: 16–64, alpha: 32–128                            │  │
│  │  • Learning rate: 1e-4 to 2e-5                                │  │
│  │  • Epochs: 3–5                                                │  │
│  │  • Batch size: 8–16 (gradient accumulation)                   │  │
│  │  • Validation split: 10%                                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  STEP 3: Validation                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Hold-out set: 5,000 transcripts                               │  │
│  │  Compare: Student predictions vs Teacher predictions            │  │
│  │  Also:    Student predictions vs Human gold labels              │  │
│  │                                                                │  │
│  │  Metrics:                                                      │  │
│  │  • Recall on validated intent labels (student):  94%           │  │
│  │  • Student–teacher agreement (label-set overlap): high         │  │
│  │  • Per-label precision & recall (every intent, separately)     │  │
│  │  • Confusion matrix across primary intents                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  STEP 4: Deployment                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Student model deployed for daily inference                    │  │
│  │  Teacher model used periodically for:                          │  │
│  │    • Quality audits (sample 500 calls/week)                   │  │
│  │    • Generating labels for new intent categories               │  │
│  │    • Refreshing student training data quarterly                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Distillation Details

**Why SFT on teacher outputs (not classic KD with KL divergence)?**

For generative LLMs, classic KD (matching logit distributions via KL divergence) requires access to the teacher's full output probability distribution over the entire vocabulary at every token position. This is:
- Computationally prohibitive for 70B models (storing/transferring logits for 32K+ vocab tokens per position)
- Often impractical when using Cortex/API-based inference (no logit access)

Instead, we used **sequence-level distillation** — the student learns to replicate the teacher's *text outputs* directly via supervised fine-tuning. This is the approach used by most production LLM distillation systems (Alpaca, Vicuna, etc.).

```
Classic KD (Hinton):        Student matches teacher's PROBABILITY DISTRIBUTION
                            Loss = KL(teacher_logits, student_logits)
                            Requires: full logit access

Sequence-Level Distillation: Student matches teacher's TEXT OUTPUT
                            Loss = Cross-entropy on (input, teacher_output) pairs
                            Requires: only teacher's generated text
                            Used by: Alpaca, Vicuna, Orca, our system
```

### Confidence-Based Filtering

Not all teacher predictions are equally reliable. We filtered training data by confidence:

```python
# Pseudocode for teacher data filtering
teacher_predictions = run_teacher_on_batch(transcripts_50k)

high_quality = [
    pred for pred in teacher_predictions
    if pred['confidence'] > 0.85
    and pred['primary_intent'] in VALID_INTENTS
    and is_valid_json(pred['output'])
]

# ~40,000 of 50,000 passed filtering
# The remaining ~10,000 had low confidence or malformed output
```

---

## 2.6 Multi-Label vs Multi-Class Classification

This project is **multi-label** — a single call can have multiple intents simultaneously.

| Aspect | Multi-Class | Multi-Label |
|---|---|---|
| **Definition** | Each sample belongs to exactly ONE class | Each sample can belong to MULTIPLE classes |
| **Example** | "This email is spam" or "not spam" | "This call is BOOKING_CHANGE + COMPLAINT" |
| **Output** | Single class label / argmax of softmax | Set of labels / independent sigmoid per label |
| **Loss function** | Cross-entropy (softmax) | Binary cross-entropy (sigmoid per class) |
| **Evaluation** | Accuracy, macro/micro F1 | Jaccard index, subset accuracy, hamming loss |
| **Our approach** | — | LLM outputs JSON with all applicable intents |

**Our framing:** The LLM generates structured JSON containing all applicable intents. The prompt explicitly instructs: *"A call may have multiple intents. List ALL that apply."*

```json
// Example: Multi-intent call
{
  "primary_intent": "BOOKING_CHANGE",
  "secondary_intent": "COMPLAINT",
  "tertiary_intent": "Date Change",
  "all_intents": ["BOOKING_CHANGE", "COMPLAINT", "Flight Amendment", "Date Change", "Service Quality"],
  "confidence": 0.91
}
```

---

## 2.7 Recall on Validated Intent Labels: Why This Metric

**Headline:** the number I quote is **94% recall on validated intent labels**. The mechanics of how that recall is computed for a *set-valued* prediction are Jaccard-style — that's a metric-design choice worth being able to defend, and it's what the rest of this section explains.

### The Formula

For multi-label classification where each sample has a **set** of predicted labels and a **set** of true labels:

**Jaccard Similarity (per sample):**

```
                    |Predicted ∩ True|
J(Predicted, True) = ─────────────────
                    |Predicted ∪ True|
```

**Set-based recall (our headline metric):**

```
                         |Predicted ∩ True|
Recall (per sample) = ─────────────────
                         |True|
```

Written properly:

$$
\text{Recall} = \frac{1}{N}\sum_{i=1}^{N} \frac{|\hat{Y}_i \cap Y_i|}{|Y_i|}
$$

where $Y_i$ is the SME-validated label set for transcript $i$ and $\hat{Y}_i$ is the predicted label set.

This measures: **of the true intents, what fraction did the model correctly predict?**

> **Say this if asked how you computed it:** "It's recall over the validated label set. For each call we compare the predicted set of intents against the SME-validated set and take the fraction of true labels we recovered, then average across the hold-out. I also report it micro-averaged — pooling true positives and false negatives across all calls — because the per-sample average over-weights short single-intent calls."

### Concrete Example

```
True intents:      {BOOKING_CHANGE, COMPLAINT, Date Change}
Predicted intents: {BOOKING_CHANGE, Date Change, Service Quality}

Jaccard Similarity = |{BOOKING_CHANGE, Date Change}| / |{BOOKING_CHANGE, COMPLAINT, Date Change, Service Quality}|
                   = 2 / 4 = 0.50

Jaccard Recall     = |{BOOKING_CHANGE, Date Change}| / |{BOOKING_CHANGE, COMPLAINT, Date Change}|
                   = 2 / 3 = 0.67

(We missed COMPLAINT; we over-predicted Service Quality)
```

### Why Set-Based Recall (Not Accuracy, Not F1)?

| Metric | Problem for Our Use Case |
|---|---|
| **Exact Match Accuracy** | Too strict — requires ALL intents to match exactly. A call with 4 intents where we get 3 right scores 0%. |
| **Hamming Loss** | Treats all labels equally. Missing "primary intent" penalized same as missing a rare tertiary. |
| **Standard F1** | Designed for single-label. Doesn't naturally handle set-based evaluation. (We *do* report per-label F1 as a secondary diagnostic.) |
| **Jaccard Similarity** | Symmetric — penalizes over-prediction as hard as under-prediction. Tracked as a guardrail, but it isn't the business metric. |
| **Set-based Recall (chosen)** | Measures coverage — "did we capture the customer's true intents?" Partial credit for partial matches. Operationally, recall matters more: missing an intent means missing an action item. |

**94% recall on validated intent labels means:** on the SME-validated hold-out set, the student model recovers 94% of the true intent labels assigned to each call.

### The Honest Caveat — Recall Alone Can Be Gamed

Recall is trivially maximised by predicting every intent on every call. That is why we always report it alongside two guardrails:

| Guardrail | What it Catches | Our Position |
|---|---|---|
| **Jaccard similarity** (symmetric) | Over-prediction — the model spraying labels to boost recall | Materially below the recall figure, which is expected and acceptable; the gap *is* the over-prediction rate |
| **Mean predicted labels per call vs mean true labels per call** | Label inflation | Monitored daily; a rising ratio is treated as a regression even if recall improves |

> **Say this:** "Recall is the headline because a missed intent is a missed action item, but I never quote it naked. I pair it with Jaccard similarity and with the average number of labels we emit per call, because a model that predicts everything gets 100% recall and is worthless. Our label-count-per-call stayed in line with the human labellers, which is what tells you the 94% is real coverage and not spray-and-pray."

---

## 2.8 Knowledge Distillation Deep Dive (70B → 7B)

> **Cross-reference:** the theory behind everything in this section — classic Hinton KD, temperature scaling, response/feature/relation-based variants, self-distillation, and the KD loss derivations — is written up in [`../learning/08_knowledge_distillation.md`](../learning/08_knowledge_distillation.md). Read that for the *why the maths works*; read this for the *what I actually built*.

This is the section the new resume bullet points at. If an interviewer digs into "distilling a 70B teacher into a 7B model," everything they can reasonably ask is below.

---

### 2.8.1 Why Distill At All? (Cost, Latency, Deployment)

The 70B teacher was never a candidate for production. Three independent constraints ruled it out, and any one of them would have been enough.

| Constraint | The 70B Problem | What the 7B Student Fixed |
|---|---|---|
| **Cost** | Roughly an order of magnitude more expensive per inference. At 2,000–5,000 calls/day, every day, forever, that difference compounds into real budget. | ~10–15x cheaper per inference. Turned a line item that needed sign-off into one that didn't. |
| **Latency** | ~8–15 sec/transcript. The nightly batch would not finish inside its window during seasonal peaks. | ~0.8–1.5 sec/transcript. Batch completes with headroom even on disruption days when volume triples. |
| **Deployment footprint** | Needs multi-GPU (4× A100 80GB class) or a large managed warehouse held open. | Fits on a single accelerator; can be quantised to INT8 and served from modest infrastructure. |
| **Prompt overhead** | Few-shot examples add ~500 tokens to *every* call. You pay for the instructions a million times. | Behaviour is baked into the weights. No examples in the prompt — just the summary. |

> **Say this:** "The teacher was a labelling instrument, not a product. It was the most accurate thing we had, but running it on every call every night would have been roughly ten times the cost and ten times the latency, and it wouldn't have finished inside the batch window during summer peak. Distillation let me keep the teacher's decision boundary and throw away its inference bill."

**The framing that lands well in interviews:** distillation converts a *recurring* inference cost into a *one-off* training cost. You pay the 70B once, on a sample, to produce a dataset. Then you pay 7B prices forever.

---

### 2.8.2 The Distillation Landscape — Which Variant, and Why

There are three canonical families of knowledge distillation. Know all three; be clear about which one you used and why the others didn't apply.

| Family | What the Student Matches | Requires | Typical Use |
|---|---|---|---|
| **Response-based** (Hinton et al., 2015) | The teacher's **output distribution** — soft targets over classes | Access to teacher logits | Classification with a fixed label space |
| **Feature-based** (FitNets, Romero et al.) | The teacher's **intermediate hidden representations** (hint layers), usually via an MSE loss on projected activations | White-box access to teacher internals; architectures that can be aligned | Compressing within an architecture family |
| **Relation-based** (RKD, Park et al.) | The **relationships between samples** — pairwise distances or angles in the teacher's embedding space, rather than individual outputs | Teacher embeddings for batches of samples | Metric learning, retrieval, embedding compression |

**What I used: sequence-level distillation** (Kim & Rush) — a practical variant of response-based KD for generative models, where the student is trained on the teacher's *generated text* rather than its logits.

```
Classic response-based KD:  Student matches teacher's PROBABILITY DISTRIBUTION
                            Loss = KL(teacher_soft || student_soft) · T²
                            Requires: full logit access over the vocabulary

Feature-based KD:           Student matches teacher's HIDDEN STATES
                            Loss = MSE(proj(student_h_l), teacher_h_m)
                            Requires: white-box teacher, layer alignment

Relation-based KD:          Student matches teacher's SAMPLE GEOMETRY
                            Loss = distance/angle agreement across sample pairs
                            Requires: teacher embeddings, batch-level structure

Sequence-level KD (ours):   Student matches teacher's TEXT OUTPUT
                            Loss = CE on (input, teacher_output) pairs
                            Requires: only the teacher's generated text
                            Used by: Alpaca, Vicuna, Orca, and this system
```

**Why sequence-level and not classic logit matching?** Two reasons, one theoretical and one entirely practical:

1. **Practical (the real reason):** the teacher ran through Snowflake Cortex. Cortex returns generated text, not per-token logit distributions over a 32K+ vocabulary. Classic KD was not available to us — you cannot match a distribution you cannot see.
2. **Theoretical:** even with logit access, storing soft targets for every position across a 32K vocabulary for tens of thousands of multi-token JSON outputs is a large, awkward artefact for a marginal gain. Sequence-level KD is the standard production compromise, and the evidence from the open instruction-tuning literature is that it captures most of the benefit.

> **Say this if challenged:** "I used sequence-level distillation, which is response-based KD applied to a generative model — the student imitates the teacher's outputs rather than its logits. I want to be precise about that because 'distillation' sometimes implies Hinton-style KL matching on soft targets, and I didn't have logit access through Cortex. If I'd been self-hosting the teacher I'd have tested true soft-target KD, and I'd expect a small additional gain on the ambiguous, low-margin cases specifically — that's exactly where the soft distribution carries information the hard label doesn't."

---

### 2.8.3 The Classic KD Objective (Know This Cold)

Even though we used sequence-level KD, you will be asked about the classic formulation. It is the single most likely whiteboard question attached to this bullet.

$$
\mathcal{L}_{\text{KD}} = \alpha \cdot T^2 \cdot \mathrm{KL}\!\left(\sigma\!\left(\frac{z_t}{T}\right) \,\Big\|\, \sigma\!\left(\frac{z_s}{T}\right)\right) + (1-\alpha) \cdot \mathrm{CE}\!\left(\sigma(z_s),\, y\right)
$$

where:

| Symbol | Meaning |
|---|---|
| $z_t, z_s$ | Teacher and student logits |
| $T$ | Temperature — softens both distributions |
| $\alpha$ | Blend weight between the soft (KD) and hard (ground-truth) terms |
| $\sigma$ | Softmax (or sigmoid, in the multi-label case) |
| $\mathrm{KL}$ | Kullback–Leibler divergence |
| $\mathrm{CE}$ | Cross-entropy against the hard labels |

**Why the $T^2$ factor?** Softening by $T$ shrinks the gradients of the soft-target term by roughly $1/T^2$. Multiplying by $T^2$ restores the gradient magnitude so that $\alpha$ actually controls the *balance* between the two terms rather than being silently confounded with the temperature. Without it, raising $T$ would quietly turn the KD term off.

**Why temperature helps — the "dark knowledge" argument:**

$$
\text{At } T=1:\quad p_{\text{teacher}} = [0.95,\; 0.03,\; 0.02] \quad \rightarrow \quad \text{student learns "it's class 1"}
$$

$$
\text{At } T=5:\quad p_{\text{teacher}} \approx [0.45,\; 0.30,\; 0.25] \quad \rightarrow \quad \text{student learns "class 2 is a near-miss, class 3 less so"}
$$

The relative ordering of the *wrong* answers is the information the hard label throws away. For our taxonomy this matters enormously: knowing that `Flight Amendment` was the runner-up to `Package Modification` teaches the student the shape of the taxonomy, not just the answer key.

**The $\alpha$ blend in practice:** $\alpha$ high (0.7–0.9) means "trust the teacher"; $\alpha$ low means "trust the human labels." The sensible policy is *not* a constant. Where we had SME gold labels, those dominate; where we only had teacher output, the KD term is all there is.

```python
# Conceptual: what the blended objective looks like for a multi-label head.
# (Our production student was a generative model trained with sequence-level KD;
#  this is the discriminative formulation you'd whiteboard.)
import torch
import torch.nn.functional as F

def multilabel_kd_loss(student_logits, teacher_logits, hard_targets,
                       T: float = 3.0, alpha: float = 0.7):
    """Per-label sigmoid KD: each intent is an independent binary decision."""
    # Soft term: binary KL between per-label Bernoullis at temperature T
    t_prob = torch.sigmoid(teacher_logits / T)
    s_logp = F.logsigmoid(student_logits / T)
    s_logp_neg = F.logsigmoid(-student_logits / T)

    soft = -(t_prob * s_logp + (1 - t_prob) * s_logp_neg).mean()
    soft = soft * (T ** 2)          # restore gradient scale

    # Hard term: standard BCE against ground truth where it exists
    hard = F.binary_cross_entropy_with_logits(student_logits, hard_targets)

    return alpha * soft + (1 - alpha) * hard
```

**Choosing $T$:** for a taxonomy with ~30–50 labels where the teacher is confident, $T$ in the **2–4** range is the defensible default, and $T=3$ is a reasonable single answer. The logic: $T$ too low (1) and there is no dark knowledge to transfer — you have re-derived plain supervised fine-tuning. $T$ too high (10+) and the distribution flattens toward uniform, so the student spends capacity learning noise in the tail of a label space it will never predict. You tune it on the validation set like any other hyperparameter, and the tell-tale of "too high" is rare-intent precision collapsing while overall recall looks fine.

---

### 2.8.4 Generating Teacher Labels Across 1M+ Transcripts

You do not run a 70B model on a million transcripts. The design question is *which* transcripts to spend the teacher's budget on.

```
┌────────────────────────────────────────────────────────────────────────┐
│         TEACHER LABEL GENERATION — SAMPLING & QUALITY CONTROL          │
│                                                                        │
│  1M+ transcripts in Snowflake                                          │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ STRATIFIED SAMPLE  (~50,000 transcripts)                     │      │
│  │  • Stratified by: coarse intent proxy (agent's raw label),   │      │
│  │    call duration bucket, channel, month (seasonality)        │      │
│  │  • Rare-intent OVERSAMPLING: rare categories deliberately    │      │
│  │    over-represented vs their natural frequency               │      │
│  │  • Why: a proportional sample would give the student only a  │      │
│  │    handful of examples of LOYALTY or SPECIAL_ASSISTANCE      │      │
│  └───────────────────────────┬──────────────────────────────────┘      │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ TEACHER INFERENCE (70B, few-shot, chain-of-thought ON)       │      │
│  │  • CoT enabled here even though it's slow — this is a        │      │
│  │    one-off labelling job, not the production path            │      │
│  │  • Emits: intent set + per-label confidence + reasoning      │      │
│  └───────────────────────────┬──────────────────────────────────┘      │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ QUALITY CONTROL GATES  (see 2.8.5)                           │      │
│  │  • Schema validity  • Confidence floor  • Taxonomy closure   │      │
│  │  • Self-consistency • SME spot-audit                         │      │
│  └───────────────────────────┬──────────────────────────────────┘      │
│                              ▼                                         │
│         ~40,000 curated (summary → intent set) training pairs          │
└────────────────────────────────────────────────────────────────────────┘
```

**Why ~50K and not more?** Two reasons. First, diminishing returns — for a classification task over a ~30–50 label taxonomy, the student's learning curve flattens well before you exhaust the data; doubling the labelling spend was not going to buy another few points. Second, teacher inference on the sample was itself a meaningful compute bill, and every additional transcript labelled was budget not spent on SME validation, which was the scarcer resource.

**Why the full 1M+ still matters:** the million-plus figure is the *processed* volume — the student ran over the entire historical corpus and then over every new call. The 50K is the *labelled* subset used to train it. Be precise about this distinction if asked; conflating them is exactly the kind of sloppiness an interviewer probes for.

---

### 2.8.5 Quality Control on Teacher Labels

The single biggest risk in distillation is **inheriting the teacher's mistakes and calling them ground truth**. Five gates addressed this.

| # | Gate | Mechanism | What It Catches |
|---|---|---|---|
| 1 | **Schema validity** | Parse the JSON; reject unparseable output | Malformed generations, truncation, trailing prose |
| 2 | **Taxonomy closure** | Every emitted label must exist in the versioned taxonomy | Hallucinated intent categories — the teacher inventing `"Flight Complaint Refund"` |
| 3 | **Confidence floor** | Keep predictions above a confidence threshold (~0.85) | Low-margin guesses that would teach the student to be confidently wrong |
| 4 | **Self-consistency** | Re-run a subset at non-zero temperature; keep labels stable across samples | Genuinely ambiguous calls where the teacher's answer is arbitrary |
| 5 | **SME spot-audit** | Domain experts review a stratified sample of teacher labels, weighted toward rare intents and low-confidence cases | Systematic teacher bias — a whole category being consistently mislabelled |

Roughly 10,000 of the 50,000 were dropped by these gates, leaving ~40,000 training pairs.

```python
# Teacher output curation
def curate_teacher_labels(predictions, taxonomy, conf_floor=0.85):
    kept, rejected = [], []
    for p in predictions:
        reasons = []
        if not is_valid_json(p["raw_output"]):
            reasons.append("schema")
        if not set(p["intents"]) <= taxonomy:          # taxonomy closure
            reasons.append("hallucinated_label")
        if p["confidence"] < conf_floor:
            reasons.append("low_confidence")
        if not p["self_consistent"]:
            reasons.append("unstable")

        (rejected if reasons else kept).append({**p, "reject_reasons": reasons})
    return kept, rejected
```

**The subtlety worth volunteering in an interview:** gate 3 (the confidence floor) is not free. Filtering to high-confidence teacher predictions systematically removes the *hard* examples — the ambiguous, multi-intent, edge-case calls. You end up with a clean dataset that under-represents exactly the cases the student will struggle with in production. Our mitigation was to keep a deliberately curated slice of hard cases with **SME labels rather than teacher labels**, so the student saw difficulty without inheriting teacher guesswork. This is the answer to "didn't your filtering make the task artificially easy?"

---

### 2.8.6 Validating That the Student Retained Recall (94%)

Validation had to answer two separate questions, and conflating them is a classic error:

1. **Did the student learn the teacher?** → student-vs-teacher agreement
2. **Is the student actually correct?** → student-vs-human-gold

Only the second one is a quality claim. The 94% is measured against **SME-validated labels**, not teacher agreement.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     VALIDATION DESIGN                                │
│                                                                      │
│  ┌────────────────────────────┐   ┌──────────────────────────────┐   │
│  │ SET A — Teacher hold-out   │   │ SET B — HUMAN GOLD           │   │
│  │ ~5,000 transcripts         │   │ ~1,000 transcripts           │   │
│  │ Labelled by teacher        │   │ Labelled by SMEs, adjudicated│   │
│  │ EXCLUDED from training     │   │ NEVER seen by any model      │   │
│  │                            │   │                              │   │
│  │ Answers: "did the student  │   │ Answers: "is the student     │   │
│  │ faithfully imitate?"       │   │ RIGHT?"  ◄── the 94% lives   │   │
│  │                            │   │                here          │   │
│  └────────────────────────────┘   └──────────────────────────────┘   │
│                                                                      │
│  Reported side by side. Teacher is also scored on SET B, so we know  │
│  the ceiling. Student sits a small margin below teacher on SET B.    │
└──────────────────────────────────────────────────────────────────────┘
```

**Slice-level validation — the part that actually protects you.** A single aggregate number hides collapse on rare intents. We cut recall by:

| Slice | Why It Matters |
|---|---|
| **Per intent label** | The only way to see a rare category silently collapsing to zero recall |
| **Head vs tail intents** | High-frequency intents are easy; the tail is where distillation degrades |
| **Number of true intents per call** (1, 2, 3+) | Multi-intent calls are strictly harder — recall should be checked separately for them |
| **Call length bucket** | Very short and very long calls fail differently |
| **Time period** | Seasonal drift; a model validated only on winter data is not validated |

> **Say this:** "The 94% is student-versus-human-gold on a thousand transcripts that our SMEs labelled and adjudicated, held out entirely. I report student-versus-teacher agreement separately, because that only tells you the distillation worked — it doesn't tell you the teacher was right. And I never look at the aggregate alone: I break recall down per label, because a headline number can look fine while a rare intent has quietly gone to zero."

**How to quote the teacher's number honestly:** the teacher sits above the student — that is the expected direction, and a student that *beat* its teacher would be a claim I'd want to explain rather than boast about. I'd describe the student as recovering the large majority of the teacher's quality at roughly a tenth of the cost, and give the teacher's figure as an approximate reference point rather than a second headline metric, because the number we validated, tracked and shipped against was the student's.

---

### 2.8.7 Multi-Label Specifics: Sigmoid, BCE, and Per-Label Thresholds

This is a **multi-label** problem, and the new resume bullet says so explicitly. Interviewers will check that you understand what that changes.

**The core distinction:**

$$
\text{Multi-class (softmax):} \quad p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad \sum_k p_k = 1
$$

$$
\text{Multi-label (sigmoid):} \quad p_k = \frac{1}{1 + e^{-z_k}}, \qquad \text{each } p_k \text{ independent}
$$

Softmax forces the labels to compete — raising one probability necessarily lowers another. That is exactly wrong when a call is genuinely both a `BOOKING_CHANGE` and a `COMPLAINT`. Sigmoid treats each intent as its own binary question: "is this intent present, yes or no?"

**The loss follows from the head.** Per-label sigmoid implies **binary cross-entropy summed over labels**:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{K}\sum_{k=1}^{K} \Big[ y_k \log p_k + (1-y_k)\log(1-p_k) \Big]
$$

**Handling imbalance inside the loss.** With a long-tailed taxonomy, plain BCE lets the head intents dominate the gradient. Two standard fixes, both defensible:

- **Positive class weighting** — scale the positive term per label by $w_k \propto \frac{N_{\text{neg},k}}{N_{\text{pos},k}}$, so rare intents contribute proportionally.
- **Focal loss** — down-weight easy examples via $(1-p_k)^\gamma$, focusing capacity on the hard, rare cases.

```python
# Per-label positive weighting for a long-tailed intent taxonomy
pos_weight = torch.tensor(
    [(n_negative[k] / max(n_positive[k], 1)) for k in range(num_labels)]
).clamp(max=20.0)   # cap: unbounded weights destabilise training on ultra-rare labels

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

#### Per-Label Threshold Tuning

A single global 0.5 threshold is the most common mistake in multi-label classification. Each label has its own base rate, its own difficulty, and its own asymmetric cost of being wrong — so each label gets its own threshold.

**The procedure:**

```
FOR each intent label k:
    1. Collect (predicted probability, true label) pairs on the VALIDATION set
       — never the test set, or your thresholds are fitted to your own metric
    2. Sweep candidate thresholds τ across [0.05, 0.95]
    3. At each τ compute precision_k(τ) and recall_k(τ)
    4. Choose τ_k = the LOWEST threshold that satisfies the label's
       precision floor
          → maximises recall subject to precision not falling below
            what the downstream team can tolerate
    5. Rare / high-cost labels get a LOWER τ  (catch them; accept more FPs)
       Head / low-cost labels get a HIGHER τ (keep the queue clean)
```

```python
def tune_threshold(probs_k, y_k, min_precision):
    """Lowest threshold meeting the precision floor => maximum recall."""
    best_tau, best_recall = 0.5, -1.0
    for tau in np.arange(0.05, 0.96, 0.01):
        pred = probs_k >= tau
        tp = (pred & (y_k == 1)).sum()
        fp = (pred & (y_k == 0)).sum()
        fn = (~pred & (y_k == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        if precision >= min_precision and recall > best_recall:
            best_tau, best_recall = tau, recall
    return best_tau
```

**Why the precision floor varies per label — this is the business argument:**

| Label Type | Threshold | Reasoning |
|---|---|---|
| `COMPLAINT`, `CANCELLATION` | **Low** | Missing one means an unaddressed customer issue. A false positive costs one wasted review. Wildly asymmetric — bias hard toward recall. |
| `SPECIAL_ASSISTANCE` | **Low** | Duty-of-care implications. Under no circumstances do you want to miss these. |
| `INFORMATION`, `SALES` | **Higher** | High base rate, low cost of a miss. A loose threshold here floods the dashboards and dilutes every downstream count. |
| `OTHER / UNCLASSIFIABLE` | **High** | This is the dumping ground. A loose threshold makes it absorb everything and destroys the taxonomy's usefulness. |

#### Why Recall Is the Business Metric for Intent Routing

The asymmetry is the whole argument, and it is worth stating in cost terms rather than ML terms:

- **False negative** (missed intent): the call is never routed to the team that should handle it. A complaint goes unlogged. A cancellation request isn't actioned. The customer churns, or escalates, or posts about it. The cost is unbounded and lands outside the contact centre.
- **False positive** (spurious intent): one extra item in a review queue. Someone spends thirty seconds dismissing it. The cost is bounded, small, and internal.

> **Say this:** "Recall is the business metric because the two error types have completely different cost profiles. A false positive costs us thirty seconds of an analyst's time. A false negative means a customer's complaint never reaches the team that could have fixed it — and we don't find out until it escalates. When the costs are that asymmetric, you tune the thresholds toward recall and you accept the precision hit deliberately, per label, rather than pretending a single 0.5 cut-off is optimal for all of them."

---

### 2.8.8 The Trade-Off Table (Accuracy vs Latency vs Cost)

The table an interviewer will want to see if they ask "quantify the distillation trade-off."

| Dimension | Teacher (70B, few-shot) | Student (7B, distilled) | Delta |
|---|---|---|---|
| **Recall on validated intent labels** | Small margin above the student | **94%** | Student gives up a little quality |
| **Latency / transcript** | ~8–15 sec | ~0.8–1.5 sec | **~10x faster** |
| **Cost / inference** | Baseline | ~1/10th to 1/15th | **~10–15x cheaper** |
| **Prompt tokens / call** | ~800 (few-shot examples + instructions + summary) | ~200 (summary + short instruction) | ~4x fewer |
| **Serving footprint** | 4× A100 80GB class / large warehouse | 1 accelerator, INT8-quantisable | Order of magnitude smaller |
| **Iteration speed on taxonomy change** | Minutes (edit the prompt) | Hours (retrain) | **Teacher wins** — this is a real cost of distillation |
| **Role in the final system** | Weekly quality audits, labels for new intent categories, quarterly training-data refresh | Every call, every night | — |

**The honest cost of distillation, which you should volunteer:** you trade *flexibility* for *efficiency*. The few-shot teacher could absorb a new intent category in the time it takes to edit a prompt. The distilled student needs a retraining cycle. That is why we kept the teacher alive rather than decommissioning it — it remained the mechanism for extending the taxonomy, and the student inherited each extension at the next refresh.

---

# 3. Key Metrics & Results

---

## 3.1 Performance Summary

| Metric | Value | Context |
|---|---|---|
| **Recall on validated intent labels (student)** | **94%** | Student model on held-out test set against human-verified SME labels — **this is the resume number** |
| **Recall (teacher, same hold-out)** | A small margin above the student | Teacher is the ceiling; the student closes most of the gap. See §2.8.6 for how to quote this honestly |
| **Precision** | Reasoned estimate, not a hard number — see Trick Q1 | We optimised recall; precision is meaningfully lower and I discuss the trade-off rather than quoting a fabricated figure |
| **Primary Intent Accuracy** | ~93% | Exact match on just the top-level intent |
| **Processing Time (teacher)** | ~8–15 sec/transcript | Including summarization + classification |
| **Processing Time (student)** | ~0.8–1.5 sec/transcript | ~10x faster |
| **Daily Throughput** | 2,000–5,000 transcripts/day | Fully automated, no manual intervention |
| **Total Processed** | 1,000,000+ | Over the project lifetime |
| **Cost Reduction** | ~10–15x | Student vs teacher inference cost |
| **Manual Review Reduction** | ~85–90% | Humans only review low-confidence predictions |

## 3.2 Business Impact

| Impact Area | Description |
|---|---|
| **Operational Efficiency** | Contact centre managers get real-time intent distribution — can staff teams to match demand (e.g., spike in cancellation calls after disruption). |
| **Agent Training** | Identified common complaint patterns → targeted training for agents on high-frequency issues. |
| **Product Insights** | Discovered previously unknown intent clusters (e.g., post-COVID travel anxiety calls became a detectable category). |
| **Cost Savings** | Eliminated ~4 FTEs worth of manual transcript review labor. |
| **Recognition** | Earned internal recognition as a **key AI innovation** at Jet2. Presented to senior leadership. |
| **Scalability** | System scales linearly — handles seasonal spikes (summer holidays, disruptions) without manual intervention. |

## 3.3 Timeline & Iterations

```
Month 1-2:  Intent taxonomy design + stakeholder alignment
Month 3:    Snowflake Cortex setup, summarization pipeline
Month 4-5:  Few-shot prompt engineering (teacher model), ~8 iterations
Month 6:    Teacher model evaluation, SME validation
Month 7-8:  Student distillation, SFT training, validation
Month 9:    Production deployment, dashboard integration
Month 10+:  Monitoring, drift detection, continuous improvement
```

---

# 4. Topics You Must Know

---

## 4.1 LLMs: Core Architecture

### Transformer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  TRANSFORMER BLOCK (Decoder-Only)            │
│                                                              │
│  Input Tokens → Token Embeddings + Positional Embeddings     │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Multi-Head Self-Attention (Causal/Masked)           │    │
│  │  • Q, K, V projections from input                    │    │
│  │  • Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V        │    │
│  │  • Causal mask: can only attend to past tokens       │    │
│  │  • Multiple heads capture different relationships     │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                             │ + Residual + LayerNorm         │
│                             ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Feed-Forward Network (FFN)                          │    │
│  │  • Linear → GELU/SiLU → Linear                      │    │
│  │  • Expands dimension (e.g., 4096 → 16384 → 4096)    │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                             │ + Residual + LayerNorm         │
│                             ▼                                │
│  Repeat N times (e.g., 32 layers for 7B, 80 for 70B)       │
│       │                                                      │
│       ▼                                                      │
│  Output → Linear → Softmax → Next Token Probability          │
└─────────────────────────────────────────────────────────────┘
```

**Key concepts to articulate in interviews:**
- **Tokenization:** Subword tokenization (BPE/SentencePiece). A word like "unbelievable" might be split into ["un", "believ", "able"]. Affects context window utilization.
- **Context Window:** Maximum number of tokens the model can process at once. Llama 2: 4K tokens; Llama 3: 8K–128K. Our summarization step ensures transcripts fit within context limits.
- **Temperature:** Controls randomness of output. T=0: deterministic (greedy), T=1: standard sampling, T>1: more creative. For classification, we used **T=0 or T=0.1** for consistency.
- **Top-p / Top-k:** Nucleus sampling parameters. For classification, deterministic decoding is preferred.

---

## 4.2 Few-Shot vs Zero-Shot vs Fine-Tuning

| Dimension | Zero-Shot | Few-Shot | Fine-Tuning |
|---|---|---|---|
| **Training data needed** | 0 | 3–10 examples (in prompt) | 1,000–100,000+ labeled samples |
| **Model weights change?** | No | No | Yes |
| **Cost per inference** | Low | Medium (longer prompt) | Low (after training) |
| **Upfront cost** | Zero | Minimal (craft prompt) | High (GPU hours, data labeling) |
| **Adaptability** | Limited to pretraining knowledge | Good — can define custom intents via examples | Best — model specializes to your domain |
| **When to use** | Simple tasks, well-known categories | Custom categories, domain-specific tasks | High-volume production, specialized domains |
| **Our project** | Initial exploration | Teacher model (production prompt) | Student model (SFT on teacher data) |

**Interview-ready framing:**
> "We used few-shot for the teacher because it gave us rapid iteration — we could change intent definitions in minutes by editing the prompt, without retraining. For the student, we fine-tuned because it needed to be fast and cheap at inference; the few-shot examples would add latency and token cost at scale."

---

## 4.3 Knowledge Distillation: Deep Dive

> **See also:** §2.8 for the project-specific implementation, and [`../learning/08_knowledge_distillation.md`](../learning/08_knowledge_distillation.md) for the full theory notes.

### Classic KD (Hinton et al., 2015)

```
Loss = α · KL(softmax(teacher_logits / T), softmax(student_logits / T)) · T²
     + (1-α) · CE(student_logits, hard_labels)

Where:
  T     = Temperature (typically 2-20, softens distributions)
  α     = Weight balancing soft vs hard loss (typically 0.5-0.9)
  KL    = KL Divergence
  CE    = Cross-Entropy
```

**Temperature's role:**
- High T → softer probability distributions → more information about inter-class relationships
- At T=1: Teacher is confident (e.g., [0.95, 0.03, 0.02]) → student learns little beyond the correct answer
- At T=5: Same teacher outputs become [0.45, 0.30, 0.25] → student learns relative similarities

### Sequence-Level Distillation (Our Approach for LLMs)

```
┌──────────────────────────────────────────────────────────────┐
│  SEQUENCE-LEVEL DISTILLATION (for generative LLMs)            │
│                                                                │
│  1. Teacher generates text output Y for input X                │
│     Y = teacher.generate(X)                                    │
│                                                                │
│  2. Collect dataset: D = {(X₁, Y₁), (X₂, Y₂), ..., (Xₙ, Yₙ)}│
│                                                                │
│  3. Fine-tune student on D using standard causal LM loss:      │
│     Loss = -Σ log P_student(yₜ | y<t, X)                      │
│                                                                │
│  This is exactly SFT — but the "labels" come from the teacher  │
│  rather than humans.                                           │
│                                                                │
│  Advantages over classic KD:                                   │
│  • No need for teacher logits (works with API models)          │
│  • Simpler implementation                                      │
│  • Works for generative tasks (not just classification)        │
│  • Standard SFT tooling (LoRA, DeepSpeed, etc.)               │
└──────────────────────────────────────────────────────────────┘
```

---

## 4.4 Prompt Engineering for Classification

### Key Techniques

| Technique | Description | How We Used It |
|---|---|---|
| **System prompt** | Define the model's role and constraints | "You are a Jet2 call intent classifier..." |
| **Intent definitions** | Explicit descriptions of each category | Listed all valid intents with 1-line definitions |
| **Few-shot examples** | Labeled input-output pairs | 5–8 representative examples per prompt |
| **Chain-of-thought** | Ask model to reason before classifying | "First explain why, then output JSON" (tried but added latency without improving accuracy significantly) |
| **Structured output** | Force JSON format | "Output only valid JSON: {primary, secondary, tertiary}" |
| **Negative instructions** | Tell model what NOT to do | "Do not create new intent categories. Use only the provided list." |
| **Confidence scores** | Ask model to self-assess | "Include confidence 0.0–1.0 for each prediction" |

### Chain-of-Thought for Classification (Tried, Partially Adopted)

```
Prompt with CoT:
"Step 1: Identify the main topic of the call.
 Step 2: Determine if there are secondary issues.
 Step 3: Map to the closest intent category.
 Step 4: Output JSON."

Result: Improved accuracy by ~2% on ambiguous calls,
        but increased latency by ~40% and token usage by ~3x.
        
Decision: Used CoT only for the teacher during training data generation,
          not for the student in production (cost/latency trade-off).
```

---

## 4.5 Snowflake Architecture & Cortex

### Snowflake Architecture Basics

```
┌──────────────────────────────────────────────────────┐
│              SNOWFLAKE ARCHITECTURE                    │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  CLOUD SERVICES LAYER                             │ │
│  │  • Authentication, metadata, query optimization   │ │
│  │  • Query parsing, access control                  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  COMPUTE LAYER (Virtual Warehouses)               │ │
│  │  • Independent, scalable compute clusters         │ │
│  │  • XS → 4XL, auto-suspend/resume                 │ │
│  │  • Multiple warehouses = no contention            │ │
│  │  • Cortex ML functions run on compute layer       │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  STORAGE LAYER                                    │ │
│  │  • Columnar, compressed, immutable micro-partitions│ │
│  │  • Automatic clustering                           │ │
│  │  • Time travel (up to 90 days)                    │ │
│  │  • S3/Azure Blob/GCS under the hood              │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Key Snowflake concepts for interviews:**
- **Stages:** Landing zones for data ingestion (internal or external like S3)
- **Warehouses:** Independent compute resources; can scale up (bigger) or out (multi-cluster)
- **Separation of Storage and Compute:** Scale independently; pay for each separately
- **Micro-partitions:** Immutable columnar storage units (~50–500MB compressed)
- **Zero-copy Cloning:** Instantly clone tables/databases without copying data
- **Cortex Functions:** SQL-callable LLM/ML functions that run on Snowflake compute

---

## 4.6 Multi-Intent Classification Approaches

| Approach | Description | Pros | Cons |
|---|---|---|---|
| **Binary Relevance** | Train N independent binary classifiers (one per intent) | Simple, parallelizable | Ignores label correlations |
| **Classifier Chains** | Chain of binary classifiers; each uses previous predictions as features | Captures dependencies | Order-dependent |
| **Label Powerset** | Treat each unique label set as a single class | Captures full correlations | Exponential class space |
| **Seq2Seq / LLM** | Generate the set of labels as text | Flexible, handles any combination | Harder to evaluate, may hallucinate |
| **Our approach** | LLM generates JSON with all intents | Natural multi-label via generation | Requires structured output enforcement |

---

## 4.7 Text Summarization: Extractive vs Abstractive

| Aspect | Extractive | Abstractive |
|---|---|---|
| **Method** | Selects important sentences verbatim from text | Generates new sentences that capture the meaning |
| **Output** | Subset of original sentences | Novel phrasing, may paraphrase |
| **Models** | TextRank, BERT-extractive, LexRank | T5, BART, GPT, LLM-based |
| **Faithfulness** | High (exact quotes) | Risk of hallucination |
| **Fluency** | Can be choppy (sentence fragments) | Natural, coherent |
| **Our choice** | — | **Abstractive** (Cortex SUMMARIZE) — produces concise, natural summaries better suited for downstream LLM classification |

---

## 4.8 Data Drift Monitoring for NLP Models

```
┌──────────────────────────────────────────────────────────────┐
│              DRIFT MONITORING STRATEGY                         │
│                                                                │
│  1. LABEL DISTRIBUTION DRIFT                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Track: distribution of predicted intents over time       │ │
│  │  Metric: Jensen-Shannon divergence vs. baseline month     │ │
│  │  Alert: if JS divergence > threshold (e.g., 0.1)         │ │
│  │                                                           │ │
│  │  Example: If "CANCELLATION" jumps from 15% to 40%,       │ │
│  │  it could be real (disruption) or drift (model broken).   │ │
│  │  Cross-reference with actual cancellation rates.          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  2. CONFIDENCE SCORE MONITORING                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Track: average confidence score per day/week             │ │
│  │  Alert: if mean confidence drops below threshold          │ │
│  │  Interpretation: model encountering unfamiliar inputs     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  3. INPUT DRIFT (EMBEDDING-BASED)                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Track: embedding distribution of incoming summaries      │ │
│  │  Method: Compute centroid of training data embeddings     │ │
│  │          Measure drift of new data from centroid          │ │
│  │  Alert: if cosine distance to centroid increases          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  4. HUMAN AUDIT LOOP                                           │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Weekly: SMEs review random sample of 200 predictions     │ │
│  │  Focus on: low-confidence predictions, new edge cases     │ │
│  │  Feed back: corrections → retrain student quarterly       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  5. TEACHER AUDIT                                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Monthly: Run teacher model on 500 random transcripts     │ │
│  │  Compare: student vs teacher agreement rate               │ │
│  │  Alert: if agreement drops below 85%                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 4.9 Model Distillation in Production: Latency vs Accuracy Trade-offs

| Factor | Optimize for Latency | Optimize for Accuracy | Our Choice |
|---|---|---|---|
| **Model size** | Smaller (7B) | Larger (70B) | 7B student for daily, 70B teacher for audits |
| **Quantization** | INT4/INT8 | FP16/FP32 | INT8 for student in production |
| **Batch size** | Large (throughput) | Small (latency per request) | Large batch — daily batch job, latency not critical per-request |
| **Context length** | Shorter summaries | Full transcripts | Short summaries (via Cortex SUMMARIZE) |
| **Few-shot examples** | 0–2 (shorter prompt) | 5–8 (more context) | 0 for student (fine-tuned), 5–8 for teacher |
| **Decoding** | Greedy (T=0) | Beam search / sampling | Greedy for both (classification task) |

---

# 5. Interview Questions (30+) with Model Answers

---

## Q1: "Walk me through this project end to end."

> **Answer:**
> "At Jet2, our contact centre handled millions of calls annually, generating massive volumes of transcripts that needed to be categorized by intent for operational decision-making. Manual review was slow and inconsistent.
>
> I built an end-to-end automated classification system. First, I worked with product managers and domain experts to consolidate over 200 raw intent labels into a clean, hierarchical taxonomy — primary, secondary, and tertiary intents. Think of it as: Level 1 is 'Booking Change,' Level 2 is 'Flight Amendment,' Level 3 is 'Date Change.'
>
> The data pipeline ran inside Snowflake. Raw transcripts were cleaned, then summarized using Snowflake Cortex's built-in SUMMARIZE function — this reduced each transcript from thousands of tokens to a concise 2–3 sentence summary, which dramatically reduced classification cost and improved signal quality.
>
> For classification, I used a two-stage approach. This is a multi-label problem — a single call routinely carries more than one intent — so the model emits a set of labels, not a single class. First, a 70B parameter teacher model with few-shot prompting: I provided 5–8 labeled examples in the prompt and iterated through about 8 prompt versions based on SME feedback, taking recall from around 72% on the first draft up to the level that became our quality ceiling.
>
> But the 70B model was too expensive for daily production use on thousands of transcripts. So I applied knowledge distillation: I ran the teacher on a stratified sample of about 50,000 transcripts, filtered hard for high-confidence, schema-valid predictions, and used those roughly 40,000 surviving examples to fine-tune a 7B student via LoRA. The student hit **94% recall on validated intent labels** — measured against SME-labelled gold data, not just teacher agreement — at roughly 10x lower cost and 10x lower latency.
>
> The student model ran daily in production, with monitoring for label distribution drift, confidence degradation, and periodic teacher audits. Results fed into operational dashboards that contact centre managers used to allocate staff, identify training needs, and spot emerging issues in real time."

---

## Q2: "Why did you use few-shot learning instead of fine-tuning for the teacher?"

> **Answer:**
> "There were three main reasons. First, iteration speed — with few-shot, I could change intent definitions, add examples, or restructure the prompt in minutes. Fine-tuning would require hours of GPU training for every change. We went through about 8 major prompt iterations; that would have been 8 training runs.
>
> Second, we were still finalizing the intent taxonomy. Few-shot let us experiment with different category definitions without committing to expensive retraining.
>
> Third, data availability. At the start, we didn't have a large labeled dataset — that's exactly what the teacher was generating. You can't fine-tune without labels, and we needed the few-shot teacher to create those labels.
>
> Once we had stable teacher outputs, we fine-tuned the student for production because: (a) the student needed to be fast and cheap, and (b) the few-shot examples added ~500 tokens to every prompt, which multiplied by thousands of daily calls is significant extra cost."

---

## Q3: "Explain teacher-student distillation in the context of your project."

> **Answer:**
> "Knowledge distillation is about transferring the capabilities of a large, expensive model to a smaller, production-friendly one.
>
> In our case, the teacher was a 70B parameter LLM that we used with few-shot prompting. It was the most accurate thing we had — it set the quality ceiling — but too expensive and slow for daily production on thousands of transcripts.
>
> For distillation, we ran the teacher on about 50,000 diverse transcripts and collected its outputs — the predicted intents in JSON format along with confidence scores. We filtered for high-confidence predictions above 0.85, giving us ~40,000 high-quality training examples.
>
> Then we took a 7B base model and fine-tuned it using supervised fine-tuning (LoRA) on these teacher-generated input-output pairs. The student learned to map call summaries to intent labels by imitating the teacher's behavior.
>
> This is technically 'sequence-level distillation' — the student matches the teacher's text output, not its internal probability distributions. We didn't need access to the teacher's logits, which was important since we used Cortex for inference.
>
> The result: the 7B student achieved **94% recall on validated intent labels** — a small margin below the teacher — at roughly 10x lower cost and 10x lower latency."

---

## Q4: "How did you validate the student model against the teacher?"

> **Answer:**
> "We used a multi-layered validation approach.
>
> First, we held out 5,000 transcripts that the teacher had classified but that weren't included in the student's training data. We compared the student's predictions against the teacher's on this set — measuring set-based recall, per-label precision and recall, and generating confusion matrices. That tells you the distillation worked; it does not tell you the teacher was right.
>
> Second — and this is where the 94% comes from — we had a separate set of ~1,000 transcripts with human gold labels from our domain experts. We evaluated both the teacher and student against these gold labels to ensure we weren't just measuring teacher-student agreement but actual correctness.
>
> Third, we did qualitative error analysis. I sampled 200 cases where the student and teacher disagreed, categorized the disagreements, and found that most were on genuinely ambiguous calls where even human annotators had low agreement. The student wasn't making systematic errors — it was struggling in the same areas the teacher struggled.
>
> Finally, we tracked per-intent metrics. The student was strong on high-frequency intents like BOOKING_CHANGE and CANCELLATION (>90% recall) and weaker on rare intents like loyalty/rewards queries (~75%), which makes sense given fewer training examples."

---

## Q5: "Why recall on the validated label set as your primary metric?"

> **Answer:**
> "Two reasons: the nature of the task and the business need.
>
> First, this is a multi-label problem — each call can have multiple intents. Traditional accuracy requires an exact match of the entire label set, which is too strict. If a call has 4 intents and we get 3 right, exact match gives 0% credit. A set-based, Jaccard-style formulation gives partial credit proportional to overlap.
>
> Second, recall matters more than precision operationally. If we miss an intent, the contact centre might not route a concern to the right team. If we over-predict an intent, the worst case is an unnecessary review — much less costly than missing a complaint or a cancellation request.
>
> So the metric is: of the true intents, what fraction did we capture? At 94%, we're recovering roughly nineteen out of every twenty true intent labels. I always report it with two guardrails — Jaccard similarity, which penalises over-prediction, and the average number of labels we emit per call — because recall on its own can be gamed by predicting everything."

---

## Q6: "How did you handle ambiguous intents?"

> **Answer:**
> "Ambiguity was one of the biggest challenges. A customer might start asking about a refund, then pivot to rescheduling — is the primary intent cancellation or booking change?
>
> We handled this at multiple levels. In taxonomy design, we allowed multi-label assignment — a call can legitimately be CANCELLATION + BOOKING_CHANGE. In prompt engineering, we included explicit examples of ambiguous calls and showed the model how to handle them — 'when a call discusses both X and Y, classify as both.'
>
> For cases where even the model was uncertain, we used the confidence score. Predictions below 0.7 confidence were flagged for human review. In practice, about 10–15% of calls fell into this low-confidence bucket, which was manageable for the team.
>
> We also discovered that many 'ambiguous' calls weren't truly ambiguous — they just had multiple intents that we hadn't anticipated. This feedback loop led us to add new intent categories in taxonomy v2 and v3."

---

## Q7: "What if a call has no clear intent?"

> **Answer:**
> "We handled this explicitly in the taxonomy with an 'OTHER/UNCLASSIFIABLE' category, and in the prompt with instructions like: 'If the call does not clearly match any intent, classify as OTHER and explain why in the reasoning field.'
>
> In practice, about 3–5% of calls fell into this bucket. These were typically: very short calls where the customer hung up quickly, wrong-number calls, or highly unusual requests outside our taxonomy.
>
> We monitored the rate of OTHER classifications over time. A spike might indicate that a new type of call was emerging that our taxonomy didn't cover — which happened once when a new booking policy launched and generated a flood of unfamiliar queries. We quickly added the relevant intent category."

---

## Q8: "How did you scale to 1M+ transcripts?"

> **Answer:**
> "Scalability came from three design decisions.
>
> First, everything ran inside Snowflake. Cortex functions process data in parallel across warehouse nodes. We used a Medium to Large warehouse for classification jobs, which could handle thousands of transcripts per hour.
>
> Second, the summarization step was critical for scale. Classifying a 3-sentence summary is ~50x cheaper than classifying a 10,000-token raw transcript. This made the 70B teacher feasible for generating training data and the 7B student very efficient in production.
>
> Third, the pipeline was fully automated via Snowflake Tasks — a scheduled job that ran nightly, picking up new transcripts, summarizing, classifying, and loading results into the production table. No manual intervention needed.
>
> The student model handled the daily volume (2,000–5,000 transcripts) in under 2 hours of batch processing. The historical backfill of 1M+ transcripts was done over a few weeks using the teacher model on a larger warehouse."

---

## Q9: "What would you improve if you could redo this project?"

> **Answer:**
> "Three things. First, I'd invest more in active learning. Instead of randomly sampling transcripts for the teacher, I'd use the student's confidence scores to prioritize uncertain or disagreement cases — this would make the teacher's labeling budget go further.
>
> Second, I'd explore structured fine-tuning approaches like RLHF or DPO with human preferences on the student model. Our SMEs could rank model outputs rather than just provide labels, which could improve quality on ambiguous cases.
>
> Third, I'd build a more sophisticated drift detection system. We tracked label distributions and confidence scores, but I'd add embedding-based drift detection and automated retraining triggers. Currently, retraining was manual and quarterly — I'd make it continuous."

---

## Q10: "How did you handle prompt engineering iterations?"

> **Answer:**
> "Very systematically. Each prompt version was documented with: the exact prompt text, the intent definitions, the few-shot examples, and the evaluation results.
>
> The process was: (1) Draft a prompt version. (2) Run it on a fixed evaluation set of 500 transcripts. (3) Measure set-based recall and per-label metrics. (4) Have 2–3 SMEs review the errors — they'd annotate whether each error was a model mistake or an ambiguous case. (5) Identify patterns — which intents are confused, which examples are misleading. (6) Refine the prompt — update definitions, swap examples, add constraints. (7) Repeat.
>
> Some key discoveries: Adding explicit intent definitions improved recall by ~5%. Including one 'hard negative' example per intent pair that was commonly confused reduced those specific confusions by ~40%. And forcing JSON output format eliminated ~3% of responses that were valid classifications but unparseable."

---

## Q11: "Walk me through the cost analysis — teacher vs student in production."

> **Answer:**
> "Let me break this down concretely. With 5,000 transcripts per day:
>
> **Teacher (70B):** Each classification uses roughly ~800 input tokens (prompt + summary + few-shot examples) and ~100 output tokens. At typical LLM API rates, that's roughly $0.01–0.02 per call. At 5,000 calls/day × 365 days, that's ~$18,000–$36,000/year just for classification, plus summarization costs.
>
> **Student (7B, fine-tuned):** No few-shot examples needed (knowledge baked in), so input is ~200 tokens (just the summary + short instruction) and ~100 output tokens. Cost is roughly $0.001–0.002 per call. Annual: ~$1,800–$3,600.
>
> That's a **10x cost reduction**, and that's before accounting for latency — the student processes calls ~10x faster, which means we can use a smaller Snowflake warehouse and save on compute costs too.
>
> We still used the teacher for monthly audits (~500 calls/month) and quarterly retraining data generation (~5,000 calls/quarter), which added ~$500–$1,000/year. Total system cost was a fraction of what teacher-only would have been."

---

## Q12: "How did you monitor for drift in NLP?"

> **Answer:**
> "We tracked drift at multiple levels.
>
> Label distribution drift: We compared the weekly distribution of predicted intents against a baseline period using Jensen-Shannon divergence. A significant shift could mean either real change (e.g., holiday disruption causing cancellation spike) or model degradation. We cross-referenced with business metrics to distinguish.
>
> Confidence drift: We monitored average prediction confidence over time. A gradual decline suggests the model is encountering inputs it wasn't trained on — perhaps new products, policies, or customer language patterns.
>
> Teacher-student agreement: Monthly, we ran the teacher on 500 random transcripts and checked agreement with the student. If agreement dropped below 85%, it triggered a review.
>
> Human audit: Weekly random sample of 200 predictions reviewed by domain experts. Error rate was tracked over time.
>
> We set up alerts in our monitoring dashboard for each of these. In practice, we detected one significant drift event — after a new booking policy launched, confidence scores dropped and OTHER classifications spiked. We quickly added new intent categories and refreshed the student model."

---

## Q13: "What's the difference between Jaccard similarity and set-based recall?"

> **Answer:**
> "Jaccard similarity measures symmetric overlap: |intersection| / |union|. It penalizes both missing labels (false negatives) and extra labels (false positives) equally.
>
> Set-based recall measures: |intersection| / |true set size|. It only cares about coverage — did we capture the true intents? It doesn't penalize over-prediction.
>
> We chose recall as the headline because operationally, missing an intent is worse than flagging an extra one. If we miss a complaint, it goes unaddressed. If we over-predict a complaint, the worst case is unnecessary review.
>
> But I report Jaccard similarity alongside it precisely because it's the symmetric one — it's the number that catches you if you've quietly bought recall by over-predicting. Ours sits meaningfully below the 94%, and that gap *is* our over-prediction rate. I'd rather show that gap than hide it, because the over-predictions were mostly on adjacent intents within the same branch of the taxonomy, which is a cheap error operationally."

---

## Q14: "How does Snowflake Cortex compare to using an external LLM API?"

> **Answer:**
> "Three main advantages of Cortex for our use case.
>
> First, data governance. Call transcripts contain PII — names, booking references, sometimes card details. With Cortex, data never leaves Snowflake's security perimeter. Using an external API would require data egress, PII masking, and additional compliance review under GDPR.
>
> Second, operational simplicity. Everything runs as SQL queries. Our data engineers could maintain and extend the pipeline without ML infrastructure expertise. No separate model hosting, no API keys, no rate limiting.
>
> Third, cost model. Cortex uses Snowflake's serverless compute — you pay for what you use, no idle GPU costs. For our batch processing pattern (process 5K transcripts at night, idle during the day), this was significantly cheaper than maintaining dedicated GPU instances.
>
> The trade-off: Cortex offers a fixed set of models and you have less control over fine-tuning. For the student model, we needed more control, so we used separate infrastructure for training but could potentially serve the student through Cortex as well."

---

## Q15: "Tell me about a time the model failed and how you fixed it."

> **Answer:**
> "One notable failure was with 'transfer to another department' calls. About 5% of calls involved a customer being transferred — the LLM was classifying based on the transferring agent's brief notes rather than the customer's actual intent.
>
> The symptom: high confusion between INFORMATION and BOOKING_CHANGE for transferred calls. The summary captured 'Agent explained they'd transfer to booking amendments team' but missed the customer's original request.
>
> The fix was two-fold. First, in the summarization prompt, I added an instruction to focus on the customer's stated needs, not the agent's actions. Second, in the classification prompt, I added an example showing a transferred call and how to classify based on customer intent, not routing action.
>
> This improved the problem category from ~60% accuracy to ~85% accuracy, and overall recall on the validated label set improved by about 2 points."

---

## Q16: "How did you handle class imbalance in the intent taxonomy?"

> **Answer:**
> "Our intent distribution was heavily skewed — BOOKING_CHANGE and INFORMATION accounted for ~50% of calls, while categories like LOYALTY or SPECIAL_ASSISTANCE were under 3%.
>
> For the teacher (few-shot), imbalance is less of an issue since the model sees examples of every category in the prompt. But we ensured rare categories had at least one clear example.
>
> For the student (fine-tuning), we used several strategies: (1) Stratified sampling to ensure rare categories were proportionally represented in training data. (2) Over-sampling rare categories by ~2–3x. (3) Evaluating per-intent metrics separately — we didn't let high performance on common intents mask poor performance on rare ones.
>
> Despite this, rare intents still had lower recall (~75% vs ~93% for common intents). We accepted this trade-off but flagged rare-intent predictions for human review when confidence was below 0.8."

---

## Q17: "Why not just fine-tune a single model instead of teacher-student?"

> **Answer:**
> "We could have, but there's a bootstrapping problem. Fine-tuning requires labeled data, and we didn't have it at scale. Manual labeling of 50,000 transcripts with a three-level intent taxonomy would take months of expert time.
>
> The teacher-student approach solved this: the teacher (few-shot, no training data needed) generated labels at scale, and the student used those labels for fine-tuning. It's a form of automated data labeling.
>
> Additionally, the teacher gave us a quality ceiling to aim for and a validation reference. We could measure how close the student got to the teacher, and we could audit by comparing their outputs on the same inputs.
>
> If we had started with a large manually labeled dataset, direct fine-tuning of a 7B model would have been simpler. But given our starting point of zero labels, teacher-student was the right architecture."

---

## Q18: "How do you ensure the model outputs valid JSON?"

> **Answer:**
> "Multiple strategies. In the prompt, I included explicit formatting instructions and examples of valid JSON output. I also added negative instructions: 'Do not include explanatory text outside the JSON block.'
>
> For the student (fine-tuned), the model learned the output format from training data, so JSON validity was high (~98% of responses were parseable).
>
> For the remaining ~2%, we had a fallback parser: regex extraction of key fields, and if that failed, the transcript was flagged for reprocessing with a slightly different prompt (adding 'Output ONLY valid JSON, nothing else' as a prefix).
>
> In more recent systems, I'd use constrained decoding or tool-calling / function-calling APIs that guarantee structured output. But at the time, prompt engineering + post-processing was effective enough."

---

## Q19: "What's your understanding of how tokenization affects this pipeline?"

> **Answer:**
> "Tokenization matters in two places. First, context window utilization — a raw transcript might be 5,000–10,000 tokens depending on the tokenizer. If the context window is 4K (Llama 2), that transcript doesn't even fit. Our summarization step compressed transcripts to ~50–150 tokens, making context windows a non-issue.
>
> Second, cost — LLM inference is priced per token. The summarization step reduced classification input tokens by ~50x, which directly translates to cost savings.
>
> Third, the few-shot examples in the teacher prompt consumed ~500 tokens. For the student (fine-tuned), no examples were needed, saving those tokens per inference.
>
> A subtlety: travel-specific terminology and booking codes (like 'PNR', specific airport codes) might be poorly tokenized by general-purpose tokenizers, potentially splitting them into subword fragments. This was another reason summarization helped — the summary used natural language rather than codes."

---

## Q20: "How would you extend this system to real-time classification?"

> **Answer:**
> "Currently, the system runs as a nightly batch job. For real-time:
>
> First, the student model would need to be served as a low-latency API — deployed on a GPU instance with optimized inference (vLLM, TensorRT-LLM, or similar). At ~1 second per call with the 7B model, real-time is feasible.
>
> Second, the summarization step would need to be moved inline — either a lighter extractive summarization or a very efficient small model, since Cortex SUMMARIZE has some latency.
>
> Third, I'd add a streaming pipeline — instead of batch SQL queries, use Snowflake's Snowpipe or a Kafka/Kinesis stream to process transcripts as they arrive.
>
> Fourth, for true real-time (during the call), we'd need streaming ASR and incremental classification — classifying intent as the conversation progresses. This is a harder problem but valuable for routing calls mid-conversation."

---

## Q21: "How did the dashboard integration work?"

> **Answer:**
> "The classified transcripts landed in a Snowflake production table with columns for call_id, timestamp, agent_id, primary/secondary/tertiary intent, confidence score, and the original summary.
>
> Contact centre managers had Tableau dashboards connected to these tables via Snowflake's native Tableau connector. Dashboards showed: real-time intent distribution, trending intents over time, agent performance by intent category, and anomaly alerts when unusual patterns emerged.
>
> One powerful use case: during flight disruptions, the CANCELLATION intent would spike. Managers could see this in near-real-time and reallocate agents from SALES to CANCELLATION teams before customer wait times increased. Before our system, this reallocation happened hours later based on manual observation."

---

## Q22: "How did you handle PII in the transcripts?"

> **Answer:**
> "PII handling was multi-layered. First, raw transcripts were processed through a PII detection and masking pipeline before summarization — names replaced with [CUSTOMER], card numbers with [CARD_REDACTED], booking references with [BOOKING_REF].
>
> Second, Snowflake's data governance features — row-level security, masking policies, and role-based access control — ensured that only authorized roles could access raw transcripts. The classification pipeline operated on summarized, PII-masked versions.
>
> Third, by keeping everything in Snowflake via Cortex, we avoided data egress — the transcripts never left the platform, which simplified GDPR compliance significantly. This was actually one of the strongest arguments for Cortex over external APIs."

---

## Q23: "Why not just use the 70B? Compute is cheap now."

> **Answer:**
> "Compute is cheaper than it was, but the economics of this particular workload didn't change shape. It's a *recurring* batch — thousands of calls, every night, indefinitely. The 70B was roughly ten to fifteen times the cost per inference and about ten times the latency, and at seasonal peak the batch wouldn't have finished inside its window. That last one isn't a budget argument, it's a correctness argument: forecasts and dashboards that land after the morning ops meeting are worthless.
>
> The other thing people miss is the prompt overhead. The teacher needed five to eight few-shot examples in every single call's prompt — about five hundred tokens of instructions that you pay for a million times over. The distilled student has that behaviour in its weights, so its prompt is just the summary.
>
> And I didn't throw the teacher away. It stayed in the loop for weekly quality audits, for labelling new intent categories when the taxonomy expanded, and for refreshing the student's training data quarterly. So the answer isn't 'the 70B was bad' — it's 'the 70B was a labelling instrument, and the 7B was the product.'"

---

## Q24: "How do you know the student didn't collapse on rare intents?"

> **Answer:**
> "Because I never looked at the aggregate alone — that's exactly the failure mode a single headline number hides. A model can post 94% recall while a rare intent has quietly gone to zero, because the rare intent contributes almost nothing to the average.
>
> Three specific defences. First, **evaluation**: I reported per-label recall for every intent in the taxonomy, plus a head-versus-tail split, and treated any label with a material drop as a release blocker regardless of what the aggregate said. The tail was genuinely weaker — rare intents like loyalty queries sat well below the common ones — and I stated that rather than burying it.
>
> Second, **training data**: the teacher-labelling sample was stratified with deliberate oversampling of rare categories. A proportional sample would have given the student a handful of examples of special assistance, which is not enough to learn anything.
>
> Third, **the loss and the thresholds**: per-label positive weighting in the BCE so rare labels contributed meaningfully to the gradient, and per-label thresholds set lower for rare, high-consequence intents. If a rare label had collapsed, its threshold sweep would have shown no threshold achieving acceptable recall at any precision — that's the diagnostic."

---

## Q25: "What temperature did you use, and why that value?"

> **Answer:**
> "I should be precise here: for the production student I used sequence-level distillation, where the student trains on the teacher's generated text, so there's no softmax temperature on the KD loss in the classic sense. The temperature that mattered operationally was the *decoding* temperature — near zero, effectively greedy, for both models, because this is a classification task and you want determinism, not creativity. Same input, same output, every run. That matters for auditability as much as accuracy.
>
> Where I did use non-zero decoding temperature was quality control: I re-ran a subset of teacher labels at moderate temperature to test self-consistency. If the teacher gave you three different intent sets across three samples, that call was genuinely ambiguous and I didn't want it as training data presented as ground truth.
>
> On the classic KD temperature — if I'd had logit access and were doing true soft-target matching, I'd have started around three and tuned it. The reasoning: at $T=1$ the teacher's distribution is so peaked that you've learned nothing beyond the hard label, and you've re-derived ordinary supervised fine-tuning. Push it to ten and the distribution flattens toward uniform, so the student burns capacity fitting noise in the tail. Two to four is the range where the runner-up ordering — the 'dark knowledge' — is visible without drowning the signal. The tell-tale that you've gone too high is rare-label precision degrading while aggregate recall still looks healthy."

---

## Q26: "How did you set the per-label thresholds?"

> **Answer:**
> "On the validation set, never the test set — otherwise you've fitted your thresholds to the number you're about to report.
>
> The procedure per label: collect predicted probabilities and true labels, sweep candidate thresholds across the range, compute precision and recall at each, and pick the *lowest* threshold that still meets that label's precision floor. Lowest-threshold-meeting-the-floor maximises recall subject to the precision constraint, which is exactly the shape of the business problem.
>
> The interesting part is that the precision floor is different per label, and it comes from the operations team, not from me. Complaint and cancellation get a low threshold — missing one means an unaddressed customer issue, and a false positive costs one wasted review. Special assistance gets a low threshold for duty-of-care reasons. High-frequency, low-consequence labels like general information get a higher threshold, because a loose threshold there floods every downstream dashboard and dilutes the counts everyone's reading. And the OTHER bucket gets the highest, because if you let it be loose it absorbs everything and the taxonomy stops meaning anything.
>
> A single global 0.5 is the most common mistake in multi-label classification. It implicitly assumes every label has the same base rate and the same cost of error, and neither is true here."

---

## Q27: "What's the difference between response-based, feature-based and relation-based knowledge distillation?"

> **Answer:**
> "They differ in *what* the student is asked to match.
>
> **Response-based** — the original Hinton formulation — matches the teacher's output distribution. The student's loss is a KL divergence against the teacher's softened logits, blended with ordinary cross-entropy on the hard labels. The insight is that the relative probabilities of the *wrong* classes carry information the hard label throws away.
>
> **Feature-based** — FitNets and its descendants — matches intermediate hidden representations. You pick a teacher layer and a student layer, project them into a common space, and penalise the distance between activations. It transfers *how* the teacher represents the input, not just what it concludes. The catch is that you need white-box access and a sensible layer correspondence, which is awkward when the architectures differ in depth.
>
> **Relation-based** — relational KD — matches the *structure* between samples rather than any individual output. You take a batch, compute pairwise distances or angles in the teacher's embedding space, and train the student to reproduce that geometry. It's the natural choice for metric learning and retrieval, where what you care about is the relationships, not the absolute positions.
>
> I used sequence-level distillation, which is response-based applied to a generative model: the student matches the teacher's emitted text rather than its logits. That was forced by the deployment — the teacher ran through Snowflake Cortex, which returns generated text, not per-token distributions over a 32K vocabulary. Feature-based and relation-based were both off the table for the same reason: no white-box access."

---

## Q28: "How do you know your teacher labels weren't just propagating the teacher's bias into the student?"

> **Answer:**
> "I can't rule it out entirely, and I'd be suspicious of anyone who claimed they could. What I can do is bound it and detect the cases that matter.
>
> The core defence is that the headline metric is measured against **human gold labels, not teacher agreement**. Those are two genuinely different questions. Student-versus-teacher tells you the distillation worked; student-versus-SME tells you the answer is right. If the teacher had a systematic bias and the student faithfully learned it, student-teacher agreement would look excellent and the SME number would drop — and the SME number is the one on the resume.
>
> On top of that: SMEs spot-audited the *teacher's* labels directly, stratified toward rare intents and low-confidence cases, so bias in the training data had a chance of being caught before it reached the student. And where the teacher was self-inconsistent across repeated samples, we dropped the example rather than baking in an arbitrary answer.
>
> The residual risk I'd own honestly: our SMEs and our few-shot examples came from the same organisation and the same taxonomy definitions. If the *taxonomy itself* encoded a blind spot — a category of customer problem we hadn't thought to name — neither the teacher nor the SME validation would catch it. That's why we monitored the OTHER rate and the low-confidence rate over time; a rising OTHER rate is the signal that reality has outgrown your label space."

---

## Q29: "Walk me through the actual training setup for the student."

> **Answer:**
> "Base model was a 7B open-weight model in the Mistral-7B / Llama-2-7B class. Training was parameter-efficient — LoRA rather than full fine-tuning — with rank in the 16 to 64 range and alpha roughly double the rank. Learning rate around 1e-4 to 2e-5 with a warmup and cosine decay, three to five epochs, effective batch size in the 8 to 16 range using gradient accumulation, and a 10% validation split held out from the training pairs.
>
> Why LoRA over full fine-tuning: with roughly 40,000 training examples, full fine-tuning of a 7B model is both unnecessary and actively risky — you get catastrophic forgetting of the general language ability that makes the model robust to ASR noise and unusual phrasing. LoRA touches a small fraction of the parameters, trains on a single accelerator, and produces adapter weights that are a few tens of megabytes rather than a 14GB checkpoint. That last point matters more than people expect operationally: you can version and roll back adapters cheaply.
>
> Data format was straightforward: input is the Cortex-generated call summary plus a short instruction, output is the JSON intent object. No few-shot examples in the training prompts, deliberately — the whole point was to move that knowledge into the weights so we'd never pay for those tokens at inference.
>
> Early stopping was on validation recall rather than validation loss, because loss and the metric we cared about weren't perfectly aligned."

---

## Q30: "Did the student ever beat the teacher? Should it?"

> **Answer:**
> "Ours didn't, and I'd have wanted to explain it carefully if it had.
>
> A student *can* legitimately beat its teacher, and there's real literature on this. The mechanisms: the confidence-based filtering means the student trains on a cleaner, less noisy version of the teacher's output distribution — you've removed the teacher's own low-margin guesses. Distillation also acts as a regulariser, so the smaller model can generalise better on data it hasn't seen. And where we mixed in SME gold labels alongside teacher labels, the student had access to a signal the teacher never saw.
>
> But if I'd observed a student beating a 10x-larger teacher by a wide margin, my first hypothesis would be a leak — training data contaminating the evaluation set — not a breakthrough. I'd check the hold-out construction before I celebrated.
>
> What we actually saw is the expected pattern: the student sits a small margin below the teacher, and the gap concentrates almost entirely on rare intents and genuinely ambiguous multi-intent calls. That's the signature of a distillation that worked properly — the student learned the bulk of the decision boundary and lost a little resolution in the tail."

---

# Trick & Follow-Up Questions

These are the adversarial probes — the questions designed to find out whether you actually did the work or memorised a bullet point. Answer them with specifics and be visibly comfortable naming trade-offs.

---

### Trick Q1: "94% recall sounds high — what was your precision, and did you trade it away?"

> **Answer:**
> "Yes, deliberately, and I want to be straight about the number rather than invent one. Precision was materially lower than recall — that's the direct consequence of tuning per-label thresholds downward to maximise coverage. If I had to characterise it I'd say we were comfortably over-predicting, and the honest way to see it is the gap between the recall figure and the Jaccard similarity, which is symmetric and penalises extra labels. That gap *is* the over-prediction.
>
> The reason I'm comfortable with the trade is the cost asymmetry. A false negative means a customer's complaint never reaches the team that could resolve it, and we don't find out until it escalates — unbounded cost, external to us. A false positive means one extra row in a review queue that someone dismisses in thirty seconds — bounded, small, internal. When the costs differ by that much, optimising a symmetric metric like F1 would be the wrong call; it would implicitly price those two errors the same.
>
> Two caveats I'd add unprompted. First, the trade isn't uniform — high-frequency labels kept higher thresholds precisely because loose predictions there would flood the dashboards. Second, over-prediction has a ceiling: I monitored average labels emitted per call against the human labellers' average, because that's the metric that catches you if recall is being bought with spray-and-pray."

---

### Trick Q2: "Multi-label with 1M transcripts — how did you handle label imbalance and rare intents?"

> **Answer:**
> "At four levels, because no single fix is sufficient.
>
> **Sampling:** the 50,000 transcripts I sent to the teacher weren't a random draw. They were stratified, with rare categories deliberately oversampled relative to their natural frequency. Proportional sampling would have given the student a couple of dozen examples of the tail intents, which is not a learnable signal.
>
> **Loss:** per-label positive weighting in the binary cross-entropy, roughly proportional to the negative-to-positive ratio per label, capped — uncapped weights on ultra-rare labels destabilise training badly. Focal loss was the alternative I'd consider, down-weighting the easy majority examples so capacity goes to the hard ones.
>
> **Thresholds:** rare intents got lower decision thresholds. This is the highest-leverage intervention and the cheapest, because it's post-hoc — you can retune thresholds without retraining anything.
>
> **Evaluation:** per-label metrics always, never aggregate alone, with head-versus-tail reported separately.
>
> And the honest outcome: rare intents still underperformed. Loyalty and special-assistance queries had noticeably lower recall than booking changes. We handled the residual operationally — rare-intent predictions below a confidence bar were routed to human review rather than being silently trusted."

---

### Trick Q3: "You said 1M+ transcripts but only 50K teacher labels. Isn't the 1M number misleading?"

> **Answer:**
> "They're two different things and I should be explicit about both. The million-plus is *processed* volume — the student model ran over the full historical corpus and then over every new call, two to five thousand a day. That's the operational scale of the system.
>
> The 50,000 is the *labelled* subset the teacher was run on to create training data, of which roughly 40,000 survived quality filtering.
>
> Those aren't in tension — that's the whole design. Distillation is precisely the technique for getting a model that can economically process a million records without paying to label a million records. If I'd needed the teacher to label all million, the project wouldn't have had a budget.
>
> If someone thinks the resume bullet conflates them, I'd rather clarify it in the first thirty seconds of the answer than have them wondering."

---

### Trick Q4: "Your confidence filter kept only high-confidence teacher predictions. Didn't that make the task artificially easy for the student?"

> **Answer:**
> "That's the sharpest critique of the setup and yes, it's a real effect. Filtering to high-confidence teacher output systematically removes the ambiguous, multi-intent, edge-case calls — exactly the cases the student will meet in production. You end up with a clean training set that under-represents difficulty, and a student that's overconfident on the hard tail.
>
> Two mitigations. First, we kept a deliberately curated slice of hard cases in the training data with **SME labels instead of teacher labels** — so the student saw difficulty without inheriting the teacher's guesswork on it. Second, the evaluation set was *not* filtered by confidence. The hold-out was a representative sample including the ambiguous calls, so the 94% is measured on realistic difficulty even though training wasn't.
>
> What I'd do differently: active learning. Rather than a static confidence cut-off, I'd use the student's own uncertainty to decide which transcripts get the teacher's expensive attention — spend the labelling budget where the student is weakest rather than where the teacher is most confident. Those are almost opposite selection criteria, and the confidence filter is arguably selecting the least informative examples."

---

### Trick Q5: "If the student imitates the teacher, isn't your ceiling the teacher's error rate? How is 94% not just an upper bound you got lucky on?"

> **Answer:**
> "For pure imitation, yes, the teacher is approximately the ceiling — and our student does sit below it, which is the expected pattern.
>
> But 'approximately' is doing real work in that sentence, and there are three reasons a student isn't strictly bounded by its teacher. Confidence filtering means the student learns from a denoised version of the teacher's behaviour, not the teacher's full error distribution. Distillation regularises — the smaller hypothesis space can generalise better on unseen data. And in our case the training set wasn't purely teacher-labelled; the hard-case slice carried SME labels the teacher never produced.
>
> On 'got lucky': the defence is the evaluation design, not the number. The 94% is on a thousand SME-labelled transcripts that were never used in training, never seen by the teacher during prompt iteration, and adjudicated between annotators. And it held up in production over subsequent months rather than being a single favourable snapshot. If it were luck, the weekly teacher audits would have shown agreement decaying, and they didn't."

---

### Trick Q6: "Sequence-level distillation is just supervised fine-tuning on synthetic data. Why call it distillation?"

> **Answer:**
> "Mechanically you're right, and I'd concede that immediately — the training loop is standard SFT with a cross-entropy loss on next-token prediction. There's no KL term, no temperature, no logit matching. If someone wants to call it 'training on synthetic labels' I won't argue about the word.
>
> What makes 'distillation' the right frame is the *intent and the structure* of the setup: a deliberately larger, more capable model is used specifically to transfer its capability into a deliberately smaller one, with the small one replacing the large one in deployment. That's the defining property of knowledge distillation as a technique, and sequence-level KD is an established named method — Kim and Rush formalised it, and it's what Alpaca, Vicuna and Orca all do.
>
> Where I'd be careful is not overclaiming. If an interviewer is specifically probing whether I understand Hinton-style soft-target KD, the correct answer is that I know the formulation, I know why $T^2$ appears in the loss, and I know I *didn't* use it — because Cortex doesn't expose logits. Pretending otherwise would fall apart in one follow-up question."

---

### Trick Q7: "You measured against 1,000 SME-labelled transcripts. What was the inter-annotator agreement? If your humans disagree, what does 94% even mean?"

> **Answer:**
> "This is the right question and it constrains what any number can mean. Our estimate of manual inter-annotator agreement on this taxonomy was around 80% — meaning two trained humans given the same call assign meaningfully different label sets about one time in five.
>
> Two implications I'd own. First, there is a genuine ceiling here below 100%, and it's set by the ambiguity of the task, not the capability of the model. A call where the customer wants a refund and then pivots to rebooking has no single correct answer. Second, and this is the more interesting one: the model is *more consistent* than the process it replaced. It applies the same decision rule every time, which for downstream analytics is arguably more valuable than being marginally more 'correct' on any single call — you can trust the trend lines.
>
> On how we handled it methodologically: the gold set wasn't single-annotated. Disagreements were adjudicated, so the reference labels represent a resolved consensus rather than one person's opinion. That's why the gold set is only a thousand transcripts — adjudication is expensive. And where annotators couldn't reach consensus at all, we excluded the call rather than forcing a label, which I'd flag as a mild optimistic bias in the 94% that I'd rather name than hide."

---

### Trick Q8: "Six percent of intents are missed. Which six percent, and did anyone get hurt?"

> **Answer:**
> "The honest answer to 'which' is: disproportionately the tail. Misses concentrate on rare intents with few training examples and on calls carrying three or more simultaneous intents, where the third and fourth labels are the ones that drop. They are *not* uniformly distributed, and if they were I'd be more relaxed, not less.
>
> That's precisely why the aggregate wasn't the number I managed to. The failure mode that would actually cause harm is a low-frequency, high-consequence intent — special assistance, safeguarding-adjacent complaints — silently going to zero recall while the headline stays at 94%, because those labels are too rare to move the average. So those categories got the lowest thresholds in the system and were monitored per-label, weekly.
>
> On 'did anyone get hurt': the system didn't operate unsupervised on high-stakes decisions. Low-confidence predictions, roughly ten to fifteen percent of calls, routed to human review, and rare-intent predictions had a lower confidence bar for triggering that review. The model was a routing and analytics layer, not the last line of defence — the contact centre's existing escalation processes still ran underneath it. I'd be much more cautious about a 94% recall model that was the *only* thing standing between a customer and a missed complaint."

---

# 6. Potential Red Flags & How to Handle

---

## 6.1 "Did you actually train the student model on teacher outputs?"

**The concern:** Interviewers may wonder if you really did distillation or just used the teacher directly.

**How to address:**
> "Yes — the student was fine-tuned via LoRA on approximately 40,000 teacher-generated examples. The training data was (input: call summary, output: teacher's JSON classification). After fine-tuning, the student model ran independently — no teacher in the loop at inference time. We validated the student against both the teacher and human gold labels on held-out data. The student achieved 94% recall on validated intent labels, a small margin below the teacher, and we tracked this metric continuously."

**If pressed on details:**
- Mention LoRA rank (16–64), learning rate schedule, number of epochs (3–5)
- Explain that you used sequence-level distillation, not classic KD with logits
- Note that the training loop was standard SFT using libraries like Hugging Face TRL or Axolotl

---

## 6.2 "94% recall — is that too good, or not good enough?"

**The concern:** this cuts both ways. Some interviewers hear 94% and think "6% of intents are missed, that's a lot of unrouted complaints." Others hear it and suspect the metric was chosen to flatter.

**How to frame it:**

> "94% recall on validated intent labels means we correctly capture roughly nineteen out of every twenty true intents per call. For a multi-label task with a 30+ category taxonomy at three levels, that's strong — but the number only means something once you know how it was measured and what it cost.
>
> - It's measured against SME-labelled gold data, held out entirely, not against teacher agreement.
> - It's recall, deliberately. I traded precision for it, per label, because a missed complaint is unbounded cost and a spurious label is thirty seconds of review.
> - The residual errors concentrate on (a) rare intents with few training examples and (b) genuinely ambiguous calls where our own annotators disagreed with each other.
> - We handle the gap operationally: low-confidence predictions (~10–15% of calls) route to human review, so the end-to-end system with a human in the loop is better than the model alone.
> - The baseline matters — manual classification had estimated inter-annotator agreement around 80%. The system is more *consistent* than the process it replaced, not just faster."

**If they push on the 6% miss rate:** don't get defensive, get specific. "The right question isn't the aggregate — it's which 6%. If we were missing 6% uniformly at random I'd be relaxed. What I actually watched was the per-label breakdown, because the failure mode that would hurt is a rare, high-consequence intent like special assistance quietly dropping to zero recall while the headline stays at 94%. That's why thresholds were tuned per label rather than globally."

---

## 6.3 "Why few-shot for the teacher instead of fine-tuning a large model?"

**The concern:** Few-shot might seem like a shortcut.

**How to frame it:**

> "Few-shot was a deliberate choice, not a shortcut. We faced a cold-start problem — no labeled data existed at scale. Fine-tuning requires labels; the teacher's job was to create those labels.
>
> Few-shot also gave us rapid iteration — 8 prompt versions in weeks, not months of training. Once we had stable teacher outputs, we invested in fine-tuning the student for production.
>
> The teacher's few-shot performance validated that the approach works — it set a ceiling the student then came within a small margin of. If we'd had a large labeled dataset upfront, direct fine-tuning would have been reasonable, but that dataset didn't exist."

---

## 6.4 "How much of this was your work vs the team?"

**How to frame it:**

> "I was the primary data scientist on this project. Specifically, I owned: the intent taxonomy design (collaborating with product for domain knowledge), the Cortex pipeline setup, prompt engineering for the teacher, the distillation process for the student, evaluation methodology, and drift monitoring.
>
> I worked with a data engineer who helped with Snowflake infrastructure and pipeline orchestration, and product managers who provided domain expertise for intent definitions and SME validation. The dashboard was built by our BI team based on the data I produced."

---

## 6.5 "What if the ASR (speech-to-text) is noisy?"

**How to address:**
> "ASR noise was a real issue — roughly 5–8% word error rate on our transcripts. We handled this in two ways. First, the summarization step acted as a denoiser — the LLM could understand the gist even with some garbled words, and the summary was cleaner than the raw transcript. Second, we explicitly included noisy examples in our few-shot prompts so the classifier learned to handle imperfect input. We did consider adding an explicit error correction step, but the summarization approach was effective enough."

---

# 7. Key Takeaways

---

## 7.1 Technical Takeaways

| # | Takeaway |
|---|---|
| 1 | **Summarize before classifying.** Reducing input length cuts cost, improves signal, and sidesteps context window limits. |
| 2 | **Few-shot is powerful for cold-start.** When you have no labels, few-shot prompting can bootstrap a labeling pipeline. |
| 3 | **Knowledge distillation converts recurring cost into one-off cost.** Pay the 70B once on a sample to build a dataset, then pay 7B prices forever — ~10x cheaper and ~10x faster at 94% recall. |
| 4 | **Iterate prompts systematically.** Version prompts, measure with fixed eval sets, use SME feedback — treat prompt engineering like software development. |
| 5 | **Set-based metrics for multi-label.** Standard accuracy doesn't work for set-valued predictions; recall is the business metric, Jaccard similarity is the guardrail against over-prediction. |
| 6 | **Per-label thresholds, not a global 0.5.** Each intent has its own base rate and its own cost of being wrong; one cut-off for all of them is the most common multi-label mistake. |
| 7 | **Data governance drives architecture.** PII concerns made Snowflake Cortex the right choice over external APIs. |

## 7.2 Behavioral / Process Takeaways

| # | Takeaway |
|---|---|
| 1 | **Cross-functional collaboration is essential.** Taxonomy design required product knowledge; engineering built the pipeline; SMEs validated quality. |
| 2 | **Start simple, iterate.** Zero-shot → few-shot → distillation. Each step built on the previous. |
| 3 | **Monitoring is not optional.** NLP models drift — especially when language, products, and policies change. |
| 4 | **Think production from day one.** The teacher was never meant for production — the student was the plan all along. |

## 7.3 One-Liner Summary for Each Interview Style

| Interview Style | Your One-Liner |
|---|---|
| **Behavioral** | "I led the design and implementation of an LLM-based classification system that automated intent extraction from 1M+ call transcripts, earning internal recognition as a key AI innovation." |
| **Technical** | "I distilled a 70B teacher into a 7B student — few-shot prompting on the 70B to generate training data, then LoRA fine-tuning on the 7B — hitting 94% recall on validated intent labels at roughly 10x lower cost." |
| **System Design** | "End-to-end NLP pipeline inside Snowflake: ASR transcripts → Cortex summarization → LLM classification → operational dashboards, with automated drift monitoring." |
| **Product/Impact** | "Replaced manual transcript review for a contact centre handling millions of calls, enabling real-time staffing decisions and surfacing previously hidden customer intent patterns." |

---

## 7.4 Quick Reference Card

```
┌──────────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE — JET2 PROJECT                 │
│                                                                    │
│  Role:     Data Scientist, Jet2 & Jet2 Holidays, Leeds, UK         │
│  Period:   June 2022 – Aug 2024                                    │
│  Scale:    1M+ transcripts, 2K-5K/day                              │
│                                                                    │
│  Pipeline: Raw calls → ASR → Snowflake → Cortex SUMMARIZE         │
│            → LLM Classification → Dashboards                       │
│                                                                    │
│  Models:   Teacher = 70B (few-shot) → Student = 7B (SFT/LoRA)     │
│  Metric:   94% RECALL on validated intent labels                   │
│            (multi-label, 3-level taxonomy, SME gold set)           │
│  Guardrail: Jaccard similarity + labels-per-call ratio             │
│                                                                    │
│  Key Tech: Knowledge distillation (sequence-level KD),             │
│            Snowflake Cortex, few-shot learning, LoRA fine-tuning,  │
│            multi-label sigmoid + BCE, per-label thresholds          │
│                                                                    │
│  Impact:   ~85-90% reduction in manual review                      │
│            10-15x cost reduction (student vs teacher)               │
│            Real-time operational dashboards                         │
│            Internal recognition as key AI innovation                │
│                                                                    │
│  Differentiators:                                                   │
│  • Cold-start → teacher-student pipeline (no labels needed upfront)│
│  • Distillation: recurring cost → one-off labelling cost           │
│  • Data governance via Cortex (PII never leaves Snowflake)         │
│  • Systematic prompt engineering (8 iterations, measured)           │
│  • Production-grade with drift monitoring                           │
└──────────────────────────────────────────────────────────────────┘
```

---

*Document prepared for Rahul Sharma — Jet2 and Jet2 Holidays (Leeds, UK) project interview preparation.*
*Last updated: August 2026 — aligned to the current resume (94% recall on validated intent labels; 70B → 7B distillation).*
