# Practical Interview Question Bank — DS / AI Engineer / FDE / Data Engineering / Analytics

**Candidate:** Rahul Sharma | **Purpose:** Real questions asked in interviews (2025–2026 market), organized by role and theme, with concise "how to answer" guidance and **references into your existing notes** instead of duplicating them.

> **How to use this file:** This is the *drill sheet*. The deep explanations live in the numbered learning docs and project docs. For every question here: (1) try answering out loud in 60–90 seconds, (2) check the referenced doc if you stumble, (3) tie the answer to one of your projects (P01–P14) whenever possible. Interviewers score *"has actually shipped this"* far above *"has read about this."*

---

## Table of Contents

1. [The Real Question List (Asked in Your Interviews) — Answered & Referenced](#1-the-real-question-list)
2. [The 2026 Interview Landscape — What Actually Gets Asked](#2-the-2026-interview-landscape)
3. [GenAI / AI Engineer Question Bank](#3-genai--ai-engineer-question-bank)
4. [Production Themes: Scalability, Reliability, Hallucinations, Validation](#4-production-themes)
5. [System Design & Architecture Brainstorming — How to Run the Round](#5-system-design--architecture-brainstorming)
6. [Forward Deployed Engineer (FDE) Question Bank](#6-forward-deployed-engineer-question-bank)
7. [Data Science / Classical ML Question Bank](#7-data-science--classical-ml-question-bank)
8. [Data Engineering Question Bank](#8-data-engineering-question-bank)
9. [Analytics / Product DS Question Bank](#9-analytics--product-ds-question-bank)
10. [AI-Conducted Interviews — How to Handle Them](#10-ai-conducted-interviews)
11. [Behavioral & Project Walkthrough Questions](#11-behavioral--project-walkthrough-questions)

---

# 1. The Real Question List

These were asked in an actual loop. Each entry: **short answer skeleton → where the full answer lives.**

### Q1. Introduction

Structure: *current role → 2 flagship projects (one GenAI, one ML/data) → what you're looking for.* Keep to 90 seconds. Rehearse the 30-second pitches at the bottom of each project doc (`projects/P01`–`P14`, each has a "30-Second Pitch" / "STAR Summary" section).

### Q2. How many GenAI projects have you done? Tools, technologies, cloud infra?

Have a **counted inventory** ready — don't improvise. Yours:

| Project | GenAI core | Infra |
|---|---|---|
| Doc2Data (`P05`/portfolio) | LangGraph state machine, Florence-2 OCR, targeted VLM rescue | FastAPI, Next.js, Docker |
| AI Cargo (portfolio) | LangGraph orchestrator, Groq LLM plan/reflect/revise, 8 agent tools | Supabase, FastAPI, React, Vercel |
| SkillSpring/TrainPI (`P08`) | Groq + Cohere embeddings, RAG distillation, career agent | Supabase pgvector, FastAPI, Vercel |
| Medical RAG (`P07`) | FAISS + BM25 hybrid, cross-encoder rerank, refusal guardrails | Streamlit |
| Legal summarisation (`P11`) | o3-mini structured legal prompts, NER | ThreadPoolExecutor batch pipeline |
| Radiology report gen (`P12`) | Vision-language generation | — |
| HIPAA chat-assistant (repo) | Agentic RAG, fan-out retrieval, LLM-as-judge evals | Ollama local models |

Answer pattern: "Seven-plus shipped GenAI systems; three are hosted with live demos. Stack: LangGraph/LangChain for orchestration, Groq/OpenAI/Cohere APIs, FAISS/pgvector for retrieval, FastAPI backends, Docker, deployed on Vercel/Supabase/Azure."

### Q3. What is regularization? L1 vs L2?

Penalty added to the loss to constrain model complexity and reduce overfitting. **L1 (Lasso)** adds \|w\|: sparse solutions, implicit feature selection, non-differentiable at 0. **L2 (Ridge)** adds w²: shrinks weights smoothly, keeps all features, has closed form. Geometric intuition: L1's diamond constraint hits corners (zeros); L2's sphere doesn't.
→ Full math + elastic net + when-to-use: `03_regularization_and_optimization.md` (L1/L2 sections), `19_classical_ml_algorithms.md` (regression chapter).

### Q4. What is a cost function? Why needed? Examples.

The single scalar the optimizer minimizes — it *defines* what "good" means for the model. Without it there's no gradient signal. Examples: MSE (regression), BCE/log-loss (classification), cross-entropy (multi-class), hinge (SVM), Huber (robust regression).
→ Full catalog with derivations: `02_loss_functions.md`.

### Q5. Bias-variance tradeoff

Expected error = bias² + variance + irreducible noise. High bias = underfit (linear model on non-linear data); high variance = overfit (deep tree). Manage via model complexity, regularization, ensembles (bagging cuts variance, boosting cuts bias), more data.
→ `19_classical_ml_algorithms.md` (bias-variance section with the decomposition math).

### Q6. Feature selection techniques

Three families: **Filter** (correlation, chi-squared, mutual information, variance threshold — fast, model-agnostic), **Wrapper** (RFE, forward/backward selection — expensive, model-specific), **Embedded** (L1, tree feature importance, SHAP-based pruning). Mention leakage risk: select on train folds only.
→ `19_classical_ml_algorithms.md` §6.5 Feature Selection; SHAP details in `36_explainability_and_interpretability.md`. Real usage: `P01_fibe_behaviour_scorecard.md` (1,000+ bureau variables → final scorecard).

### Q7. Ensembles: boosting vs bagging; how bootstrap + aggregation work

**Bagging:** train N models in parallel on bootstrap samples (sample n rows *with replacement* — each sample leaves out ~37% OOB), then aggregate (majority vote / average). Cuts **variance**. **Boosting:** sequential; each model fits the residuals/reweighted errors of the previous. Cuts **bias**. Boosting overfits noisy data more easily; bagging is parallelizable.
→ `19_classical_ml_algorithms.md` (Random Forest + Gradient Boosting chapters, incl. why feature randomness decorrelates trees).

### Q8. Random Forest working + hyperparameters

Bagging + per-split feature subsampling (√p classification, p/3 regression) → decorrelated trees → averaged. Key hyperparameters: `n_estimators`, `max_depth`, `min_samples_split/leaf`, `max_features`, `bootstrap`, `class_weight`. OOB score = free validation.
→ `19_classical_ml_algorithms.md` (RF section). Used in `P03_jet2_event_detection.md` and `P04_jet2_risk_forecasting.md`.

### Q9. Detecting and handling outliers

Detect: z-score / robust z (MAD), IQR fences, Isolation Forest, LOF, domain rules. Handle: investigate first (data error vs true signal!), then cap/winsorize, transform (log), use robust models/losses (Huber), or keep if signal.
→ Statistical detection: `24_statistics_and_ab_testing.md`; algorithmic (Isolation Forest math, LOF): `P03_jet2_event_detection.md` §Isolation Forest + Q24; winsorization for metrics: `40_cuped_and_variance_reduction.md`.

### Q10. Positional encoding — why, techniques, why not needed for RNN/LSTM?

Self-attention is **permutation-invariant** — without position info, "dog bites man" = "man bites dog". Techniques: sinusoidal (original), learned embeddings (BERT/GPT-2), relative (T5), **RoPE** (Llama — rotates Q/K pairs by position-dependent angle), ALiBi (linear attention bias). RNN/LSTM don't need it because they process tokens **sequentially** — order is baked into the recurrence.
→ `04_transformers_and_attention.md` §4 (all variants compared) + Q6 in its interview section.

### Q11. Transformer architecture + decoding techniques

Embeddings + positional encoding → N blocks of (multi-head self-attention → add&norm → FFN → add&norm) → LM head. Decoding: greedy, beam search, temperature sampling, top-k, top-p/nucleus, contrastive; speculative decoding for latency.
→ `04_transformers_and_attention.md` (full walk-through + decoding section); KV cache in `16_context_engineering.md`.

### Q12. RAG architecture

Ingest → chunk → embed → index (vector DB) → retrieve (hybrid: BM25 + dense) → rerank (cross-encoder) → assemble context → generate with citations → evaluate. Always mention **evaluation** (faithfulness / context relevance / answer relevance) — it's the differentiator.
→ `05_rag_systems.md` (end-to-end + your Medical RAG numbers); `P07_bilbo_medical_rag.md` for the shipped version.

### Q13/Q14. Chunking strategies; semantic chunking

Fixed-size + overlap (simple prose), recursive/structure-aware (headers, code), sentence/paragraph, **semantic chunking** (split where embedding similarity between consecutive sentences drops below threshold — chunks follow topic boundaries, not char counts), parent-document retrieval (small chunks for search, big for context).
→ `05_rag_systems.md` §2b + `16_context_engineering.md` (semantic chunking code example).

### Q15. Semantic search

Search by **meaning**, not keywords: embed query + docs into a shared vector space, rank by cosine similarity/ANN. Contrast with lexical (BM25): lexical wins on exact IDs/SKUs/error codes, semantic wins on paraphrase — hence hybrid + RRF.
→ `18_nlp_and_embeddings.md` (embeddings, bi- vs cross-encoders) + `06_vector_databases.md`.

### Q16. Why not store embeddings in a conventional database?

A regular B-tree index answers exact-match/range queries; similarity search needs **nearest-neighbor in high-dimensional space**, where B-trees are useless (curse of dimensionality → full scan, O(n·d) per query). Vector DBs use ANN indexes (HNSW graphs, IVF, PQ compression) for sub-linear approximate search plus metadata filtering. Nuance to mention: Postgres **can** do this via pgvector — the point is the *index type*, not the database brand.
→ `06_vector_databases.md` (HNSW/IVF/PQ internals, pgvector vs Pinecone vs FAISS tradeoffs).

### Q17. Python: pairs of indices in a sorted list summing to target — O(n)

Two pointers from both ends (list is sorted). Handle duplicates carefully:

```python
def pair_indices(li, target):          # li sorted
    lo, hi, out = 0, len(li) - 1, []
    while lo < hi:
        s = li[lo] + li[hi]
        if s == target:
            out.append([lo, hi])
            lo += 1                     # advance both to avoid reuse
            hi -= 1
        elif s < target:
            lo += 1
        else:
            hi -= 1
    return out

# li=[1,2,3,4,4,6,7,9], target=8 → [[0,6],[1,5],[3,4]]
```

O(n) time, O(1) extra space. If unsorted → hash-map of value→index, one pass, O(n) time / O(n) space.
→ More patterns (two pointers, sliding window, hash maps): `25_python_for_interviews.md` + `projects/FusionSpan_Python_Coding_Questions.md`.

### The topic checklist from that loop → where each lives

| Topic | Your doc |
|---|---|
| Supervised vs unsupervised, LinReg/LogReg, SVM, trees, gradient descent | `19_classical_ml_algorithms.md`, `03_regularization_and_optimization.md` |
| Transformer, self/multi-head attention, encoder-decoder vs decoder-only | `04_transformers_and_attention.md` |
| Tokenization / BPE | `18_nlp_and_embeddings.md` §1.1 (BPE vs WordPiece vs SentencePiece, with training example) |
| Embeddings + positional encoding | `18_nlp_and_embeddings.md`, `04_transformers_and_attention.md` |
| LLM APIs, statelessness, memory, token limits | `16_context_engineering.md`, `15_fastapi_and_backend.md` (TrainPI session mgmt) |
| Zero/few-shot, CoT, KV cache, prompt caching | `30_prompt_engineering_patterns.md`, `16_context_engineering.md` (KV cache math) |
| RAG, chunking, vector DBs, LangChain/LlamaIndex | `05_rag_systems.md`, `06_vector_databases.md`, `17_langchain_langgraph.md` |
| Agents: tool use, ReAct, planning, LangGraph, where agents break | `10_agentic_ai_and_multi_agent.md`, `27_agentic_patterns_deep_dive.md`, `17_langchain_langgraph.md` |
| Cosine similarity, matmul, eigenvalues (surface) | `18_nlp_and_embeddings.md` (similarity), `19_classical_ml_algorithms.md` (PCA/eigen) |
| Fine-tune vs prompt, PEFT, LoRA/QLoRA, instruction tuning | `07_fine_tuning_and_peft.md`, `13_quantization.md` |

> **BPE soundbite to memorize:** *"BPE starts from characters and greedily merges the most frequent adjacent pair into a new token, repeating until the vocab budget is hit. GPT-4/Claude/Llama use byte-level BPE variants so any string is tokenizable with no UNK. Practical implications: token counts drive cost and context budget; rare words split into multiple tokens; multilingual and code text tokenize less efficiently."* Full comparison (BPE vs WordPiece vs SentencePiece): `18_nlp_and_embeddings.md` §1.1.

---

# 2. The 2026 Interview Landscape

What loops actually look like now (from current hiring guides and reported loops):

| Role | Typical split | The differentiator question |
|---|---|---|
| **AI Engineer** | ~40% RAG/evals/agents, ~30% production systems, ~20% LLM internals, ~10% behavioral | *"How do you know your AI system is actually working?"* — evals discipline |
| **FDE** | ~50% case study/customer judgment, ~50% coding + system design | The 45–60 min ambiguous **decomposition case study** (lowest pass rate) |
| **Data Scientist** | Stats/ML fundamentals + product case + SQL + sometimes GenAI | Bias-variance, experiment design, metric tradeoffs |
| **Data Engineer** | SQL (heavy), pipeline design, Spark, modeling, one coding round | Incremental loads, late/duplicate data, "design this pipeline" |
| **Analytics/Product DS** | SQL + metric design + experiment interpretation + case | "Metric X dropped 10% — investigate" |

Key shifts vs. older loops:

1. **The opener changed.** ~80% of senior AI-eng loops open with *"design a RAG system for X"* or *"add an LLM feature without burning budget."* Prepare that story cold (§5).
2. **Evals are the filter.** Candidates who can build RAG but can't describe a labeled eval set, LLM-as-judge calibration, or CI gating get rejected. Your ammo: HIPAA chat-assistant eval reports (LLM-as-judge, baseline-vs-model comparisons) — talk about them.
3. **Production incidents beat theory.** "Walk me through an LLM incident you triaged" — prepare 2 real stories (e.g., Doc2Data OCR quality-gate failures → rescue lane; AI Cargo tool-call failures → retry/escalation).
4. **Interviewers expect numbers.** Cost per query, p95 latency, index size, eval set size, token budgets. Vague = junior.
5. **AI assistants during coding are often allowed** — they test whether you can *debug* the assistant's output, not type from memory.

---

# 3. GenAI / AI Engineer Question Bank

Practice these out loud. References point to full treatments.

## 3.1 LLM API & integration

1. **Which providers have you shipped, and what's your fallback strategy?** Name ≥2 (Groq, OpenAI, Cohere in your case). Fallback on timeout/quota: provider B, circuit breaker, cached/degraded response. Pin model versions to avoid silent upgrades. → `23_cloud_mlops_deployment.md`, your AI Cargo/TrainPI stacks.
2. **Rate limits and retries?** Exponential backoff + jitter, distinguish transient (retry) vs permanent (don't), request queueing, per-tenant buckets. → `15_fastapi_and_backend.md`.
3. **Prompt versioning?** Prompts live in code, reviewed like code, every change runs the eval suite pre-merge. → `30_prompt_engineering_patterns.md`.
4. **How do you enforce structured output?** JSON schema / function calling, Pydantic validation, retry-on-parse-failure, fallback model. You do this in Doc2Data (typed validators) and TrainPI. → `15_fastapi_and_backend.md` (Pydantic), `27_agentic_patterns_deep_dive.md`.
5. **Temperature vs top-p vs top-k?** → `04_transformers_and_attention.md` (decoding). Production default: temperature ~0 for deterministic tasks.
6. **Why are LLMs stateless and how do you handle memory?** Each API call is independent; you replay history. Strategies: full history until token budget, rolling summarization, external memory (DB/vector store), session objects server-side. → `16_context_engineering.md`, TrainPI session management in `15_fastapi_and_backend.md`.
7. **KV cache & prompt caching?** KV cache stores per-token key/value tensors so generation is O(1) attention per new token; memory = layers × heads × d × 2 × seq (GQA shrinks it). Prompt caching (Anthropic/OpenAI) reuses the prefix KV across requests — big cost cut when the system prompt + docs prefix is stable; order your prompt static-first to exploit it. → `16_context_engineering.md`.
8. **Fine-tune vs prompt vs RAG?** Freshness → RAG; format/style consistency or latency/cost at volume → fine-tune; else prompt. "Model must know our new product" is a RAG problem, not fine-tuning. → `07_fine_tuning_and_peft.md` §when-to-fine-tune.

## 3.2 RAG (beyond the basics in §1)

9. **Hybrid search — why?** Dense misses exact-match (SKUs, error codes); BM25 catches them; fuse with RRF. → `05_rag_systems.md` §hybrid.
10. **Evaluate retrieval quality?** Recall@k, MRR, nDCG on a labeled set — measured **separately** from generation quality (faithfulness, answer relevance) so you can debug each. Bootstrap labels: 50–100 manually labeled production queries + synthetic augmentation. → `14_evaluation_metrics.md` (RAG metrics), `05_rag_systems.md` (eval section).
11. **Knowledge base updates?** Incremental re-embedding/CDC vs full re-index, corpus versioning, tombstones for deletions, A/B old vs new index. → `06_vector_databases.md`.
12. **Docs exceed context window?** Parent-document retrieval, two-step retrieve (broad→narrow), recursive summarization, long-context model for final synthesis. → `16_context_engineering.md`.
13. **A RAG answer was wrong — debug it.** Trace the pipeline: was the right chunk retrieved (retrieval failure → better chunking/hybrid/rerank)? Retrieved but ignored (context assembly/position)? Present but contradicted (generation failure → grounding prompt, citations, judge)? Three surfaces, three fixes. → `05_rag_systems.md` Q&A section.

## 3.3 Agents

14. **Agent vs workflow?** Workflow = scripted branches; agent = runtime decisions. Say when a workflow is the *right* answer — deterministic logic doesn't need an LLM. Your Doc2Data is deliberately a **state machine** with bounded LLM use; that's a strong answer. → `27_agentic_patterns_deep_dive.md`.
15. **Where do agents break in production?** Loops without termination bounds, tool-call failures, error amplification across steps, cost blowup (5–30× a single call), prompt injection via tool outputs, non-reproducibility. Mitigations: max iterations, checkpointing (LangGraph), human-in-the-loop for destructive tools, per-step tracing. → `10_agentic_ai_and_multi_agent.md`, `17_langchain_langgraph.md`; live example: AI Cargo plan→reflect→revise→observe with 8 tools + approval gates.
16. **ReAct?** Thought → Action (tool call) → Observation loop until answer; interleaves reasoning with acting, grounds each step in tool results. → `27_agentic_patterns_deep_dive.md`.
17. **MCP?** Open standard (JSON-RPC) for agents to discover/call tools & resources; use when integrating third-party tools or exposing yours. → `11_a2a_and_mcp_protocols.md`.
18. **Multi-agent orchestration pattern?** Supervisor, parallel specialists, sequential pipeline; know when multi-agent is *worse* (coordination overhead, ~15× token cost) than one well-prompted agent. → `10_agentic_ai_and_multi_agent.md`.

## 3.4 Evaluation & observability

19. **How do you measure whether a prompt change improved things?** Labeled eval set + automated scoring (exact match / rubric / LLM-as-judge) + deploy gate. Never "I tested it manually." → `14_evaluation_metrics.md`; your HIPAA `eval/results/` comparisons are the shipped proof.
20. **LLM-as-judge — when does it work?** Subjective dimensions (helpfulness, tone); needs grounding for factuality; calibrate against human labels; beware position/verbosity bias — randomize order, use rubrics, multiple judges. → `14_evaluation_metrics.md`.
21. **What do you monitor in production?** Quality (judge score on sampled traffic, thumbs-down rate, refusal rate, hallucination flags), system (p50/p95/p99, error rate), cost ($/query, tokens in/out, cache hit rate). Alert on regressions. → `26_system_design_for_ml.md` (monitoring), `23_cloud_mlops_deployment.md`.
22. **Eval in CI/CD?** Every PR touching prompts/models runs the eval set; block merge if faithfulness/accuracy drops >2–3pts below baseline. → `23_cloud_mlops_deployment.md` (CI/CD section).

---

# 4. Production Themes

The four themes you asked to drill — each with the standard question forms and answer skeletons.

## 4.1 Scalability

**"Your RAG/LLM feature goes from 100 to 100K users — what breaks first and what do you do?"**

Answer in layers:
1. **LLM calls** (the bottleneck & cost driver): batch where possible, cache (semantic cache for similar queries; prompt caching for shared prefixes), route easy queries to smaller models (cascade), cap tokens.
2. **Retrieval:** ANN index tuning (HNSW ef/M), shard/replicate the vector DB, metadata pre-filtering, quantize embeddings (PQ).
3. **Serving:** async FastAPI + queue, horizontal scaling, streaming (SSE) so perceived latency drops, timeouts + backpressure.
4. **Data pipeline:** move embedding/ingestion offline/batch; incremental indexing.
→ `26_system_design_for_ml.md` (scaling patterns), `06_vector_databases.md` (ANN tuning), `15_fastapi_and_backend.md` (async/streaming).

**"How do you cut LLM cost 10× without killing quality?"** Model routing/cascades, prompt caching + shorter prompts (trim retrieved chunks post-rerank), semantic caching, fine-tune a small model for the narrow high-volume task, batch offline work, cap agent iterations. Quantization if self-hosting → `13_quantization.md`, `08_knowledge_distillation.md`.

## 4.2 Reliability

**"Design for the day OpenAI is down."** Multi-provider fallback behind an abstraction, circuit breaker, degraded mode (cached answers / heuristic / honest "degraded" banner), kill switch per feature, retry budget so you don't amplify the outage.

**"What's different about an LLM outage vs a normal outage?"** LLMs fail *soft* — quality degrades silently while uptime looks fine. So reliability = continuous quality evals on sampled production traffic, not just health checks. Distinguish "model wrong" (not pageable, feeds eval backlog) from "model unreachable" (pageable).
→ `26_system_design_for_ml.md` (reliability/guardrails), `31_ai_safety_guardrails_responsible_ai.md`.

## 4.3 Hallucinations

**"How do you prevent hallucinations?"** — the honest structure:
1. You **reduce**, not eliminate: ground with RAG, require citations, verify citations against retrieved context (rule or second model), refuse below retrieval-confidence threshold, temperature ↓ for factual tasks.
2. **Detect:** faithfulness scoring (NLI / LLM-judge vs context), self-consistency sampling, adversarial eval set that tries to elicit hallucinations.
3. **Contain:** UX (citations, confidence, easy escalation), human-in-the-loop for high-stakes actions.
Your shipped example: Medical RAG's exact-refusal guardrail ("I cannot advise — no relevant content found") — say it.
→ `05_rag_systems.md` Q2, `14_evaluation_metrics.md` §6.1 (hallucination taxonomy + detection), `31_ai_safety_guardrails_responsible_ai.md`, `30_prompt_engineering_patterns.md` Q2.

## 4.4 Validation (models & data)

**ML validation:** proper splits (time-based for temporal data — you did this in P04/P10; stratified otherwise), CV, leakage checks (fit transforms inside folds), baseline-first, holdout gating before deploy, post-deploy drift monitoring (PSI — implemented in `P01`/`P04`).
**Data validation:** schema + nulls + duplicates + volume + freshness gates before load — your PySpark `DataQualityValidator` (`P05b_atcs_pyspark_etl_pipeline.md`) and experiment diagnostics (`P14`, `43_debugging_measurement_systems.md`) are the shipped stories.
**LLM validation:** eval sets + judges + CI gates (§3.4).

---

# 5. System Design & Architecture Brainstorming

## 5.1 The 8-step script for any AI/ML system design round

```
1. REQUIREMENTS   (3-5 min) — users, scale, latency budget, cost budget,
                   accuracy bar, freshness, compliance. ASK, don't assume.
2. SUCCESS METRICS — primary + guardrails; how you'd EVALUATE the system.
                   (Saying this early is a senior signal.)
3. HIGH-LEVEL ARCHITECTURE — boxes & arrows: data in → processing →
                   model/retrieval → serving → feedback loop.
4. DATA & RETRIEVAL DESIGN — sources, chunking/features, index/storage.
5. MODEL/LLM CHOICES — buy vs build, model tier, fallback, structured output.
6. SERVING & SCALE — API design, caching layers, async/streaming, cost math.
7. RELIABILITY & SAFETY — evals in CI, monitoring, guardrails, incident plan.
8. ITERATE — name the v1 you'd ship this week vs the v3 roadmap.
```

Deep treatments + worked designs (recommender, LLM app, fraud, search): `26_system_design_for_ml.md`. Practice by re-deriving your own systems from scratch: Doc2Data (`doc2data/documents/PIPELINE_OVERVIEW.md`), AI Cargo, TrainPI, `P14` experimentation platform.

## 5.2 Prompts to drill (30 min each, out loud, whiteboard)

1. Design a RAG system for enterprise customer-support tickets (the modal 2026 opener).
2. Add an LLM feature to an existing SaaS product without blowing the budget.
3. Design a document-extraction pipeline for healthcare forms (you shipped this — Doc2Data — practice presenting it as a design).
4. Design an agent that triages supply-chain risk alerts (AI Cargo).
5. Design semantic search over 100M product listings.
6. Design an experimentation platform for a marketplace (P14 + `41_experimentation_platform_systems.md`).
7. Design a fraud/risk scoring service with real-time features (P01/P06 + `26_system_design_for_ml.md`).
8. Design evaluation infrastructure for a company's LLM features (evals-as-a-product; `14` + `23`).

---

# 6. Forward Deployed Engineer Question Bank

FDE loops (Palantir, OpenAI, Anthropic, Databricks, ElevenLabs): ~50% case study/customer judgment. The signature round is the **ambiguous decomposition case study** — lowest pass rate, highest weight. Most candidates fail by **solving before scoping**.

## 6.1 The decomposition case study — script

Given: *"A hospital chain / logistics firm / bank wants to 'use AI to improve operations.' Go."*

```
1. CLARIFY (5-10 min — spend real time here; this IS the test)
   - Who are the stakeholders? Who owns the decision? Who are the users?
   - What does success look like in 6 months, in numbers?
   - What data exists? Where? Quality? Access/compliance constraints?
   - What's been tried? Why did it fail?
2. SCOPE — pick ONE wedge use case with measurable value and available
   data. Say explicitly what you are NOT doing in v1.
3. DECOMPOSE — data ingestion → cleaning → model/LLM component →
   integration into the existing workflow → evaluation → rollout.
4. RISKS & TRADEOFFS — data quality, adoption, compliance, model risk;
   mitigation for each.
5. PLAN — week 1 deliverable, 30/60/90 days, decision checkpoints.
```

## 6.2 Questions to prepare

1. *"Why forward deployed / customer-facing work?"* — tie to consulting-style delivery you've done (ATCS client engagements, Doc2Data for a real workflow).
2. *"A customer's team distrusts the model's outputs. What do you do?"* — sit with users, collect failure examples, build an eval set from *their* cases, show precision on their data, add human-in-the-loop until trust is earned.
3. *"You're on-site; the demo breaks in front of the client."* — stay calm, fall back to a scripted path, be honest, timebox live debugging, follow up with root cause within 24h.
4. *"The customer asks for X but actually needs Y."* — deliver a thin slice of X to build trust while showing data for Y; escalate scope tradeoff to the sponsor with options, not refusals.
5. *"How do you know your deployed AI system is working?"* — THE differentiator: baseline metrics before deploy, eval set from customer data, business metric it moves, weekly quality review with the customer.
6. Technical depth: same banks as §3–§5 (RAG, evals, agents, production reliability), plus practical integration — auth, data access patterns, on-prem/VPC constraints (`23_cloud_mlops_deployment.md`, `28_terraform_iac_platform_engineering.md`).
7. *"Tell me about a time you translated a vague business ask into a shipped system."* — use P01 (scorecard revamp), Doc2Data, or the GPU benchmark → recommendation engine story.

---

# 7. Data Science / Classical ML Question Bank

Fundamentals get re-asked forever. Rapid-fire list with references:

| # | Question | Reference |
|---|---|---|
| 1 | Supervised vs unsupervised vs self-supervised | `19_classical_ml_algorithms.md` §1 |
| 2 | Linear regression assumptions + diagnostics | `19` (regression), `44_econometrics_deep_dive.md` (OVB, robust SE) |
| 3 | Logistic regression: why log-odds, MLE fitting | `19`, `47_deep_statistical_reasoning.md` §3.6 |
| 4 | Gradient descent variants (batch/mini/SGD, momentum, Adam) | `03_regularization_and_optimization.md` |
| 5 | SVM: margin, kernel trick, C/γ | `19` (SVM chapter) |
| 6 | Decision trees: splitting criteria, pruning | `19` |
| 7 | RF vs GBM vs XGBoost — when which | `19`; applied in P01/P04/P06/P10 |
| 8 | Class imbalance: resampling, class weights, threshold moving, PR-AUC | `19`, `14_evaluation_metrics.md`; P01/P06 stories |
| 9 | Precision/recall/F1/ROC-AUC vs PR-AUC — when which | `14_evaluation_metrics.md` |
| 10 | Overfitting detection + cures | `03`, `19` |
| 11 | Curse of dimensionality; PCA intuition + eigen decomposition | `19` |
| 12 | Missing data strategies (MCAR/MAR/MNAR) | `19`; scorecard handling in P01 |
| 13 | p-value meaning (and what it is NOT) | `47_deep_statistical_reasoning.md` §10 |
| 14 | CLT and why it matters | `47` §5, `24_statistics_and_ab_testing.md` |
| 15 | Confidence vs credible intervals | `47` §6 |
| 16 | A/B test end-to-end design (power, MDE, SRM, guardrails) | `24_statistics_and_ab_testing.md` §6; platform view in `41` |
| 17 | CUPED / variance reduction | `40_cuped_and_variance_reduction.md` |
| 18 | Can't randomize — now what? (DiD, PSM, IV, RDD) | `33_causal_inference_and_experimentation.md` |
| 19 | Correlation ≠ causation — give a concrete confounder story | `24` Q&A, `33` |
| 20 | Time series: stationarity, ARIMA vs ML, backtesting | `34_time_series_and_forecasting.md`; P04/P10 |
| 21 | Model explainability: SHAP vs LIME vs permutation importance | `36_explainability_and_interpretability.md` |
| 22 | "Your model's production performance dropped — walk me through debugging" | drift (PSI) → data pipeline breaks → label shift → seasonality; `23` (monitoring), P01 §monitoring |

For GenAI-flavored DS roles add: `38_genai_for_data_scientists.md`.

---

# 8. Data Engineering Question Bank

| # | Question | Reference |
|---|---|---|
| 1 | Design an ETL pipeline for daily CSV/API/DB sources | `P05b_atcs_pyspark_etl_pipeline.md` (your shipped answer), `39_oop_etl_concepts.md` |
| 2 | ETL vs ELT — when which | `22_data_engineering_and_sql.md` |
| 3 | Full vs incremental loads; high-water marks; late-arriving data | `22`, `P05b` §load |
| 4 | Dedup at scale (exact + business-key with window functions) | `P05b` §transform, `22` (ROW_NUMBER pattern) |
| 5 | SCD Type 2 | `22` |
| 6 | Star vs snowflake schema; fact/dimension design | `22` §database design |
| 7 | Partitioning vs sharding vs indexing | `22` |
| 8 | Query optimization: EXPLAIN, index choice, join order | `22` |
| 9 | Spark: lazy evaluation, wide vs narrow transforms, shuffle, broadcast joins, skew | `22` (PySpark deep dive), `P05b` §performance |
| 10 | Airflow: DAG design, idempotency, backfills, sensors | `32_data_pipeline_orchestration.md` |
| 11 | Data quality gates: completeness/uniqueness/freshness/volume | `P05b` §quality framework; measurement debugging mindset in `43` |
| 12 | CDC approaches (timestamp, log-based, streams) | `22` |
| 13 | Batch vs streaming; exactly-once semantics | `22`, `26_system_design_for_ml.md` |
| 14 | Warehouse comparison: Snowflake vs Redshift vs BigQuery | `22` §warehouses |
| 15 | "Pipeline produced wrong numbers yesterday — debug it" | lineage: source → joins (dup explosion?) → filters → timezone/late data → schema drift; `43_debugging_measurement_systems.md` |

SQL practice: write, don't read — top-N per group, dedup with ROW_NUMBER, funnel with LEFT JOINs, retention with self-join/window, running totals (`22` §window functions).

---

# 9. Analytics / Product DS Question Bank

| # | Question | Reference |
|---|---|---|
| 1 | "DAU dropped 10% — investigate" | Segment (platform/geo/cohort), check instrumentation first (logging bugs — `43`), internal vs external causes, quantify each; framework in `37_product_sense_and_business_metrics.md` §8 |
| 2 | Define a north-star metric for product X + guardrails | `37` §2–3 |
| 3 | Design metrics for a new feature launch | primary + secondary + guardrail; `37`, `14` §experiment metrics |
| 4 | Leading vs lagging metrics | `37` §2.3 |
| 5 | The experiment is flat/negative but PM wants to ship | check power/MDE, segments (HTE — `33` §8), triggered analysis (`41`), be honest about nulls; decision rules in `platform_exp/docs/decision_rules.md` |
| 6 | SRM detected — what now? | stop, investigate assignment/logging before reading results; `24` §SRM, `43`, `P14` diagnostics |
| 7 | Novelty & primacy effects | `24` (pitfalls), `43` |
| 8 | Simpson's paradox — real example | `24` Q&A; P14 scenario has one built in |
| 9 | Metrics for a marketplace / ads / SaaS business | `37` §9, `45_marketplace_science.md` §8, `46_ads_ranking_and_auctions.md` |
| 10 | Estimate/sizing questions ("how many X…") | structure > answer: decompose, state assumptions, sanity-check bounds |
| 11 | Communicate a technical result to execs | lead with the decision, one number, one caveat; `37` case framework |

---

# 10. AI-Conducted Interviews

More screens are run by AI interviewers (async video or chat-based). They score differently than humans:

1. **Structure is scored literally.** AI graders reward explicit signposting: "There are three parts to this: first… second… third…". Use frameworks by name (STAR, the 8-step design script in §5).
2. **Keyword coverage matters.** Rubrics look for expected terms (e.g., for RAG: chunking, embeddings, reranking, faithfulness, eval set). Don't be obscure — say the standard word for the thing.
3. **Complete sentences, steady pace.** ASR + scoring penalize fragments, long silences, and trailing off. It's fine to pause 3–5 seconds, then answer fluently.
4. **Answer the question asked, fully, then stop.** AI follow-ups are less adaptive; front-load the direct answer ("Yes — because X, Y, Z"), then elaborate.
5. **For coding with an AI proctor:** narrate intent before code, state complexity unprompted, run through one example input aloud — these hit rubric rows humans might let slide.
6. **Don't try to game it with fluff.** Modern rubrics penalize keyword stuffing without logic; the structure has to carry real content.
7. **Practice setup:** record yourself answering 5 questions from §3/§7 in 90 seconds each; listen for filler words and missing signposts. Same drill works for human loops.

---

# 11. Behavioral & Project Walkthrough Questions

Every loop includes these. Prepare **one page per story**, STAR format — most exist already in your project docs (each has a STAR summary; rehearse from there).

Must-have stories (map before interview day):

| Prompt | Your best story |
|---|---|
| Biggest technical challenge | Doc2Data lane routing + OCR rescue; or P01 scorecard on 1,000+ variables |
| A production incident you owned | AI Cargo tool failures / Doc2Data quality-gate regression — timeline, blast radius, root cause, fix, postmortem |
| Conflict / pushback | Model choice or metric disagreement — end with data resolving it |
| A time you were wrong | e.g., over-engineered agent → replaced with deterministic workflow (also a great §3.3 answer) |
| Ambiguity → shipped | GPU benchmark → recommender; ATCS client pipelines |
| Mentoring / raising the bar | ETL framework adopted by 4 engagements (`P05b`) |
| Why this company | research their product; connect to a specific system you built |

**Project deep-dive defense (60 min "present and defend"):** pick ONE system (Doc2Data or AI Cargo), and be ready for: why this architecture and what you'd change today; what broke and how you found it; how you evaluated it (numbers); cost/latency; what you'd cut to ship in a week. Rapid follow-ups are the test — short, direct, honest answers beat polish.

---

## Final reminders

- **Numbers beat adjectives.** "Cut processing from hours to ~20 minutes on 10M records" > "made it much faster."
- **Always close the loop with evaluation.** Any design answer that doesn't end with "and here's how I'd know it works" is incomplete in 2026.
- **"It depends" + the two branches** is a senior answer; a single absolute is a junior answer.
- **Honesty about limits scores points:** "You can't fully prevent hallucinations; you reduce the rate and design fallbacks" is the *correct* answer, not a weakness.
