# Resume Master Map — Every Bullet, Its Proof, and Its Drill

**Source of truth:** `Rahul Sharma_edited 18th.pdf` (the new resume). This file replaces `Resume_A10.pdf` as the baseline for all preparation.

**Why this file exists:** your old prep was built on the A10 resume. Several headline numbers and whole capabilities changed. If you rehearse the old numbers you will contradict your own resume in an interview — the fastest way to lose credibility. Read §1 first and burn the corrections in.

---

## 1. ⚠️ What changed from the old resume (fix these in your head NOW)

| Claim | OLD (A10) — do not say this | NEW — say this | Where |
|---|---|---|---|
| Jet2 intent classification | 88% **Jaccard recall** | **94% recall** on validated intent labels, multi-label | `projects/P02` |
| Jet2 anomaly detection | 92% specificity | **96% specificity** | `projects/P03` |
| Jet2 forecasting | "drift <2%" only | **MAPE 18% → 7%**, XGBoost/LSTM/GRU bake-off, Optuna, walk-forward, **<2% error variation over 18+ months** | `projects/P04` |
| ATCS computer vision | 85% mAP | **92% mAP**, ResNet-101-FPN backbone, **−63% manual review** | `projects/P05` |
| ATCS ETL | vague "large-scale datasets" | **10M+ records**, **4 h → 18 min**, config-driven DQ, **4+ engagements**, Docker + CI/CD | `projects/P05b` |
| Bilbo title | AI Engineer **Intern** | **AI Engineer** (Sept 2025 – May 2026) | `projects/P07` |
| Bilbo retrieval | not quantified | **+28% top-five relevance**, RRF + cross-encoder, **sub-5s** | `projects/P07` |
| Amazon Pay title | Research **Intern** | **Research Analyst** | `projects/P06` |
| Jet2 location | Pune, India | **Leeds, UK** | all Jet2 docs |

### Brand-new capabilities on the resume (you had NO prep for these)

| New claim | Why it matters | Study |
|---|---|---|
| **Distilling a 70B teacher into a 7B model** (Jet2) | Turns a classification project into an *LLM efficiency* story — very interviewable | `projects/P02` + `learning/08_knowledge_distillation.md` |
| **LangGraph agentic workflows** (Bilbo) — stateful orchestration, conditional routing, memory, tool calling | This is the #1 hot topic in 2026 AI interviews | `projects/P07` §8 |
| **Structured output validation + LLM guardrails** (Bilbo) | Highest-scrutiny claim in a healthcare context | `projects/P07` §9 |
| **WOE/IV + monotonic binning + KS/Gini/OOT** (Fibe) | Full classical credit-risk vocabulary | `projects/P01` + `learning/51_credit_risk_and_scorecard_modeling.md` |
| **MLflow MLOps workflows** (Fibe) | Experiment tracking / registry questions | `learning/49_llmops_and_observability.md` |
| **Snowflake Cortex + few-shot prompting** (Jet2) | In-warehouse LLM — a governance story | `projects/P04` |
| **KYC JSON → address normalization, +37% coverage** (Amazon Pay) | Messy-data engineering story | `projects/P06` |
| **chi² / mutual information / LASSO, −42% dimensionality** (Amazon Pay) | Classic feature-selection drill | `projects/P06` + `learning/19_classical_ml_algorithms.md` |
| **Docker + CI/CD** (ATCS) | Every MLE loop asks this | `learning/23_cloud_mlops_deployment.md` |
| Skills: **LLMOps, LangSmith, vLLM, speculative decoding, Kubernetes, model serving** | You listed them — you WILL be asked | `learning/49`, `learning/50` |

> **Rule:** never list a skill you can't survive two follow-up questions on. If you keep `vLLM` and `Speculative Decoding` on the resume, read `learning/50_llm_serving_and_inference_optimization.md` before your next interview. Non-negotiable.

---

## 2. The resume, bullet by bullet, with its drill

### Bilbo.ai — AI Engineer (Sept 2025 – May 2026) → `projects/P07` ⭐ FLAGSHIP

| Bullet | Number | Likely follow-up | Your move |
|---|---|---|---|
| Production healthcare RAG for protocol queries, hybrid retrieval + reranking + citation grounding | **sub-5s** | "Where does the 5 s go?" | Break the budget down: retrieval ms, rerank ms, generation dominates |
| Eight-stage document pipeline → citation correctness | **8 stages** | "Why not LangChain's PDF loader?" | Numbered-paragraph format, heading hierarchy, cross-page stitching, citation metadata |
| FAISS + BM25 + RRF + cross-encoder | **+28% top-5** | "28% of what? Overfit to your eval set?" | Relative hit-rate on a labelled set; admit set size; bootstrap CI |
| 4-bit Mistral-7B via Ollama, LangGraph, structured output, guardrails | Q4_K_M ~4.37 GB | "Why not GPT-4?" | PHI can't leave the box → local inference is a *compliance* control |
| Agentic LangGraph workflows: routing, memory, tools | 9-stage traversal | "Couldn't this be a for-loop?" | Branching, relevance-gated refusal, bounded repair, resumable sessions |

### Jet2 — Data Scientist (June 2022 – Aug 2024), Leeds UK → `projects/P02`, `P03`, `P04`

| Bullet | Number | Likely follow-up |
|---|---|---|
| Multi-label intent classification, 1M+ transcripts, fine-tuned LLMs, **70B → 7B distillation** | **94% recall** | "What was precision? Did rare intents survive distillation?" |
| Anomaly detection for LivePerson chats: topic modelling + moving averages + z-scores + Isolation Forest | **96% specificity** | "Why optimize specificity, not recall? What's your false-negative rate?" |
| XGBoost vs LSTM vs GRU, Optuna, walk-forward validation | **MAPE 18% → 7%** | "MAPE is bad for intermittent demand — why MAPE? How many folds?" |
| Dataiku deployment, scheduled retraining, Pydantic validation, Tableau | **<2% over 18+ months** | "What triggered a retrain? What did drift look like?" |
| Snowflake Cortex + few-shot summarization | — | "Why summarize in-warehouse instead of exporting?" |

### Fibe — Data Scientist (Sept 2021 – June 2022) → `projects/P01`

| Bullet | Number | Likely follow-up |
|---|---|---|
| Scorecard from CIBIL + Experian + transaction + behavioral, **1,000+ variables**, WOE/IV + monotonic binning | **0.94 ROC-AUC** | "0.94 on credit is suspiciously high — where's the leakage?" |
| Validation: ROC-AUC, KS, Gini, OOT, risk-segment analysis | — | "When do KS and AUC disagree? What's your PSI?" |
| MLflow: tracking, versioning, reproducible validation | — | "What exactly did you log? How do you reproduce a model from 6 months ago?" |
| Automated FE, scoring, validation gates | **15x: 3 days → <5 h** | "What was the 3 days actually spent on?" |

### ATCS — Associate Data Scientist (Nov 2020 – Sept 2021) → `projects/P05b` (ETL), `P05` (CV)

| Bullet | Number | Likely follow-up |
|---|---|---|
| Distributed PySpark ETL, multi-source | **10M+ records, 4 h → 18 min** | "What actually took 4 hours — Spark or the DB? How did you find the skew?" |
| Config-driven DQ framework: validation, profiling, referential integrity | **4+ engagements** | "How do you check referential integrity at scale?" (anti-join) |
| Mask R-CNN + **ResNet-101-FPN** on Azure | **92% mAP, −63% review** | "mAP at which IoU? How did you measure the 63%?" |
| Docker + CI/CD for Python/ML workloads | — | "How do you unit-test a Spark job?" |

### Amazon Pay India — Research Analyst (May 2020 – Nov 2020) → `projects/P06`

| Bullet | Number | Likely follow-up |
|---|---|---|
| Fraud-risk model, behavioural features, segmentation, hypothesis testing | **95% F1** | "95% F1 on fraud is implausible — explain your class balance" |
| KYC JSON → standardized addresses via regex + rule-based extraction | **+37% coverage** | "What was the denominator? How did you validate correctness?" |
| chi² + mutual information + LASSO feature selection | **−42% dimensionality** | "chi² assumes expected counts >5 — did you check? MI vs correlation?" |
| Power BI: OCF, credit KPIs, portfolio risk | **−30% turnaround** | "What was manual before?" |

### Projects → `projects/P15` (on resume), plus `P16`, `P17` (portfolio depth)

| Bullet | Number | Follow-up |
|---|---|---|
| Six-layer cold-chain risk platform, live telemetry, real-time scoring | 6 layers | "What does 'real time' mean with synthetic data?" |
| 14 features + 8 deterministic rules + Optuna XGBoost + SHAP | **14 / 8** | "How do you know XGBoost isn't just relearning the rules?" |
| FastAPI 25 endpoints, WebSockets, React, Supabase, Vercel, **4 LLM providers + deterministic fallback** | **25 / 4** | "25 endpoints sounds like bad API design — defend it" |

**Not on the new resume but keep warm:** `P16` Doc2Data and `P17` PathWise are your strongest *systems* and *eval* stories. Bring them up as "other things I've built" when asked about agents, cost-aware inference, or evaluation — they're often better answers than the resume projects.

### Achievements
- **IEEE ICSCDS 2025** — "Deep Learning for Reliable Routing in WANETs" → `projects/P13`
- **1st prize + $4,000, Smith Agentic AI Challenge** → `projects/P15`. Lead with this; a competitive win against other builders is third-party validation.

---

## 3. Your resume's skills → where to actually learn them

| Skill block on resume | Primary docs |
|---|---|
| Agentic AI, Multi-Agent Systems, LangGraph, LangChain | `10`, `27`, `17`, + `P07` §8, `P15`, `P17` |
| LLM Evaluation, LLMOps, LangSmith | **`49_llmops_and_observability.md`**, `14_evaluation_metrics.md` |
| vLLM, Speculative Decoding, Model Serving | **`50_llm_serving_and_inference_optimization.md`**, `13_quantization.md` |
| Fine-Tuning, PEFT | `07_fine_tuning_and_peft.md`, `09_llm_alignment_rlhf.md`, `08_knowledge_distillation.md` |
| Transformers, Vector Databases, Semantic Search | `04`, `06`, `18`, `05` |
| MLflow, Docker, Kubernetes, CI/CD | `23_cloud_mlops_deployment.md`, `49`, `50` |
| FastAPI | `15_fastapi_and_backend.md` |
| ETL Pipelines, Spark, SQL, Snowflake, Databricks | `22`, `32`, `39`, + `P05b` |
| AWS / Azure / GCP | `23` |
| Python, C/C++, Bash | `25_python_for_interviews.md` |
| *(implied by Fibe)* credit risk, WOE/IV, KS/Gini | **`51_credit_risk_and_scorecard_modeling.md`** |
| *(2026 landscape)* MCP, GraphRAG, DSPy, reasoning models, GRPO | **`52_frontier_ai_and_2026_landscape.md`**, `11_a2a_and_mcp_protocols.md`, `16_context_engineering.md` |

---

## 4. Which story to lead with, by role

| Role | Lead project | Second | Concept docs to have hot |
|---|---|---|---|
| **AI / GenAI Engineer** | P07 Bilbo (current role, RAG+agents) | P15 AI Cargo | `05`, `49`, `50`, `10`, `52` |
| **Applied AI / Agentic** | P15 AI Cargo | P07, P17 | `10`, `27`, `17`, `52` |
| **MLE (platform/serving)** | P07 + P16 | P05b | `50`, `23`, `26`, `49` |
| **Data Scientist (risk/credit)** | P01 Fibe | P06 Amazon Pay | `51`, `19`, `24`, `14` |
| **Data Scientist (product/exp)** | P14 experimentation | P04 forecasting | `24`, `33`, `40`, `37` |
| **Data Engineer** | P05b ATCS ETL | P04 (Dataiku/Cortex) | `22`, `32`, `39` |
| **Forward-Deployed Engineer** | P16 Doc2Data | P07 | `48` §6, `05`, `22` |

---

## 5. Your 60-second intro (rehearse verbatim, then make it yours)

> "I'm a data scientist and AI engineer with about five years across risk, NLP, and now applied GenAI. I did my Master's in Data Science at Maryland, and most recently I've been an AI Engineer at Bilbo.ai building a production healthcare RAG platform — hybrid retrieval with FAISS and BM25 fused through reciprocal rank fusion, cross-encoder reranking, paragraph-level citations, all running on a locally-hosted quantized Mistral-7B because the data can't leave the environment. That lifted top-five retrieval relevance about 28% and answers come back in under five seconds.
>
> Before that I spent two years at Jet2 as a data scientist, where I did multi-label intent classification over a million-plus call transcripts — including distilling a 70B teacher into a 7B model to make it deployable at 94% recall — plus anomaly detection and forecasting where I took MAPE from 18% to 7%. Earlier I built credit-risk scorecards at a lending company, reaching 0.94 AUC using WOE-based scorecard methodology, and did fraud modelling at Amazon Pay.
>
> On the side I build agentic systems — I won first prize in the Smith Agentic AI Challenge for a pharma cold-chain risk platform that fuses deterministic compliance rules with explainable ML under a LangGraph agent with human-in-the-loop approvals. What I care about most is making AI systems you can actually trust in production: grounded, evaluated, and bounded."

**Why this works:** current role first (most relevant), one hard number per stint, ends on a differentiator (competitive win) and a *principle* — which invites the interviewer into your strongest territory.

---

## 6. The five through-lines (your engineering identity)

Interviewers score judgment, and judgment reads as *consistent principles across projects*. Say these and cite two projects each:

1. **Never let the model be the only vote.** AI Cargo's deterministic veto; Bilbo's citation verification; Doc2Data's validators.
2. **Spend expensive compute only where it changes the answer.** Doc2Data's lane routing + blank detection; Bilbo's retrieve-then-rerank cascade; the 70B→7B distillation at Jet2.
3. **Make failure loud and cheap, not silent and expensive.** Doc2Data's alignment gate; AI Cargo's "unknown escalates"; Bilbo's relevance-gated refusal.
4. **Constrain agents in code, not prompts.** AI Cargo's hard tool filter; Bilbo's bounded repair loop and planned stage order.
5. **Reliability is a number, not a hope.** PathWise's eval harness; Fibe's KS/PSI/OOT battery; Bilbo's labelled retrieval set.

---

## 7. Honest-answer bank (use these instead of bluffing)

| Trap | Say this |
|---|---|
| Metric sounds too high (0.94 AUC, 95% F1) | Name the reason it's plausible (bureau data is genuinely predictive / class balance), then volunteer how you checked for leakage. Volunteering the check is what buys credibility. |
| Small eval set | "Directional, not definitive — I'd report a bootstrap CI and expand the set before making a stronger claim." |
| Synthetic data (AI Cargo) | "The contribution is the system design and governance; swapping a real telemetry feed is a config change." |
| Skill you listed but haven't used deeply | "I've read the internals and can reason about it, but I haven't run it at scale in production — here's what I *have* shipped that's adjacent." Never fake depth. |
| "What broke?" | Always have a concrete failure + how you instrumented it + the fix. AI Cargo's phantom tool re-execution and Doc2Data's silent misalignment are your two best. |

---

*Update this file whenever the resume changes. If a number here disagrees with the resume, the resume wins — then fix this file.*
