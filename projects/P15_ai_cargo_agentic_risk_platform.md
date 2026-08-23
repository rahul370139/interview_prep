# P15: AI Cargo — Agentic Risk Intelligence Platform for Pharma Cold-Chain

> **Project Type:** Flagship self-project — **Smith Agentic AI Challenge, 1st prize + $4,000**
> **Timeline:** **March 2026 – April 2026** (~6 weeks, solo build)
> **Live Demo:** [ai-cargo-monitor-prod.vercel.app](https://ai-cargo-monitor-prod.vercel.app/)
> **GitHub:** [github.com/rahul370139/SmithAgenticAIChallenge](https://github.com/rahul370139/SmithAgenticAIChallenge)

**One-line pitch:** *A six-layer platform that ingests live shipment telemetry, scores pharmaceutical spoilage risk in real time by combining FDA/GDP compliance rules with explainable ML, and lets a LangGraph agent plan and run eight specialized tools — always pausing for a human before anything consequential ships, with a full SHAP-annotated audit trail.*

---

## Table of Contents

0. [Resume Bullet ↔ Proof Map](#0-resume-bullet--proof-map)
1. [STAR Summary](#1-star-summary)
2. [Architecture](#2-architecture)
3. [Risk Scoring Engine (the "is this real?" core)](#3-risk-scoring-engine)
4. [Agentic Orchestration (LangGraph)](#4-agentic-orchestration-langgraph)
5. [The 8 Tools + Cascade Enrichment](#5-the-8-tools--cascade-enrichment)
6. [RAG Compliance Sub-system](#6-rag-compliance-sub-system)
7. [Real-Time Telemetry & the Live Dashboard](#7-real-time-telemetry--the-live-dashboard)
8. [Reliability & the LLM Provider System](#8-reliability--the-llm-provider-system)
9. [Topics You Must Know](#9-topics-you-must-know)
10. [Interview Q&A](#10-interview-qa)
11. [Trick & Follow-Up Questions](#11-trick--follow-up-questions)
12. [Red Flags & How to Handle](#12-red-flags--how-to-handle)
13. [Key Takeaways](#13-key-takeaways)

---

## 0. Resume Bullet ↔ Proof Map

Three resume bullets. Every hard number below must be traceable to a section of this document. If an interviewer picks a bullet at random, jump straight to the "Where the proof lives" column.

| # | Resume bullet | Hard metric | Where the proof lives | Say this (one sentence) |
|---|---------------|-------------|----------------------|-------------------------|
| 1 | "Engineered a **six-layer** cold-chain risk platform ingesting **live shipment telemetry** and scoring pharmaceutical spoilage risk **in real time** using compliance rules and machine-learning signals." | 6 layers · live telemetry · real-time scoring | §2 Architecture (all six layers named), §7.1 Ingestion, §3 Risk engine | *"Telemetry lands in Supabase, a Realtime subscription wakes the scorer within a second or two, the hybrid rules-plus-ML engine tiers the window, and anything MEDIUM or above immediately enters the agent — six layers from raw row to human-reviewed decision."* |
| 2 | "Combined **14 engineered features** and **eight deterministic compliance rules** with **Optuna-tuned XGBoost** and **SHAP** explanations to generate interpretable, auditable risk scores." | 14 features · 8 rules · Optuna (30 trials, PR-AUC) · SHAP top-5 per decision | §3.1 Feature table, §3.2 Rule table, §3.3 Optuna, §3.4 SHAP, §3.5 Fusion | *"Eight hard rules encode FDA/GDP physics and can't be argued down; fourteen window features feed an Optuna-tuned XGBoost; the two fuse 0.4/0.6 with a safety veto, and every decision writes its SHAP top-five drivers to an immutable audit record."* |
| 3 | "Deployed a **FastAPI backend with 25 endpoints, WebSockets, React, Supabase, and Vercel**, integrating **four LLM providers with deterministic fallback** for resilient inference routing." | 25 endpoints · `/ws/events` · React 19 on Vercel · Supabase · 4 providers + zero-LLM path | §2 Tech stack, §7.2 WebSockets, §8.1 Provider chain, §8.2 Zero-LLM mode | *"FastAPI serves 25 endpoints plus a WebSocket event stream to a React dashboard on Vercel, and the LLM layer tries Groq, Ollama, OpenAI and Anthropic in priority order before falling through to a fully deterministic path — so the system still runs with no provider at all."* |
| — | **Achievement** | **1st prize + $4,000, Smith Agentic AI Challenge** | §1 STAR Result | *"It won first prize and $4,000 in the Smith Agentic AI Challenge, judged on combining FDA/GDP regulatory rules with explainable ML rather than on a demo."* |

> **Honest framing rule:** the numbers I stand behind are the ones on the resume — six layers, 14 features, 8 rules, 25 endpoints, 4 providers, 1st prize / $4,000, Mar–Apr 2026. The dataset is **synthetic** (7,411 telemetry windows), so I lead with that rather than waiting to be caught, and I never quote a model performance figure as if it were measured on real shipments. What I claim is the *system design*: the fusion logic, the governance, the fallback behaviour, and the audit trail. Where I name individual feature or hyperparameter details below, they're reconstructed from the pipeline — if pushed on one I'm not certain of, I say "that's from memory, directionally X."

---

## 1. STAR Summary

| Component | Detail |
|-----------|--------|
| **Situation** | Temperature-sensitive pharma logistics (vaccines, biologics) needs immediate, compliant, auditable decisions when a shipment starts drifting out of range. Manual intervention is slow and inconsistent for medium/high-risk excursions, and regulators (FDA 21 CFR, WHO, ICH, GDP) demand a defensible paper trail for every call. |
| **Task** | Build an end-to-end agentic platform that ingests live telemetry, scores spoilage risk reliably, and orchestrates mitigation (compliance logging, cold-storage backup, rerouting, notifications, insurance) — while keeping a human in control and every decision auditable. |
| **Action** | Built a 6-layer system: Supabase data pipeline → hybrid risk engine (14 features, 8 deterministic rules, XGBoost+Optuna+SHAP, rules/ML fusion with a safety veto) → context assembler → a **LangGraph** "act-first, always-review" orchestrator running 8 LangChain tools with cascade enrichment → RAG-backed compliance validation (pgvector) → FastAPI backend (25 endpoints + WebSocket) and a React dashboard. Provider-agnostic LLM layer (Groq/Ollama/OpenAI/Anthropic) with a full deterministic fallback so it runs with **zero LLM**. |
| **Result** | Live production demo; **won 1st prize and $4,000 in the Smith Agentic AI Challenge**, on the strength of **combining FDA/GDP regulatory rules with explainable ML** rather than on demo polish. Every MEDIUM+ event runs its mandatory tools, reflects on the real results, and pauses at a human-review gate; every decision writes an immutable SHAP-annotated audit record. Demonstrates practical agent orchestration with governed autonomy, not a demo-ware chatbot. Built solo over ~6 weeks (Mar–Apr 2026). |

### Why this project matters in interviews

It hits four things at once that most candidate projects don't: **(1)** agent orchestration that actually runs tools reliably and in parallel-ish sequence, **(2)** real statistics/ML with a "is this signal real or noise?" discipline, **(3)** human-in-the-loop governance, and **(4)** production concerns — fallbacks, audit, observability. It's the strongest story for *Applied AI Engineer, agentic AI, and AI-platform* roles.

---

## 2. Architecture

```
SUPABASE (cloud data platform)
  window_features (7,411 rows) | product_profiles | facilities | product_costs
  compliance_knowledge (pgvector) | compliance_docs (FDA/WHO/ICH PDFs)
        │
        ▼
LAYER 1 — DATA PIPELINE
  supabase_client.py  (paginated fetch, all with local JSON fallback)
  stream_listener      (Supabase Realtime → auto-score + orchestrate MEDIUM+)
        │
        ▼
LAYER 2 — RISK SCORING ENGINE  (pipeline.py)
  feature_engineering  → 14 derived features (rolling temp std, lag, slope,
                          delay ratio, progress %, humidity×temp, hours-to-breach…)
  deterministic_engine → det_score (0–1) from 8 hard rules
  predictive_model     → XGBoost + Optuna (30 trials, PR-AUC), SHAP per-row
  risk_fusion          → final = 0.4·det + 0.6·ML  (+ deterministic veto)
                          tier = LOW / MEDIUM / HIGH / CRITICAL
  compliance_logger    → immutable audit_YYYYMMDD.jsonl (SHAP top-5, snapshot)
        │  risk_input dict per window
        ▼
LAYER 3 — CONTEXT ASSEMBLER
  delay_ratio, delay_class, hours_to_breach, facility, product_cost
        │
        ▼
LAYER 4 — AGENTIC ORCHESTRATION  (LangGraph StateGraph)
  interpret → plan(LLM) →
     LOW    → output (monitoring only)
     MEDIUM+→ execute → observe(LLM) → reflect(LLM) →
                adequate → human_review → output
                gaps     → revise(LLM) → human_review → output
        │
        ▼
LAYER 5 — 8 AGENT TOOLS (LangChain StructuredTools)
  compliance(RAG) · notification(agentic) · cold_storage · scheduling
  insurance · route · triage · approval_workflow      (+ cascade enrichment)
        │
        ▼
LAYER 6 — BACKEND + DASHBOARD
  FastAPI (25 endpoints + /ws/events WebSocket)
  React 19 + Vite + Tailwind + Recharts + Mermaid
  (Overview, Monitoring, ShipmentDetail, AgentActivity, AuditLog, Approvals)
```

### Full tech stack

| Layer | Technology |
|-------|-----------|
| **Orchestration** | LangGraph (StateGraph), LangChain StructuredTools |
| **ML** | XGBoost, Optuna (hyperparameter search), SHAP (explainability), scikit-learn, pandas, numpy |
| **LLM** | Provider-agnostic: Groq (`llama-3.3-70b`), Ollama (`qwen2.5`), OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-5-haiku`) |
| **RAG** | Supabase pgvector, SentenceTransformers (`all-MiniLM-L6-v2`, 384-dim) |
| **Backend** | FastAPI, Pydantic, WebSocket, Uvicorn |
| **Data** | Supabase (Postgres + pgvector + Storage), local JSON/CSV fallback |
| **Frontend** | React 19, Vite, Tailwind v4, Recharts, Mermaid |
| **Deploy** | Vercel (frontend), containerized backend |

---

## 3. Risk Scoring Engine

This is the part interviewers probe hardest, because it's where "AI hype" meets "would you trust this with a $110k vaccine shipment?"

### 3.1 Feature engineering (14 features)

From raw telemetry windows: `rolling_temp_std_3`, `lag_temp_1`, `temp_deviation_from_mean`, `temp_slope_c_per_hr`, `progress_pct`, `delay_ratio`, `minutes_outside_range`, `humidity_x_temp` (interaction), `hours_to_breach`, etc. Split is **by shipment, not by row** — so a shipment's later windows never leak into training for its earlier ones (no temporal/entity leakage).

### 3.2 Deterministic engine (8 hard rules)

Rules encode domain physics/regulation and each contributes a bounded score:

| Rule | Score | Trigger |
|------|-------|---------|
| `temp_critical_breach` | 0.60 | outside critical limits |
| `temp_warning_breach` | 0.30 | outside normal limits |
| `temp_trend` | 0.20 | slope > 1°C/hr toward breach |
| `excursion_duration` | 0.30 | cumulative minutes > product tolerance |
| `freeze_risk` | 0.50 | freeze-sensitive product + temp ≤ 0°C |
| `delay_temp_stress` | 0.25 | delay > 120 min + near breach |
| `battery_critical` | 0.15 | sensor battery < 20% |
| `humidity_alert` | 0.10 | humidity > product threshold |

### 3.3 XGBoost predictor (6-hour-ahead spoilage)

- **Optuna, 30 trials, optimizing PR-AUC** (not accuracy) because spoilage is rare — `scale_pos_weight ≈ 5` for class imbalance.
- Reports **recall at fixed precision (≥0.70)** — the metric that matters when a false negative means a ruined shipment.
- **SHAP TreeExplainer** gives per-prediction top-k feature attributions, logged for every decision (FDA/GDP explainability requirement).

### 3.4 Risk fusion — the trust layer (the code interviewers love)

```python
# final = alpha·deterministic + (1-alpha)·ML, with a hard safety veto
blended = 0.4 * det_score + 0.6 * ml_score
if det_score >= 0.8:                       # a hard rule fired strongly
    final = max(det_score, blended)        # ML can't talk the score DOWN
else:
    final = blended
# NaN handling: missing score defaults to the other; both missing → 0.5 (MEDIUM)
```

Three design decisions worth stating out loud:
1. **Deterministic veto.** A benign ML prediction can never override a hard safety rule. Model judgment is *bounded* by rules that can't be argued down.
2. **"Unknown" is loud, not silent.** A missing/NaN score defaults to **MEDIUM (escalate)** — never "LOW/safe". The dangerous failure mode is confident silence.
3. **Tiers drive actions + human gate.** `TIER_THRESHOLDS` map score→tier; `HUMAN_APPROVAL_TIERS = {HIGH, CRITICAL}`; MEDIUM+ always reviewed.

---

## 4. Agentic Orchestration (LangGraph)

### 4.1 Topology — "Act-First, Always-Review" HITL

```
interpret → plan → [tier route]
  LOW    → output                         (monitoring only, bypasses human gate)
  MEDIUM+→ execute → observe → reflect → [should_revise]
             adequate → human_review → output
             gaps     → revise → human_review → output
```

State is a shared `OrchestratorState` TypedDict. Each node is chosen at build time to be **agentic (LLM) or deterministic** depending on whether an LLM is available — so the whole graph degrades gracefully to a template-based pipeline with zero LLM.

### 4.2 Node responsibilities

| Node | Mode | Job |
|------|------|-----|
| `interpret` | deterministic | Map risk_tier/score/rules to severity, urgency, primary_issue |
| `plan` | LLM (or template) | Produce a `draft_plan` of tool steps; LLM prompt has condensed tool schemas + domain context (GDP, FDA 21 CFR, WHO, ICH) |
| `execute` | deterministic | Run tools sequentially with **cascade enrichment** + dependency tracking |
| `observe` | LLM | Summarize what actually ran (success/failure per tool) |
| `reflect` | LLM | Compare results to the **mandatory tools for this tier**; flag only *mandatory* gaps |
| `revise` | LLM | Propose *corrective-only* steps (hard-filtered to `mandatory ∪ failed`) |
| `human_review` | deterministic | Always fires for MEDIUM+; creates an approval with full reasoning trace |
| `fallback` / `output` | deterministic | Backup plan on errors; compile final decision JSON + confidence |

### 4.3 The design lesson (my favorite talking point)

The earliest version was "plan → maybe-loop → act." The reflect→revise loop kept **re-triggering already-succeeded tools** and proposing *optional* tools (rerouting, triage) as "gaps" — the LLM was *helpfully hallucinating work*. On the dashboard it looked busy, so it was easy to miss: **technically functional but untrustworthy.** Two fixes:

1. **A hard filter in code, not the prompt.** `revise` can only emit tools in `mandatory_set ∪ failed_set`; anything else is rejected before execution. I stopped trusting a prompt to constrain behavior code could guarantee.
2. **Redesigned to "act-first, always-review."** MEDIUM+ runs once, *observes* real results, reflects on those, and the human gate fires every time. That killed the phantom re-execution and made runs reproducible.

> Lesson: *an agent that looks busy is not an agent that's correct; the cheapest guardrail is usually deterministic code wrapping the model, not a longer prompt.*

### 4.4 Human-in-the-loop, post-graph

The graph always terminates at `human_review` (no autonomous consequential action). The human then acts **outside** the graph via endpoints:
- `POST /api/approvals/{id}/confirm` — first pass adequate, close (no re-execution).
- `POST /api/approvals/{id}/execute` — run human-selected corrective tools via `run_orchestrator_selective()`, which bypasses the LLM plan entirely (interpret → execute → observe → compile).

---

## 5. The 8 Tools + Cascade Enrichment

| Tool | What it does |
|------|-------------|
| `compliance_agent` (RAG) | Logs immutable audit event, retrieves regulations via pgvector, LLM interprets → decision/disposition/citations |
| `cold_storage_agent` | Scores facilities by temp-compatibility × distance × capacity × urgency; returns best + alternatives |
| `notification_agent` (agentic) | LLM selects stakeholders + composes multi-channel messages (email/Slack/SMS/dashboard) with FDA 21 CFR Part 11 audit entries |
| `scheduling_agent` | Reschedule/reroute decision + financial impact + priority score |
| `insurance_agent` | Drafts a claim with a computed loss breakdown (product × spoilage prob + disposal + downstream) |
| `route_agent` | Temp-class-aware carrier/route recommendation (LLM picks among *safe* candidate options only) |
| `triage_agent` | Ranks many shipments by urgency for processing order |
| `approval_workflow` | Creates the human approval record |

### Cascade enrichment (why it's more than "call 8 functions")

`execute()` passes each tool's output into a `cascade_context`, and downstream tools get enriched inputs — e.g. `cold_storage_agent`'s chosen facility + advance-notice flow into `notification_agent` and `scheduling_agent`; `compliance_agent`'s `log_id` becomes `insurance_agent`'s supporting evidence; `estimated_loss = product_cost × spoilage_prob`. There's an explicit `_DEPENDS_ON` map, and if an upstream tool fails, the downstream tool receives an injected `_upstream_warning` instead of silently using stale data.

---

## 6. RAG Compliance Sub-system

- **Corpus:** FDA 21 CFR, WHO TRS, ICH, GDP PDFs in Supabase Storage → chunked (500 words, 50 overlap) → embedded (`all-MiniLM-L6-v2`, 384-dim) → stored in `compliance_knowledge` (pgvector).
- **Retrieval:** Supabase RPC `match_compliance_documents()` (cosine), with a brute-force cosine fallback, and a mock-regulations fallback (6 hardcoded FDA/ICH/WHO/GDP rules with keyword scoring) so it never hard-fails.
- **Interpretation:** Groq LLM turns *shipment context + retrieved regulations* into a JSON decision: `compliance_decision`, `severity`, `human_approval_required`, `approval_level`, `product_disposition` (release/quarantine/destroy/investigate), `violated_regulations`, `required_actions`, `reasoning`. Deterministic tier-based fallback if the LLM is unavailable.
- Always writes an immutable `compliance_events.jsonl` audit record first (GDP-compliant), regardless of what the LLM decides.

---

## 7. Real-Time Telemetry & the Live Dashboard

This is the layer the first resume bullet means by **“live shipment telemetry”** and **“real time.”** Be precise in interviews: real time here is **event-driven scoring on each incoming window**, not a hard 10 ms SLA and not a claim that the dataset is from a live carrier.

### 7.1 Ingestion

- Telemetry windows land in Supabase (`window_features`). A **Realtime** subscription (`stream_listener`) wakes on insert/update, runs the hybrid scorer, and — if the fused tier is MEDIUM or above — hands the window to the LangGraph orchestrator.
- Every Supabase fetch has a **local JSON/CSV fallback**, so the demo still scores offline. Swapping a real carrier feed is a client/config change, not a rewrite of the fusion or agent graph.
- Dataset honesty: **7,411 synthetic windows**. Lead with that. What is real is the *pipeline*: ingest → score → (maybe) agent → audit → dashboard.

### 7.2 WebSockets, dashboard, and the 25 endpoints

- FastAPI exposes **25 HTTP endpoints** plus **`/ws/events`**. The WebSocket pushes score updates, agent-node transitions, and approval-state changes to the React 19 dashboard (Overview, Monitoring, ShipmentDetail, AgentActivity, AuditLog, Approvals).
- Endpoint families (do not invent a fake numbered list of all 25 in the interview): scoring / shipments, orchestrator run + selective re-run, approvals confirm/execute, LLM configure/status, audit/compliance reads, health. If asked for the exact count, *“twenty-five plus the WebSocket — I can walk the families; I will not recite a fabricated route table.”*
- “Within a second or two” is the **demo-path** feel from Realtime → scorer → UI event. I do not quote a production p95 unless I have measured it on a real feed.

---

## 8. Reliability & LLM Provider System

- **Provider priority + failover:** `CARGO_LLM_PRIORITY="groq,ollama,openai,anthropic"`. `get_llm()` returns the first working ChatModel; the graph recompiles when the provider changes; hot-reconfigurable at runtime via `POST /api/llm/configure`.
- **Zero-LLM mode:** every agentic node has a deterministic twin. Set `CARGO_LLM_ENABLED=0` and the whole system still scores, plans (templates), executes, reflects (checklist), and reviews. This is the single best "production judgment" signal in the project.
- **Local fallback everywhere:** every Supabase fetch falls back to local JSON/CSV; the demo runs offline.
- **Confidence scoring:** compiled output attaches a calibrated confidence (0.95 LOW/no-tools, 0.80 adequate-awaiting-confirm, 0.70 corrections-pending, 0.50 partial-failure).

---

## 9. Topics You Must Know

- **PR-AUC vs ROC-AUC & recall@precision** — why PR-AUC for rare spoilage, why report recall at fixed precision. (See `learning/14_evaluation_metrics.md`.)
- **Class imbalance** — `scale_pos_weight`, threshold selection, why accuracy is useless here.
- **SHAP** — TreeExplainer, additive attributions, using top-k drivers as the audit "why".
- **Rules + ML fusion** — when to blend, why a veto, guarding against silent NaN.
- **LangGraph** — StateGraph, nodes/edges, conditional routing, shared typed state, HITL gates. (See `learning/17_langchain_langgraph.md`, `10_agentic_ai_and_multi_agent.md`, `27_agentic_patterns_deep_dive.md`.)
- **Agentic patterns** — plan/act/observe/reflect/revise, tool-calling with schema validation, cascade/dependency handling.
- **RAG** — chunking, embeddings, pgvector cosine search, grounding an LLM decision. (See `05_rag_systems.md`, `06_vector_databases.md`.)
- **Temporal/entity leakage** — split-by-shipment; walk-forward thinking.
- **Human-in-the-loop governance** — when autonomy is/ isn't appropriate; audit trails.

---

## 10. Interview Q&A

**Q1: Walk me through AI Cargo end to end.**
> A shipment streams telemetry windows into Supabase. Layer by layer: I engineer 14 features, score each window with 8 deterministic safety rules and an XGBoost model, then fuse them — `0.4·rules + 0.6·ML` with a hard veto so a strong rule can't be overridden. The fused score maps to a tier. LOW is monitoring-only; MEDIUM and above enter a LangGraph agent that plans tool actions, executes them, observes the real results, reflects against the mandatory tools for that tier, optionally revises, and then always pauses at a human-review gate. Eight tools handle compliance, cold-storage, rerouting, scheduling, insurance, notification, triage, and approvals, with cascade enrichment passing one tool's output into the next. Compliance is RAG-backed over FDA/WHO/ICH docs in pgvector. Everything writes an immutable SHAP-annotated audit record, and the whole thing degrades to a deterministic, zero-LLM pipeline if a provider is down.

**Q2: Why fuse rules and ML instead of just using the model?**
> Trust and safety. The ML model catches subtle, learned patterns the rules miss, but I never let it be the only vote on a $100k biologic. The deterministic veto guarantees a hard safety rule can't be talked down by a benign prediction, and rules give me an auditable, regulator-friendly reason even when the model is uncertain. It's the same philosophy as a fraud system with hard blocklists on top of a score.

**Q3: Why PR-AUC and not accuracy?**
> Spoilage is rare, so a model that predicts "fine" every time scores high accuracy and is useless. PR-AUC focuses on the positive (spoilage) class, and I actually select the operating threshold by recall at a fixed precision of 0.70 — because a false negative (missed spoilage) is far more expensive than a false positive (an unnecessary check). I also set `scale_pos_weight` for the imbalance and tuned with Optuna on PR-AUC directly.

**Q4: How does the LangGraph agent avoid doing dumb things?**
> Three guardrails. First, act-first-always-review: MEDIUM+ events always end at a human gate, so no consequential action is autonomous. Second, a hard code filter — the revise step can only emit mandatory or failed tools, so the LLM can't invent unnecessary work even if it wants to. Third, the reflect step is scoped to a fixed mandatory-tool list per tier, so optional tools are never flagged as "gaps." I learned all three the hard way when an early version kept re-running succeeded tools and looked busy without being correct.

**Q5: What was the hardest bug?**
> The reflect→revise loop re-triggering already-succeeded tools and proposing optional tools as gaps. It was invisible because the dashboard just looked active. I instrumented every node's input/output, traced it to the LLM "helpfully" flagging optional tools, and fixed it with a deterministic hard filter plus the act-first redesign. The meta-lesson: constrain agent behavior in code, not in the prompt, whenever the code *can* guarantee it.

**Q6: How would you sanity-check whether a risk signal is real?**
> Never let the model be the only vote (veto + rules); make "unknown" escalate rather than default safe; demand an inspectable explanation (SHAP top-5); read PR-AUC/recall-at-precision rather than accuracy; split by entity to rule out leakage; and put a human on the consequential path. Same discipline I'd apply to an apparent conversion lift — check power, segment it, rule out an instrumentation bug before believing the number.

**Q7: Why LangGraph over a plain agent loop or LangChain AgentExecutor?**
> I wanted explicit, inspectable control flow with a typed shared state and conditional edges — LangGraph gives me a real state machine I can diagram, gate, and make deterministic node-by-node. A free-form ReAct loop is harder to bound, harder to audit, and harder to force a human gate into. For a governed, regulated workflow, explicit topology beats emergent behavior.

**Q8: How do you make it production-reliable?**
> Provider-agnostic LLM with priority failover; a full deterministic twin for every agentic node so it runs with zero LLM; local JSON/CSV fallback for every Supabase call; immutable append-only audit logs; calibrated confidence on every decision; and a WebSocket event stream for live observability. The design goal was "the demo still works on a plane with no internet and no API keys."

**Q9: How does the RAG compliance check work, and how do you keep it honest?**
> Regulatory PDFs are chunked, embedded with all-MiniLM-L6-v2, and stored in pgvector. For an event I retrieve the top regulations by cosine similarity and have the LLM produce a structured decision grounded in those citations. To keep it honest: I always write the immutable audit log first regardless of the LLM, I fall back to a deterministic tier-based decision if the LLM fails, and the disposition (quarantine/destroy/etc.) plus approval level is surfaced to the human — the LLM proposes, the human disposes.

**Q10: What would you improve with more time?**
> Persist orchestrator/approval state in Postgres instead of in-memory; add proper auth (JWT) on the endpoints; move to `pgvector` HNSW as the compliance corpus grows; add a small eval harness for the agent (golden incidents → expected tool sets) so I can regression-test prompt/model changes; and calibrate the XGBoost probabilities (Platt/isotonic) so the fused score is a true probability.

---

## 11. Trick & Follow-Up Questions

These are the adversarial follow-ups after the 30-second pitch. Answer the **definition** first, then the **caveat**. Do not invent live-shipment metrics.

---

#### T1: "You said real time. The data is synthetic. So is it real time?"

> "The *pipeline* is real time: a Supabase Realtime insert wakes the scorer, MEDIUM+ enters the agent, and the dashboard gets a WebSocket event. The *dataset* is synthetic — 7,411 windows. I lead with that. Real time is an architecture claim, not a claim that I instrumented a live carrier. Swapping a real feed is a client change; the fusion, veto, and HITL graph stay."

---

#### T2: "0.4 / 0.6 — did you tune that, or did you pick it because it looks balanced?"

> "It's a starting prior, not a fitted meta-model. Rules get 0.4 so they always vote; ML gets 0.6 so subtle patterns can move the score. The **veto** is the safety mechanism: if `det_score ≥ 0.8`, ML cannot talk the score down. With labelled outcomes I would learn the blend (stacking or a logistic meta-model) and calibrate. I will not pretend 0.4/0.6 was Optuna's answer."

---

#### T3: "SHAP with correlated temperature features — aren't the attributions lying?"

> "TreeSHAP is still additive, but when `rolling_temp_std_3`, `temp_slope_c_per_hr`, and `minutes_outside_range` move together, credit gets split in a way that is not unique. I use SHAP as an **audit why** — top-five drivers for the human — not as a causal ranking. The rules table is the causal/physics layer; SHAP is the model's local explanation. If two features are near-duplicates I would drop one before training, not after explaining."

---

#### T4: "Why PR-AUC and not ROC-AUC? ROC looks prettier."

> "Spoilage is rare. ROC-AUC can look strong while the model is useless on the positive class — lots of true negatives inflate the false-positive rate denominator. PR-AUC is precision vs recall on the spoilage class. I also pick the operating point by **recall at precision ≥ 0.70**, because a missed spoiled shipment is the expensive error. Optuna optimized PR-AUC, not accuracy, not ROC."

---

#### T5: "XGBoost doesn't relearn online. What happens when a new product or route shows up?"

> "It doesn't. This is a batch-trained model on historical windows, scored online. A new product with a different temp band is handled first by the **deterministic rules** and product profile, not by hoping the trees generalize. I would retrain on a schedule (or when PSI / PR-AUC on a sliding window drops), with a shipment-level split so we don't leak. I never claimed online learning."

---

#### T6: "Telemetry arrives out of order. Does your rolling std still mean anything?"

> "If I compute rolling features on arrival order, a late window corrupts the slope and the 3-window std. The correct key is **event time**, not ingest time: buffer, order by timestamp within shipment, then roll. For the demo, windows are already ordered. In a real feed I would watermark and hold a short buffer, and the rules (`minutes_outside_range`, freeze) still fire on the raw point even if the ML window is incomplete — unknown ML score escalates to MEDIUM, it does not fail open."

---

#### T7: "25 endpoints — prove it. Name them."

> "I will not recite a fabricated list of twenty-five paths. I will walk the **families**: scoring and shipments, orchestrator run and selective re-run, approvals confirm/execute, LLM configure/status, audit and compliance reads, health, plus `/ws/events`. The resume number is the FastAPI surface area of the repo. If you want to count, the live demo and GitHub are the source of truth."

---

#### T8: "How would you A/B a change to the risk model in a regulated cold-chain?"

> "You do not randomly assign live vaccine shipments to a riskier model. Shadow first: new model scores every window, old model still drives the agent, you compare PR-AUC, recall@0.70, tier disagreement rate, and veto-fire rate. Then a **canary** on a low-cost product class with a human still on MEDIUM+. Promote only if disagreement is reviewed and the expensive errors (missed HIGH/CRITICAL) do not increase. Same discipline as a credit-policy change — not a consumer-button A/B."

---

#### T9: "LangGraph vs a for-loop that calls tools. Why is this not over-engineered?"

> "The graph is the audit surface: typed state, conditional edges, a forced human gate, and a deterministic twin per node. A for-loop can call the same eight tools; it cannot show a reviewer *which node decided what* or fail over to zero-LLM without a second codebase. For a regulated workflow I want a state machine I can diagram, not an emergent ReAct loop."

---

#### T10: "Four LLM providers — isn't that resume padding?"

> "The point is **failover and a zero-LLM path**, not that I need four brands. Priority is Groq → Ollama → OpenAI → Anthropic, hot-reconfigurable. If all four are down, `CARGO_LLM_ENABLED=0` still scores, templates a plan, executes tools, checklists the reflect step, and waits on the human. The prize story is governed autonomy, not provider count."

---

#### T11: "NaN → 0.5 MEDIUM. Isn't that just guessing?"

> "It's a **fail-loud** default, not a prediction. Missing det and missing ML both mapping to 0.5 means 'escalate, don't silently mark safe.' The dangerous bug is fail-open. A human then sees 'unknown scores' in the audit, not a green LOW."

---

#### T12: "Would you let this destroy a shipment autonomously if the model is 0.99?"

> "No. Disposition is proposed, not executed. HIGH/CRITICAL always hit `human_review`. A 0.99 from a model trained on synthetic windows is the last thing I would let touch a destroy action. Autonomy is earned on a real labelled feed, with shadow mode, then canary, then still a human on destroy/quarantine."

---

## 12. Red Flags & How to Handle

| Red flag | How to handle |
|----------|---------------|
| "It's a synthetic dataset." | Agree openly — the *system design, fusion logic, agent governance, and audit* are the contribution; the data is a stand-in for a real telemetry feed and swapping it in is a config change. |
| "In-memory approvals don't scale." | Correct; it's demo-grade. Persist to Postgres; the orchestrator already writes durable audit logs, so the durable source of truth exists. |
| "Isn't the LLM the weak link?" | That's exactly why every agentic node has a deterministic twin, tool calls are schema-validated and hard-filtered, and a human gates all consequential actions. The LLM improves quality; it's never the safety mechanism. |
| "Why not fully autonomous?" | Regulated cold-chain: a wrong autonomous "destroy" or "reroute" is expensive and non-compliant. Autonomy is *earned* by being right over time; I default to human-gated. |
| "0.4/0.6 fusion weights seem arbitrary." | They're a starting prior; the veto is the real safety net. With labeled outcomes I'd learn the weights (stacking/logistic meta-model) and calibrate. |

---

## 13. Key Takeaways

- **What it demonstrates:** governed agent orchestration (LangGraph) + trustworthy hybrid ML (rules+XGBoost+SHAP, PR-AUC) + RAG compliance + human-in-the-loop + real production reliability (provider failover, zero-LLM fallback, immutable audit).
- **Signals:** you can build agents that *run tools reliably*, you understand *when not to trust a model*, and you think about *audit, fallback, and observability* — not just the happy path.
- **Best-fit roles:** Applied AI Engineer, Agentic AI, AI Platform, Forward-Deployed Engineer.
- **30-second pitch:** *"AI Cargo is an agentic risk platform for pharma cold-chain. A hybrid rules+XGBoost engine scores spoilage with a safety veto, a LangGraph agent plans and runs eight tools with cascade enrichment, and every medium-or-higher event pauses for a human with a full SHAP-annotated audit trail. It runs provider-agnostic and degrades to a deterministic zero-LLM pipeline — it won first prize in the Smith Agentic AI Challenge."*
