# P17: PathWise — AI Career-Readiness Platform with a Multi-Agent Career Simulator

> **Project Type:** Flagship self-project | **Agents League · Creative Apps (GitHub Copilot) submission**
> **Live:** [pathwise-jade.vercel.app](https://pathwise-jade.vercel.app/) | **GitHub:** [github.com/rahul370139/PathWise](https://github.com/rahul370139/PathWise)
> **Relationship to P08:** PathWise is the **productionized evolution of SkillSpring/TrainPI (P08)**. This note focuses on what's *new and interview-worthy* — the **Career Simulator multi-agent loop**, **Microsoft Foundry IQ hybrid retrieval**, and the **offline eval harness**. For the shared foundations (PDF distiller, chunking, map-reduce summary, RIASEC career matching, LRU cache, robust JSON parsing), read **[P08_skillspring_ai_learning_platform.md](P08_skillspring_ai_learning_platform.md)** — they carry over unchanged.

**One-line pitch:** *Drop in a resume + a job posting and PathWise runs a live, inspectable multi-agent mock interview — a team of named agents plans your competency gaps, grounds each question in a citation-backed knowledge base (Microsoft Foundry IQ, with a Supabase pgvector failover), adapts difficulty to your last answer, auto-builds a micro-lesson for anything you're weak on, and finishes with a 30/60/90 readiness report — with an offline eval harness proving groundedness and refusal correctness.*

---

## Table of Contents

1. [STAR Summary](#1-star-summary)
2. [What's New vs SkillSpring (P08)](#2-whats-new-vs-skillspring-p08)
3. [Architecture](#3-architecture)
4. [The Career Simulator — Multi-Agent Loop](#4-the-career-simulator--multi-agent-loop)
5. [Microsoft Foundry IQ — Hybrid Retrieval with Failover](#5-microsoft-foundry-iq--hybrid-retrieval)
6. [Reliability: Safety Guard + Offline Eval Harness](#6-reliability-safety-guard--offline-eval-harness)
7. [Resume → Upgrade Plan Pipeline](#7-resume--upgrade-plan-pipeline)
8. [Topics You Must Know](#8-topics-you-must-know)
9. [Interview Q&A](#9-interview-qa)
10. [Red Flags & How to Handle](#10-red-flags--how-to-handle)
11. [Key Takeaways](#11-key-takeaways)

---

## 1. STAR Summary

| Component | Detail |
|-----------|--------|
| **Situation** | Learners have content tools *or* career advice, never a single grounded system that assesses you against a real job posting and closes the gaps. And most "AI interview" demos hide their reasoning and hallucinate confidently — no grounding, no evaluation, no honest refusals. |
| **Task** | Build a career-readiness platform whose signature feature is a **transparent, grounded, multi-agent mock interview**: resume + JD → competency gap analysis → adaptive interview → auto-remediation → readiness report — with visible agent reasoning, citation-backed answers, and a measurable reliability story (built for the Agents League Creative Apps rubric). |
| **Action** | Refactored the SkillSpring backend into a clean `pathwise/` package (`learn/`, `career/`, `dashboard/`, `infra/`, `eval/`). Built `career_simulator.py` — six named agents (Planner, Retrieval, Interviewer, Scorer, Remediation, Report) over a per-session **SSE event bus** that streams an "Agent Thinking" timeline. Added **Microsoft Foundry IQ** (Azure AI Search) as citation-first retrieval with **Supabase pgvector as a transparent failover**, and an offline **eval harness** (`eval_simulator.py`) that reports groundedness, refusal correctness, answer relevance, and p95 latency. Deployed FastAPI on a Hostinger VPS (Docker) + Next.js 14 on Vercel. |
| **Result** | Live public demo. A user uploads a resume + JD and watches agents plan → retrieve (with citation chips) → ask adaptive questions → score with supported/unsupported flags → auto-generate micro-lessons for weak areas → produce a 30/60/90 readiness report. The dashboard renders an **eval card** from the offline harness. Grounding guard refuses out-of-scope/too-short inputs instead of hallucinating. |

---

## 2. What's New vs SkillSpring (P08)

| Dimension | SkillSpring (P08) | PathWise (P17) |
|-----------|-------------------|----------------|
| **Signature feature** | PDF → learning content + RIASEC career match | **Career Simulator**: resume+JD → multi-agent mock interview → readiness report |
| **Agents** | Router/Summarizer/Diagnostic (single-shot helpers) | **6 named agents** in an inspectable PLAN→RETRIEVE→GENERATE→VERIFY→REMEDIATE→FINALIZE loop with SSE streaming |
| **Retrieval** | In-memory cosine over PDF chunks | **Foundry IQ (Azure AI Search) citation-first**, Supabase **pgvector RPC** failover, `interview_prep_kb` corpus |
| **Reliability proof** | Resilience patterns (fallbacks, retries) | **Offline eval harness** → groundedness / refusal-correctness / relevance / p95 latency → dashboard eval card |
| **Safety** | JSON repair, graceful degradation | **SafetyGuard**: scope refusal + grounding threshold (won't answer without ≥200 chars of grounded context) |
| **Packaging / deploy** | Monolith `main.py` on Railway | Clean `pathwise/` package; FastAPI on **Hostinger VPS (Docker)** + Next.js 14 on Vercel; runtime `/api/*` proxy |
| **Career flow** | Quiz → match → roadmap | Adds resume-driven **`/resume/parse` → `/plan/build` → `/upgrade`** single-shot O*NET-grounded plan pipeline + snapshot persistence |

> In an interview: lead with the Career Simulator and Foundry IQ grounding + eval harness. Mention P08 as the foundation ("the distiller, chunking, map-reduce summary, and RIASEC matching carry over") rather than re-explaining them.

---

## 3. Architecture

```
Browser (Next.js 14 on Vercel)
  Learn · Career Simulator · Career · Dashboard
        │  same-origin /api/* proxy (avoids HTTPS→HTTP mixed content)
        ▼
FastAPI on Hostinger VPS (backend/main.py, ~1,800 lines)
  /api/chat · /api/distill · /api/simulator/{start,answer,stream,report,eval}
  /api/career/{resume/parse, plan/build, upgrade, match} · /api/dashboard/* · /api/rag/{status,probe}
        │
        ▼  pathwise/ package
  learn/distiller.py     — chunk · Cohere embed · Groq gen · intent dispatch (from P08)
  learn/rag_kb.py        — retrieve() indirection: Foundry IQ → Supabase failover
  infra/foundry_iq.py    — Microsoft Foundry IQ (Azure AI Search) client
  career/career_simulator.py — 6-agent loop + SSE SessionBus
  career/resume_career.py    — resume parse + single-shot O*NET-grounded plan
  career/career_matcher.py   — RIASEC + Cohere similarity over O*NET (from P08)
  eval/eval_simulator.py     — offline golden-set reliability harness
  dashboard/{dashboard,mastery}.py + infra/supabase_helper.py
        │
        ▼  Models + Knowledge + Data
  Groq (llama-3.3-70b)   Cohere (embed-english-light-v3, 384-dim)
  Microsoft Foundry IQ (Azure AI Search, prepkb-index)  ⇄  Supabase pgvector (interview_prep_kb)
  Supabase Postgres (lessons, cards, progress, mastery, roles)  ·  O*NET CSV  ·  plan snapshots
```

---

## 4. The Career Simulator — Multi-Agent Loop

`career/career_simulator.py`. The design principle written into the module docstring: **"REUSE, don't rewrite."** Every agent delegates to code that already exists (resume parser, distiller, rag_kb) — the orchestrator is thin glue plus an event bus. That's the maturity signal: adding a headline feature *without* duplicating business logic.

### The loop

```
PLAN → RETRIEVE → GENERATE(adaptive question) → VERIFY(score)
     → (remediate if weak) → advance → … → FINALIZE(readiness report)
```

| Agent | Responsibility | Delegates to | UI event (SSE) |
|-------|----------------|--------------|----------------|
| **Planner** | Diff resume vs JD → 5–7 competencies with `gap_level` 0–3 → focus the interview on the real gaps (gap_level ≥1, hardest first) | `parse_resume`, `_find_onet_row`, `_onet_summary` | "Here's what I'll test you on" + readiness estimate |
| **Retrieval** | Ground each competency in the KB | `rag_kb.retrieve()` → Foundry IQ / Supabase | Citation chips + grounded/low-grounding flag |
| **Interviewer** | Ask ONE question; **difficulty adapts to prior score** (≥75 → harder; 45–74 → same, new angle; <45 → easier/foundational) | Groq via `call_groq` | Adaptive question |
| **Scorer** | Score 0–100 with `supported` (is the score backed by the grounded rubric?), strengths, missing points | Groq judge prompt | Score bar + supported/unsupported |
| **Remediation** | If score < 60, auto-build a micro-lesson + 3-item quiz from the same grounded context | `map_reduce_summary`, `gen_flashcards_quiz` | "Micro-lesson ready" → `/learn?topic=` |
| **Report** | Average only *answered* competencies; skipped ones become unproven gaps; build the roadmap | `build_career_plan` | 30/60/90 readiness report |

### Implementation details worth citing

- **SSE event bus (`SessionBus`)** is append-only with **replay** — a late subscriber gets the full history so the "Agent Thinking" timeline is complete even if you open the stream mid-run. Every agent emits `running`/`done`/`skipped` events; this is what turns hidden reasoning into a visible, rubric-scoring UX.
- **Adaptive difficulty is explicit code**, not a vague prompt: the Interviewer branches on the previous ScorerAgent score.
- **Skips are first-class:** a candidate can skip any question with an optional comment; skipped competencies aren't scored but resurface in the report as *unproven gaps* so the plan still adapts.
- **JD-optional:** `synthesize_jd()` builds a grounded JD-equivalent from a target role + interests via the O*NET row, so users who only know the role they want can still run the full flow.
- **State is in-memory per session** (`SESSIONS` dict) — demo-grade, mirroring the conversation-store pattern; explicitly flagged as "move to durable store for multi-instance."

---

## 5. Microsoft Foundry IQ — Hybrid Retrieval

The retrieval indirection lives in `learn/rag_kb.py::retrieve()` — everything calls that, nothing calls the providers directly.

```
retrieve(query)
  ├── FOUNDRY_SEARCH_* configured → Foundry IQ (Azure AI Search, prepkb-index)   [citation-first]
  └── env missing OR error       → Supabase pgvector RPC match_interview_prep()   [transparent failover]
        → grounded context + citations (doc, section, source_uri, score)
```

- **Corpus** (`backend/knowledge_base/`): `learning/` (technical prep), `behavioral/` (STAR), `onet_careers/` (271 O*NET briefs), `projects/` (portfolio deep-dives). Pushed to Foundry with `push_to_foundry.py --create-index`; ingested to Supabase with `python -m pathwise.learn.rag_kb ingest`.
- **Supabase side** (the failover I fully control): `interview_prep_kb` table, `vector(384)` to match Cohere `embed-english-light-v3`, **IVFFlat** index (`lists=100`, `vector_cosine_ops`), server-side RPC `match_interview_prep` returning `1 - cosine_distance`. `chunk_hash UNIQUE` makes ingestion **idempotent** — editing one markdown file only re-embeds changed chunks.
- **Observability**: `GET /api/rag/status` (last provider + Foundry DNS check + fix hints) and `GET /api/rag/probe?q=...` (live retrieval with `retrieval_provider` per chunk). When Foundry is down, `retrieve()` logs a warning and uses Supabase — **the demo never stops**. That failover *is* the reliability story.
- **Why 384-dim / IVFFlat**: same embedding model as the PDF-chunk path so query vectors are interchangeable; IVFFlat is right for a few-thousand-row table — documented upgrade path to HNSW past ~50k chunks.

---

## 6. Reliability: Safety Guard + Offline Eval Harness

**SafetyGuard** (keeps the agent honest):
- `check_scope()` — refuses a too-short JD (<40 chars) or an unreadable/scanned resume (no skills) *before* any LLM call.
- `grounding_ok()` — requires ≥200 chars of retrieved context; low grounding is surfaced as a flag, and the Scorer marks answers `supported=false` when the rubric is empty rather than fabricating a grade.

**Offline eval harness** (`eval/eval_simulator.py`) — this is the differentiator most candidate agent projects lack. It runs a small **golden set** (a strong-fit case, a partial-fit case, and an out-of-scope case) through the real agents and writes `data/eval/simulator_eval_report.json`, surfaced by `GET /api/simulator/eval` and the dashboard eval card:

| Metric | What it proves |
|--------|----------------|
| `groundedness` | fraction of scored answers the retrieved rubric actually supports |
| `refusal_correctness` | out-of-scope inputs are refused, not answered |
| `answer_relevance` | the Planner produces JD-grounded competency gaps |
| `latency_p95_sec` | end-to-end responsiveness (p95) |

Being able to say *"here's how I measure whether my agent is grounded and whether it refuses when it should"* is worth more than any single feature.

---

## 7. Resume → Upgrade Plan Pipeline

`career/resume_career.py`, endpoints `/api/career/resume/parse` → `/plan/build` → `/upgrade`:

1. **Parse** — resume PDF → text → Groq extracts a structured profile (skills, experience, projects, education) as JSON.
2. **Anchor to O*NET** — `_find_onet_row(target_role)` with three fallbacks (exact title → substring → word-overlap score) so even off-catalog roles like "MLOps Engineer" anchor to a credible salary/growth/skills row.
3. **Single-shot plan** — one Groq call produces gap analysis + 3-stage roadmap + 90-day plan + project suggestions, then the O*NET salary band is stitched into `market_insights`.

Why one LLM call instead of three: ~3× cheaper, more coherent output (the model sees the gap analysis and the plan it's writing in the same context), and one JSON to validate. Snapshots persist via `career_plan_storage.py` for restore UX.

---

## 8. Topics You Must Know

- **Multi-agent orchestration** — named agents, a shared session bus, plan/retrieve/generate/verify/remediate/report; delegation over duplication. (`learning/10_agentic_ai_and_multi_agent.md`, `27_agentic_patterns_deep_dive.md`.)
- **SSE streaming + replay** — how to surface agent reasoning live; why replay matters for late subscribers.
- **Hybrid RAG with failover** — provider indirection, citation-first retrieval, `vector(384)`, IVFFlat vs HNSW, idempotent ingest via chunk hashing. (`learning/05_rag_systems.md`, `06_vector_databases.md`.)
- **LLM-as-judge / evaluation** — scoring rubrics, `supported` flags, groundedness, refusal correctness, golden-set eval, p95 latency. (`learning/14_evaluation_metrics.md`, `31_ai_safety_guardrails_responsible_ai.md`.)
- **Grounding & safety guards** — scope refusal, minimum-grounding thresholds, honest "insufficient evidence."
- **RIASEC + embedding career matching** — carried from P08 (`career_matcher.py`).
- **Adaptive difficulty** — feedback loop from score → next-question difficulty.
- **Deployment** — VPS Docker + Vercel, same-origin proxy to dodge HTTPS/HTTP mixed content.

---

## 9. Interview Q&A

**Q1: Walk me through the Career Simulator.**
> You upload your resume and paste a target job posting. A Planner agent parses the resume, pulls the matching O*NET reference, and diffs the two into 5–7 competencies with gap levels, focusing the interview on the real gaps hardest-first. Then it loops: a Retrieval agent grounds each competency in a citation-backed knowledge base, an Interviewer agent asks one question whose difficulty adapts to your previous score, a Scorer grades your answer 0–100 and flags whether the score is actually supported by the retrieved rubric, and if you score below 60 a Remediation agent auto-builds a micro-lesson and quiz from the same grounded context. Every step streams to an "Agent Thinking" timeline over SSE. At the end a Report agent produces a 30/60/90 readiness report. The whole orchestrator is thin — each agent delegates to modules I already had.

**Q2: What makes this more than a chatbot pretending to interview?**
> Three things: it's *grounded* (every question and score cites a knowledge base, and the Scorer explicitly marks answers unsupported when there's no rubric), it's *inspectable* (the agent timeline streams live with citation chips), and it's *measured* (an offline eval harness reports groundedness, refusal correctness, and p95 latency, rendered on the dashboard). Most agent demos have none of those — they hide reasoning and hallucinate confidently.

**Q3: How does the Foundry IQ / Supabase retrieval work and why two systems?**
> All retrieval goes through one `retrieve()` function. If Microsoft Foundry IQ (Azure AI Search) is configured it's the citation-first provider; if the env is missing or it errors, it transparently falls back to a Supabase pgvector RPC over the same corpus. Foundry was a requirement of the competition track and gives managed semantic search; Supabase is the failover I fully control so the demo never dies on stage. The corpus is the same, embeddings are 384-dim Cohere on both sides, and I can debug which provider served a query live via `/api/rag/status` and `/api/rag/probe`.

**Q4: How do you keep the agent honest / prevent hallucination?**
> A SafetyGuard refuses out-of-scope inputs before any LLM call — too-short JD, or a scanned resume with no readable skills — and enforces a minimum grounding threshold (≥200 chars of retrieved context). The Scorer is instructed to set `supported=false` and grade conservatively when the rubric is empty rather than inventing a grade. And the eval harness's refusal-correctness metric actually tests that out-of-scope inputs get refused. Honesty is a measured property, not a hope.

**Q5: How do you evaluate an agent like this?**
> An offline golden set of scenarios — a strong candidate, a partial-fit, and an out-of-scope input — run through the real agents. I compute groundedness (fraction of scored answers the retrieved rubric supports), refusal correctness (did we refuse the out-of-scope case), answer relevance (did the Planner find JD-grounded gaps), and p95 latency. It writes a JSON report that the dashboard renders. That harness is what lets me change a prompt or model and know if I regressed.

**Q6: Why one LLM call for the career plan instead of three sequential ones?**
> Cost and coherence. One Groq round trip is about three times cheaper than three, and the model writes a better plan when it can see the gap analysis and the roadmap it's producing in the same context window. It also means one JSON to validate instead of three failure points. The O*NET lookup has three fallbacks so even off-catalog roles anchor to a credible salary/skills row.

**Q7: The 'REUSE, don't rewrite' principle — why does it matter?**
> The Career Simulator is a headline feature, but I built it as thin orchestration over existing modules — the resume parser, the distiller's LLM/JSON helpers, the RAG retrieve function. So there's exactly one place that knows how to parse a resume or retrieve grounded context. If I swap the embedding model or the skill-gap logic, I change one file, not five. Duplicating that logic into the simulator would have created drift between the Learn page and the interview.

**Q8: How is this deployed and what was tricky?**
> FastAPI in Docker on a Hostinger VPS, Next.js 14 on Vercel. The tricky part is that Vercel serves HTTPS but the VPS API is plain HTTP, so browsers block the mixed content. I solved it with Next.js rewrites that proxy `/api/*` and `/health` same-origin through Vercel to the backend, so the browser only ever talks HTTPS to Vercel. Documented upgrade is putting a reverse proxy with Let's Encrypt in front of the API.

**Q9: What would you productionize next?**
> Move simulator session state out of memory into Postgres/Redis so it survives restarts and scales horizontally; add auth; expand the eval golden set and wire it into CI so every prompt change is regression-tested; and add HNSW on pgvector as the corpus grows. The eval harness makes all of those safe because I can measure regressions.

---

## 10. Red Flags & How to Handle

| Red flag | How to handle |
|----------|---------------|
| "In-memory sessions don't scale." | Correct, demo-grade — mirrors the conversation-store pattern; move to Redis/Postgres for multi-instance. The durable artifacts (plan snapshots, eval report) already persist. |
| "LLM-as-judge is unreliable." | That's why the Scorer must flag `supported` vs unsupported against a *retrieved rubric*, why there's a grounding threshold, and why groundedness is a tracked metric — the judge is constrained and measured, not trusted blindly. |
| "Isn't this just SkillSpring again?" | No — same foundation, but the Career Simulator multi-agent loop, Foundry IQ hybrid retrieval, and the eval harness are new. I lead with those and reference P08 for the shared plumbing. |
| "Hackathon project = toy?" | It's a live, deployed, grounded, *evaluated* system with a clean package structure and a documented failover story — the eval card and provider observability are production instincts, not demo theater. |
| "Foundry IQ dependency?" | Fully abstracted behind `retrieve()` with a Supabase failover I control; the system runs identically if Foundry is unavailable. |

---

## 11. Key Takeaways

- **What it demonstrates:** transparent multi-agent orchestration (named agents + streamed reasoning), grounded hybrid RAG with real failover, and — rare for candidate projects — a *measured reliability story* (groundedness, refusal correctness, latency) plus safety guards that refuse honestly.
- **Signals:** you build headline features *without duplicating logic*, you make agent reasoning *inspectable*, and you *evaluate* your AI rather than vibe-check it.
- **Best-fit roles:** Applied AI Engineer, Agentic AI, AI Platform, Forward-Deployed Engineer, GenAI Engineer.
- **30-second pitch:** *"PathWise runs a live multi-agent mock interview from your resume and a job posting. Named agents plan your competency gaps, ground every question in a citation-backed knowledge base — Microsoft Foundry IQ with a Supabase pgvector failover — adapt difficulty to your answers, auto-build micro-lessons for weak spots, and produce a 30/60/90 readiness report, all streamed as a visible agent timeline. It ships with an offline eval harness that measures groundedness and refusal correctness, so reliability is a number, not a claim. It's the productionized evolution of my SkillSpring platform."*
