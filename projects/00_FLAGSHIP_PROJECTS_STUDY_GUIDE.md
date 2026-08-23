# How to Study the Flagship Projects — A Systematic Guide

**Purpose:** deep notes exist for your four strongest projects — [Bilbo RAG](P07_bilbo_medical_rag.md), [AI Cargo](P15_ai_cargo_agentic_risk_platform.md), [Doc2Data](P16_doc2data_healthcare_extraction.md), and [PathWise](P17_pathwise_career_simulator.md). This guide tells you *how to internalize them* so you can walk into any AI/ML interview and defend them under pressure — not just recite them. Read this once, then use the drills weekly.

> Folder map: [`../00_STUDY_GUIDE.md`](../00_STUDY_GUIDE.md). Resume-to-proof mapping: [`../00_RESUME_MASTER_MAP.md`](../00_RESUME_MASTER_MAP.md). This guide is specifically about **mastering the flagship projects.**

---

## 1. Which one to lead with

Interviewers pick ONE project and drill for 30–45 minutes. Know which one to steer toward:

| Priority | Project | Lead with it for… | The one-sentence hook |
|---|---------|-------------------|-----------------------|
| **#1** | **Bilbo RAG (P07)** | **Any AI/GenAI Engineer role** — it's your *current job* and the top of your resume | *Production grounded RAG: hybrid retrieval + reranking + paragraph citations + LangGraph agents, all local for PHI safety.* |
| **#2** | **AI Cargo (P15)** | Agentic AI, Applied AI Engineer, AI Platform, "hard problems" roles | *Governed agent orchestration + trustworthy hybrid ML + human-in-the-loop.* |
| #3 | **Doc2Data (P16)** | ML/AI Engineer (multimodal/CV/OCR), Forward-Deployed, systems/perf | *Cost-aware multimodal extraction — spend compute only where it changes the answer.* |
| #4 | **PathWise (P17)** | Anything that values RAG + **evaluation** (the 2026 differentiator) | *Transparent multi-agent system with grounded retrieval and a measured reliability story.* |

**Rules:**
- **P07 is mandatory.** It's your current role. An interviewer who reads your resume top-down starts there. You cannot be caught weaker on your own job than on a side project.
- If the JD says "experimentation / causal / stats," steer to P14/P01/P04 instead. These four are your *AI-engineering* flagships.
- If the JD says "credit risk / fintech / lending," steer to **P01 Fibe** and read `../learning/51_credit_risk_and_scorecard_modeling.md`.

---

## 2. The 4-layer mastery model (how to "know" a project)

Don't study a project linearly. Build it in four layers, each of which is a separate interview skill. You are done with a project only when you can do all four **cold, out loud, without notes.**

### Layer 1 — The 30-second pitch (elevator)
The last section of each note. Memorize it *as a spoken script*, not text. It must include: what it does, the one clever idea, and the outcome. Practice until it's automatic — this is the first thing you say and it sets the frame.

### Layer 2 — The architecture walkthrough (2–3 min)
Draw the block diagram from memory on paper. For each project the spine is:
- **Bilbo RAG:** PDF → **8-stage ingestion** (extract→headings→anchors→paragraphs→clean→window→embed→index) → **FAISS + BM25 → RRF → cross-encoder rerank** → relevance gate → **LangGraph** stage loop (retrieve→rerank→gate→generate→validate→repair→advance) → citation verification → checklist.
- **AI Cargo:** data → risk engine (14 features → 8 rules + XGBoost → fusion+veto → tier) → LangGraph (interpret→plan→execute→observe→reflect→revise→human_review) → 8 tools → backend/dashboard.
- **Doc2Data:** load → identify → **plan (3 lanes)** → align-gate → OCR v2 (blank-detect + batched Florence-2) → validate → **reflect → targeted VLM rescue** → finalize.
- **PathWise:** resume+JD → **Planner→Retrieval→Interviewer→Scorer→Remediation→Report** over an SSE bus, grounded by **Foundry IQ ⇄ Supabase failover**, proven by an **eval harness**.

If you can't draw it, you can't defend it. Draw each one 5×.

### Layer 3 — The "clever idea" deep-dives (the memorable core)
Every project has 2–3 ideas that make you sound senior. Rehearse each as a standalone 60–90s answer:

| Project | Deep-dive #1 | Deep-dive #2 | Deep-dive #3 |
|---------|-------------|-------------|-------------|
| **Bilbo RAG** | **Hybrid retrieval + RRF + cross-encoder → +28% top-5** (and how you measured it) | **Citation grounding**: verify the anchor exists *and* supports the claim (citation-washing) | **LangGraph** stateful routing + memory + bounded repair; local 4-bit Mistral-7B *as a compliance guardrail* |
| **AI Cargo** | Rules+ML fusion with a **deterministic veto** + "unknown escalates" | The **reflect→revise hallucination bug** and the "constrain in code, not prompt" fix | **PR-AUC / recall@precision** for rare spoilage + SHAP audit |
| **Doc2Data** | **Three-lane routing** = don't pay for OCR you don't need | **Alignment gate + blank detection** (258s→41s), gate expensive models behind cheap checks | **Reflect → targeted VLM rescue** on ≤12 fields |
| **PathWise** | **Multi-agent loop** + "REUSE don't rewrite" + SSE-with-replay | **Foundry IQ ⇄ Supabase failover** (grounding that never dies) | **Eval harness**: groundedness / refusal-correctness / p95 |

These are what the interviewer remembers about you. Over-prepare them.

### Layer 4 — The defense (red flags & trade-offs)
The "Red Flags & How to Handle" table in each note. An interviewer *will* poke at the weak spots (synthetic data, in-memory state, hardcoded thresholds, LLM reliability). The senior move is to **agree fast, explain the trade-off, and state the production fix** — never get defensive. Rehearse each red-flag response as a calm one-liner.

---

## 3. The cross-project through-lines (this is what makes you sound senior)

Interviewers score "engineering judgment," and judgment shows up as *consistent principles across projects*. Learn these five through-lines so you can say "I do this in all three":

1. **Never let the model be the only vote.** Bilbo drops unverifiable citations; AI Cargo has a deterministic veto; Doc2Data validates + rescues OCR; PathWise's Scorer flags `supported` vs unsupported. → *"I bound model output with deterministic checks."*
2. **Spend expensive compute only where it changes the answer.** Bilbo's retrieve-then-rerank cascade and the Jet2 70B→7B distillation; Doc2Data lanes/blank-detect/rescue; AI Cargo LOW-tier bypasses the agent. → *"I gate cost behind cheap checks."*
3. **Make failure loud and cheap, not silent and expensive.** Bilbo's relevance-gated refusal; Doc2Data alignment gate; AI Cargo "unknown → MEDIUM"; PathWise SafetyGuard refusal. → *"The dangerous failure is confident silence."*
4. **Constrain agents in code, not prompts.** AI Cargo's hard tool filter; Bilbo's bounded repair loop and pre-planned stage order; PathWise's grounding threshold. → *"If code can guarantee a constraint, I don't leave it to a prompt."*
5. **Reliability is a number, not a hope.** Bilbo's labelled retrieval set (+28%); Fibe's KS/PSI/OOT battery; Doc2Data F1 vs gold; PathWise's eval harness. → *"I measure whether the AI is right."*

**Constraints drive architecture** is a sixth worth having ready: PHI meant local inference at Bilbo (which forced quantization), regulated cold-chain meant human-gated autonomy in AI Cargo, PHI again meant self-hosting in Doc2Data. Senior engineers explain choices by constraint, not by preference.

Memorize these five sentences. They convert three separate stories into one coherent engineer.

---

## 4. The learning-doc backfill map

When an interviewer drills a concept behind a project, you back it with a learning doc. Study the project and its concept doc together:

| If they drill… | In project | Read this learning doc |
|----------------|-----------|------------------------|
| LangGraph / agents / tool-calling | all three | `17_langchain_langgraph`, `10_agentic_ai_and_multi_agent`, `27_agentic_patterns_deep_dive` |
| RAG / vector search / pgvector | AI Cargo, PathWise | `05_rag_systems`, `06_vector_databases` |
| PR-AUC, imbalance, thresholds, SHAP | AI Cargo | `14_evaluation_metrics`, `19_classical_ml_algorithms`, `36_explainability_and_interpretability` |
| VLMs / OCR / classic CV | Doc2Data | `21_computer_vision` |
| LLM-as-judge / eval / groundedness | PathWise | `14_evaluation_metrics`, `31_ai_safety_guardrails_responsible_ai` |
| Prompt design / structured output | all three | `30_prompt_engineering_patterns`, `16_context_engineering` |
| Serving / FastAPI / SSE / WebSocket | all three | `15_fastapi_and_backend`, `26_system_design_for_ml` |
| Cloud / Supabase / deploy | all three | `23_cloud_mlops_deployment` |

---

## 5. A concrete 2-week study plan

Assumes ~60–75 min/day. Adjust to your interview date.

**Week 1 — Build depth, one project at a time.**

| Day | Focus |
|-----|-------|
| Mon | **Bilbo (P07)**: read §0 bullet map + §2, draw the architecture 3× from memory, rehearse the 30s pitch. |
| Tue | **Bilbo**: §8 LangGraph + §9 guardrails + §10 trick questions cold. Backfill: `05_rag_systems`, `49_llmops`. |
| Wed | AI Cargo: read, draw 3×, pitch. Rehearse fusion+veto and the reflect→revise bug. Backfill: `14_evaluation_metrics`. |
| Thu | Doc2Data: read, draw 3×, pitch. Rehearse the 3 lanes + the 258s→41s story. |
| Fri | PathWise: read, draw 3×, pitch. Rehearse the 6-agent loop + Foundry failover + eval harness. |
| Sat | Employed-work day: Jet2 numbers (94% / 96% / MAPE 18→7) + Fibe (0.94 AUC, WOE/KS) out loud. Backfill `51_credit_risk` if targeting fintech. |
| Sun | Rest or watch back a recording of yourself pitching all four. |

**Week 2 — Integrate, pressure-test, connect.**

| Day | Focus |
|-----|-------|
| Mon | The 5 through-lines (§3): for each, say the sentence + name where you do it in all three. |
| Tue | Rapid-fire: record yourself answering all "Interview Q&A" for one project without notes. Grade 0–3. |
| Wed | Same for project two. Re-drill anything ≤1. |
| Thu | Same for project three. |
| Fri | **Full mock** (see §6). Pick one project by "here's the JD," defend for 20 min. |
| Sat | Weak-spot day: re-drill the 3 lowest-scoring answers across all projects. |
| Sun | Final: 30s pitch × 3, one architecture drawing × 3, the 5 through-lines. Stop early. |

---

## 6. Rehearsal drills (do these, don't just read)

- **Whiteboard cold-draw:** blank paper, 3 minutes, draw the full architecture and narrate. If you stall, you found your gap.
- **The 5-year-old + the staff engineer:** explain each clever idea twice — once in plain words, once with full technical depth. Interviews swing between both.
- **Follow-the-thread:** have someone (or an AI) ask "why?" five times on one decision (e.g., "why fuse rules and ML?" → "why a veto?" → "why does unknown escalate?" …). This is exactly how deep-dives go.
- **Red-flag gauntlet:** have them read only the red-flag column and fire the objections at you rapid-fire. Answer each in one calm sentence + a fix.
- **The pivot drill:** given a random JD, decide in 10 seconds which of the three to lead with and why.

**AI-mock prompt** (paste into ChatGPT/Claude):
> *"Act as a strict senior AI engineer interviewing me. I'll describe a project. Pick it apart with follow-ups, challenge weak points, and after 20 minutes grade me 0–3 on: clarity, technical depth, engineering judgment, and handling pushback — one sentence each."*

---

## 7. Traps that hit all three (pre-load the answers)

- **"Is the data real?"** (AI Cargo synthetic, Doc2Data fixtures) → *the system design is the contribution; swapping in a real feed is a config change.*
- **"This won't scale — in-memory state."** → *demo-grade; durable artifacts already persist; here's the Redis/Postgres move.*
- **"Isn't the LLM the weak link?"** → *that's why every agentic node has a deterministic path / hard filters / a human gate / a measured eval; the LLM improves quality, it's never the safety mechanism.*
- **"Why not fully autonomous?"** → *consequential + regulated/PHI/high-cost; autonomy is earned by measured correctness; I default to human-gated or validated.*
- **"You used a framework — what did you actually build?"** → *the orchestration logic, the guardrails, the fusion/routing/eval — the framework is plumbing; point at the specific code decisions.*
- **"What broke and how did you debug it?"** → AI Cargo phantom re-execution; Doc2Data silent misalignment. Always have a concrete failure + instrumentation + fix.

---

## 8. The 48-hour-before checklist (flagship edition)

1. Re-read the JD → decide your **primary flagship** and a backup.
2. Cold-draw all three architectures once.
3. Say all three 30-second pitches out loud.
4. Rehearse your primary project's 3 deep-dives + top 3 red flags.
5. Say the 5 through-lines.
6. Prepare **one crisp "what broke" story** for the primary project.
7. Prepare 3 questions to ask them (their evals, their guardrails, their deploy cadence — signals you've shipped).
8. Sleep. You know this.

---

*Study the project and its backfill doc together. Draw before you talk. Rehearse out loud. Agree fast on trade-offs. Reliability is a number, not a hope.*
