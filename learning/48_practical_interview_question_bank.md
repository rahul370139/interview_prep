# Practical Interview Question Bank — DS / AI Engineer / FDE / Data Engineering / Analytics

**Candidate:** Rahul Sharma | **Purpose:** Real questions asked in interviews (2025–2026 market), organized by role and theme. Sections 3–9 contain **full, speakable answers** — each with a *"Say this"* script you can deliver almost verbatim, a *memory hook* to recall the structure under pressure, and the likely *follow-up*. Section 1 stays topic-level with references (those are concepts to learn from the deep docs, not scripts to rehearse).

> **How to use this file:** (1) Read a question, answer out loud *before* reading the script. (2) Compare — did you hit the same structure? (3) Memorize the **hook**, not the script; the hook regenerates the answer. (4) Tie every answer to one of your projects (P01–P14) — interviewers score *"has actually shipped this"* far above *"has read about this."* Deep theory lives in the referenced learning docs; this file is fluency practice.

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

These were asked in an actual loop. Numbered **R1–R17** (R = real) so they don't clash with the practice-bank numbering (Q1–Q44) in sections 3–9. Each entry: **short answer skeleton → where the full answer lives.**

### R1. Introduction

Structure: *current role → 2 flagship projects (one GenAI, one ML/data) → what you're looking for.* Keep to 90 seconds. Rehearse the 30-second pitches at the bottom of each project doc (`projects/P01`–`P14`, each has a "30-Second Pitch" / "STAR Summary" section).

### R2. How many GenAI projects have you done? Tools, technologies, cloud infra?

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

### R3. What is regularization? L1 vs L2?

Penalty added to the loss to constrain model complexity and reduce overfitting. **L1 (Lasso)** adds \|w\|: sparse solutions, implicit feature selection, non-differentiable at 0. **L2 (Ridge)** adds w²: shrinks weights smoothly, keeps all features, has closed form. Geometric intuition: L1's diamond constraint hits corners (zeros); L2's sphere doesn't.
→ Full math + elastic net + when-to-use: `03_regularization_and_optimization.md` (L1/L2 sections), `19_classical_ml_algorithms.md` (regression chapter).

### R4. What is a cost function? Why needed? Examples.

The single scalar the optimizer minimizes — it *defines* what "good" means for the model. Without it there's no gradient signal. Examples: MSE (regression), BCE/log-loss (classification), cross-entropy (multi-class), hinge (SVM), Huber (robust regression).
→ Full catalog with derivations: `02_loss_functions.md`.

### R5. Bias-variance tradeoff

Expected error = bias² + variance + irreducible noise. High bias = underfit (linear model on non-linear data); high variance = overfit (deep tree). Manage via model complexity, regularization, ensembles (bagging cuts variance, boosting cuts bias), more data.
→ `19_classical_ml_algorithms.md` (bias-variance section with the decomposition math).

### R6. Feature selection techniques

Three families: **Filter** (correlation, chi-squared, mutual information, variance threshold — fast, model-agnostic), **Wrapper** (RFE, forward/backward selection — expensive, model-specific), **Embedded** (L1, tree feature importance, SHAP-based pruning). Mention leakage risk: select on train folds only.
→ `19_classical_ml_algorithms.md` §6.5 Feature Selection; SHAP details in `36_explainability_and_interpretability.md`. Real usage: `P01_fibe_behaviour_scorecard.md` (1,000+ bureau variables → final scorecard).

### R7. Ensembles: boosting vs bagging; how bootstrap + aggregation work

**Bagging:** train N models in parallel on bootstrap samples (sample n rows *with replacement* — each sample leaves out ~37% OOB), then aggregate (majority vote / average). Cuts **variance**. **Boosting:** sequential; each model fits the residuals/reweighted errors of the previous. Cuts **bias**. Boosting overfits noisy data more easily; bagging is parallelizable.
→ `19_classical_ml_algorithms.md` (Random Forest + Gradient Boosting chapters, incl. why feature randomness decorrelates trees).

### R8. Random Forest working + hyperparameters

Bagging + per-split feature subsampling (√p classification, p/3 regression) → decorrelated trees → averaged. Key hyperparameters: `n_estimators`, `max_depth`, `min_samples_split/leaf`, `max_features`, `bootstrap`, `class_weight`. OOB score = free validation.
→ `19_classical_ml_algorithms.md` (RF section). Used in `P03_jet2_event_detection.md` and `P04_jet2_risk_forecasting.md`.

### R9. Detecting and handling outliers

Detect: z-score / robust z (MAD), IQR fences, Isolation Forest, LOF, domain rules. Handle: investigate first (data error vs true signal!), then cap/winsorize, transform (log), use robust models/losses (Huber), or keep if signal.
→ Statistical detection: `24_statistics_and_ab_testing.md`; algorithmic (Isolation Forest math, LOF): `P03_jet2_event_detection.md` §Isolation Forest + Q24; winsorization for metrics: `40_cuped_and_variance_reduction.md`.

### R10. Positional encoding — why, techniques, why not needed for RNN/LSTM?

Self-attention is **permutation-invariant** — without position info, "dog bites man" = "man bites dog". Techniques: sinusoidal (original), learned embeddings (BERT/GPT-2), relative (T5), **RoPE** (Llama — rotates Q/K pairs by position-dependent angle), ALiBi (linear attention bias). RNN/LSTM don't need it because they process tokens **sequentially** — order is baked into the recurrence.
→ `04_transformers_and_attention.md` §4 (all variants compared) + Q6 in its interview section.

### R11. Transformer architecture + decoding techniques

Embeddings + positional encoding → N blocks of (multi-head self-attention → add&norm → FFN → add&norm) → LM head. Decoding: greedy, beam search, temperature sampling, top-k, top-p/nucleus, contrastive; speculative decoding for latency.
→ `04_transformers_and_attention.md` (full walk-through + decoding section); KV cache in `16_context_engineering.md`.

### R12. RAG architecture

Ingest → chunk → embed → index (vector DB) → retrieve (hybrid: BM25 + dense) → rerank (cross-encoder) → assemble context → generate with citations → evaluate. Always mention **evaluation** (faithfulness / context relevance / answer relevance) — it's the differentiator.
→ `05_rag_systems.md` (end-to-end + your Medical RAG numbers); `P07_bilbo_medical_rag.md` for the shipped version.

### R13/R14. Chunking strategies; semantic chunking

Fixed-size + overlap (simple prose), recursive/structure-aware (headers, code), sentence/paragraph, **semantic chunking** (split where embedding similarity between consecutive sentences drops below threshold — chunks follow topic boundaries, not char counts), parent-document retrieval (small chunks for search, big for context).
→ `05_rag_systems.md` §2b + `16_context_engineering.md` (semantic chunking code example).

### R15. Semantic search

Search by **meaning**, not keywords: embed query + docs into a shared vector space, rank by cosine similarity/ANN. Contrast with lexical (BM25): lexical wins on exact IDs/SKUs/error codes, semantic wins on paraphrase — hence hybrid + RRF.
→ `18_nlp_and_embeddings.md` (embeddings, bi- vs cross-encoders) + `06_vector_databases.md`.

### R16. Why not store embeddings in a conventional database?

A regular B-tree index answers exact-match/range queries; similarity search needs **nearest-neighbor in high-dimensional space**, where B-trees are useless (curse of dimensionality → full scan, O(n·d) per query). Vector DBs use ANN indexes (HNSW graphs, IVF, PQ compression) for sub-linear approximate search plus metadata filtering. Nuance to mention: Postgres **can** do this via pgvector — the point is the *index type*, not the database brand.
→ `06_vector_databases.md` (HNSW/IVF/PQ internals, pgvector vs Pinecone vs FAISS tradeoffs).

### R17. Python: pairs of indices in a sorted list summing to target — O(n)

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

Every question has a **"Say this"** answer you can deliver almost verbatim, a **memory hook** to recall it under pressure, and the **follow-up** interviewers usually chase. Practice out loud.

## 3.1 LLM API & integration

### Q1. Which LLM providers have you shipped to production, and what's your fallback strategy?

> **Say this:** "I've shipped with Groq for low-latency inference, OpenAI for reasoning-heavy tasks, and Cohere for embeddings — for example AI Cargo uses Groq for its agent loop and SkillSpring uses Groq plus Cohere embeddings. For fallback, I never depend on a single provider: the LLM call sits behind an abstraction layer, so if the primary times out or hits quota, a circuit breaker trips and traffic routes to the secondary provider. If both are down, we degrade gracefully — serve a cached response or an honest 'service degraded' message rather than an error. I also pin model versions explicitly, because providers silently upgrade models and that can shift your outputs overnight."

**Memory hook:** *Two providers, one abstraction, circuit breaker, pinned versions.*
**Follow-up:** "Has your fallback ever actually fired?" — have a story or say honestly it's tested in staging with chaos-style drills.

### Q2. How do you handle rate limits and retries in production?

> **Say this:** "Three parts. First, **retry policy**: exponential backoff with jitter — jitter matters because without it all failed requests retry at the same moment and you hammer the API in waves. Second, **classify the failure**: a 429 or timeout is transient, so retry; a 400 bad request or content-policy rejection is permanent, so retrying just burns quota — fail fast and log it. Third, **protect yourself upstream**: queue requests instead of firing them all at once, and use per-tenant rate buckets so one heavy user can't starve everyone else. In FastAPI I implement this with an async semaphore plus a library like tenacity for the backoff logic."

**Memory hook:** *Backoff + jitter → classify transient vs permanent → queue + per-tenant buckets.*

### Q3. What's your strategy for prompt versioning and deployment?

> **Say this:** "I treat prompts exactly like code, because they *are* the behavior of the system. Prompts live in the repo — not in someone's notebook or a dashboard — every change goes through code review, and each version is tagged so we can roll back. The critical part: every prompt change runs against the eval suite before merge. If the new prompt drops faithfulness or accuracy more than a couple of points below baseline, the merge is blocked. Without that gate, prompt changes are silent regressions waiting to happen — a wording tweak that helps one case can break ten others, and you'd never know."

**Memory hook:** *Prompts = code: repo, review, eval gate, rollback.*

### Q4. How do you enforce structured output from an LLM?

> **Say this:** "Layered defense. First choice is the provider's native mechanism — function calling or JSON mode — because the model is constrained at generation time. Second layer is validation: parse the output against a Pydantic schema, and on failure, retry with the error message fed back to the model — 'your output failed validation because X, fix it.' Usually one retry fixes it. Third layer is a fallback to a more reliable model for the rare persistent failures. I shipped this pattern in Doc2Data, where every extracted field passes typed validators before it's accepted, and failures route to a targeted rescue step instead of silently corrupting the output."

**Memory hook:** *Constrain (JSON mode) → validate (Pydantic) → retry with error → fallback model.*
**Follow-up:** "What's the latency cost?" — structured output modes add small overhead; retries are the real cost, so log the retry rate and fix prompts when it creeps up.

### Q5. Explain temperature, top-k, and top-p. When do you change them?

> **Say this:** "All three control how the model samples the next token from its probability distribution. **Temperature** rescales the whole distribution — low temperature sharpens it toward the top token, high flattens it toward uniform, so higher means more diverse and more risky. **Top-k** truncates to the k highest-probability tokens before sampling. **Top-p**, or nucleus sampling, truncates to the smallest set of tokens whose cumulative probability exceeds p — it adapts to the distribution shape, which is why it aged better than top-k. In production I lock temperature at 0 to 0.1 for anything deterministic — extraction, classification, structured output — and raise it to 0.7-plus only for genuinely creative tasks. The classic junior mistake is leaving default temperature on an extraction task and wondering why outputs vary between runs."

**Memory hook:** *Temperature = flatten/sharpen; top-k = fixed cut; top-p = adaptive cut. Deterministic task → temp 0.*

### Q6. Why are LLMs stateless, and how do you handle memory?

> **Say this:** "The API has no memory — every call is independent, and the 'conversation' is an illusion the application creates by resending history with each request. That's actually by design: it makes serving scalable since any server can handle any request. So memory is my job, and there's a ladder of strategies. Level one: replay full history until you approach the token budget. Level two: rolling summarization — compress older turns into a summary, keep recent turns verbatim. Level three: external memory — store facts or embeddings in a database and retrieve only what's relevant to the current turn, which is basically RAG over your own conversation. In TrainPI I manage sessions server-side with the history trimmed to budget, and the summarization threshold is tuned to keep p95 latency stable."

**Memory hook:** *Stateless by design → replay → summarize → external memory (RAG over history).*

### Q7. What is the KV cache, and what is prompt caching?

> **Say this:** "During generation, each new token needs attention over all previous tokens' key and value vectors. Without caching you'd recompute those for the whole sequence at every step — quadratic waste. The **KV cache** stores them, so each new token costs one incremental computation. The tradeoff is memory: cache size grows with layers × heads × head-dimension × sequence length, which is why long contexts eat GPU memory and why architectures use Grouped-Query Attention to shrink it — Llama-3 shares 8 KV heads across 32 attention heads for a 4× reduction. **Prompt caching** is the provider-level cousin: if many requests share the same prefix — your system prompt, your tool definitions, your retrieved documents — the provider caches that prefix's KV state and you pay a fraction of the input cost. The practical rule: structure prompts static-first, dynamic-last, so the cacheable prefix is as long as possible."

**Memory hook:** *KV cache = don't recompute attention; prompt caching = don't recompute the shared prefix. Static-first prompt order.*

### Q8. When do you fine-tune vs prompt-engineer vs use RAG?

> **Say this:** "Three axes decide it. **Freshness**: if the knowledge changes faster than you'd retrain — product docs, policies, news — that's RAG; retrieval updates instantly, fine-tuning bakes in stale facts. **Format and style consistency**: if prompting can't reliably lock in an output format or brand voice, fine-tuning fixes it permanently. **Cost and latency at volume**: fine-tuning a small model on a narrow task can replace an expensive large model — that's a distillation play. Everything else, start with prompting because it's free and instant to iterate. The trap answer interviewers listen for: 'we need the model to know about our new product' is a **RAG problem, not a fine-tuning problem** — fine-tuning teaches behavior and style, not reliable factual recall."

**Memory hook:** *Fresh facts → RAG. Format/style → fine-tune. Everything else → prompt first.*

## 3.2 RAG (beyond the basics in §1)

### Q9. Why hybrid search instead of pure dense retrieval?

> **Say this:** "Dense embeddings capture meaning but blur exact identifiers — if a user searches for error code 'E4021' or SKU 'AB-1234', semantically similar-but-wrong codes can rank higher than the exact match. BM25 keyword search nails those exact matches but misses paraphrases — 'how do I get my money back' won't match a doc titled 'refund policy'. Hybrid runs both and fuses the rankings, typically with Reciprocal Rank Fusion, which combines rank positions without needing to calibrate the two incomparable score scales. In my Medical RAG I ran FAISS dense plus BM25, fused with RRF, then cross-encoder reranked the shortlist — each stage fixes the previous stage's blind spot."

**Memory hook:** *Dense = meaning, misses IDs. BM25 = exact, misses paraphrase. RRF fuses ranks, not scores.*

### Q10. How do you evaluate retrieval quality?

> **Say this:** "The key discipline is measuring retrieval **separately** from generation, because they fail differently and get fixed differently. For retrieval: build a labeled set of queries with known-relevant chunks, then compute Recall@k — did the right chunk make the top k — MRR for how high the first relevant result ranks, and nDCG for overall ranking quality. For generation: faithfulness — is the answer supported by the retrieved context — and answer relevance. The chicken-and-egg problem is building labels with no labels: I bootstrap by manually labeling 50 to 100 real queries, augment with synthetic question-answer pairs generated from the corpus, and human-spot-check the synthetic ones. Then the eval set grows continuously from production traffic."

**Memory hook:** *Two surfaces: retrieval (Recall@k, MRR, nDCG) vs generation (faithfulness, relevance). Bootstrap: 50–100 manual + synthetic.*

### Q11. What happens when your knowledge base updates?

> **Say this:** "Depends on the change rate. For occasional updates, incremental indexing: detect changed documents — hash comparison or CDC from the source system — re-chunk and re-embed only those, and upsert into the index. Deletions need explicit handling — tombstones or index rebuild — because a deleted doc that stays in the index is a stale-answer generator. For a major change, like a new embedding model, it's a full re-index, and I'd A/B the new index against the old on the eval set before cutover. Version the corpus so every answer is traceable to the document versions that produced it — that's your audit trail when someone asks 'why did the bot say this last Tuesday?'"

**Memory hook:** *Incremental upsert + tombstones for deletes + full reindex on embedding change + corpus versioning.*

### Q12. How do you handle documents that exceed the context window?

> **Say this:** "Four tools, picked per workload. **Parent-document retrieval**: index small chunks for precise search, but feed the LLM the larger parent section for context — small-to-search, big-to-read. **Two-step retrieval**: retrieve broad sections first, then search within them for the specific passage. **Recursive summarization**: summarize chunks, then summarize the summaries — lossy, so only for overview-type questions. And **long-context models** for the final synthesis step when you genuinely need the whole document — but long context isn't free: cost scales with tokens, and models still attend unevenly across very long inputs, the 'lost in the middle' problem. So retrieval-first is still the default; long context is the fallback."

**Memory hook:** *Small-to-search big-to-read → broad-then-narrow → summarize recursively → long-context as last resort.*

### Q13. A RAG answer was wrong. Walk me through debugging it.

> **Say this:** "I trace the pipeline backwards with three questions. **One: was the right chunk retrieved?** Look at the actual retrieved set. If the answer wasn't there, it's a retrieval failure — fix with better chunking, hybrid search, query rewriting, or reranking. **Two: was it retrieved but ignored?** If the right chunk was in position 7 of 8, the model may have missed it — that's a context-assembly problem: rerank harder, feed fewer better chunks, put the strongest evidence first. **Three: was it retrieved, well-placed, and still contradicted?** That's a generation failure — tighten the grounding prompt, require citations, lower temperature, or add a faithfulness check. Three surfaces, three completely different fixes — which is exactly why you log every stage. In production, every response should carry a trace of what was retrieved, in what order, with what scores."

**Memory hook:** *Not retrieved → retrieval fix. Retrieved but buried → context fix. Present but contradicted → generation fix.*

## 3.3 Agents

### Q14. What's the difference between an agent and a workflow?

> **Say this:** "A **workflow** follows branches a developer scripted in advance — the logic is deterministic even if an LLM performs individual steps. An **agent** decides its own next step at runtime based on what it has observed so far. The line is really 'who controls the control flow' — the developer or the model. The senior insight is that a workflow is often the *better* answer: if the task decomposes into known steps, deterministic routing is cheaper, faster, testable, and debuggable. My Doc2Data pipeline is deliberately a LangGraph **state machine** — the extraction lanes and quality gates are scripted, and LLM judgment is used only at bounded decision points like lane selection and field rescue. I reach for a true agent only when the space of situations is too open-ended to enumerate, like AI Cargo's risk-mitigation loop."

**Memory hook:** *Who owns control flow? Developer = workflow, model = agent. Default to workflow; agent for open-ended.*

### Q15. Where do agents break in production?

> **Say this:** "Six failure modes I design against. **Unbounded loops** — the agent never decides it's done, so I cap max iterations and budget. **Tool-call failures** — APIs fail; the agent needs retry, graceful degradation, and an escalation path to a human. **Error amplification** — a wrong observation at step 2 poisons every step after; per-step tracing and checkpointing let you find where it went wrong. **Cost blowup** — an agent task costs 5 to 30× a single LLM call, so I track cost per task, not per call. **Prompt injection through tool outputs** — a scraped webpage or retrieved doc can contain instructions the agent obeys; treat tool output as untrusted data. And **non-reproducibility** — same input, different run; mitigated with temperature 0 where possible, checkpoint replay, and logged seeds. In AI Cargo the loop is plan → act → observe → re-plan, with human approval gates on consequential actions and every step traced."

**Memory hook:** *Loops, tool failures, error amplification, cost, injection, reproducibility — cap, retry, trace, budget, sanitize, checkpoint.*

### Q16. Explain the ReAct pattern.

> **Say this:** "ReAct interleaves **Reasoning and Acting** in a loop: the model writes a *Thought* about what it needs next, takes an *Action* — a tool call — receives an *Observation* — the tool result — and repeats until it can answer. The power is grounding: instead of hallucinating facts, each reasoning step is anchored to a real tool result, and the thought trace makes failures debuggable — you can see exactly where reasoning went off the rails. Practical caveats: every cycle is an LLM call, so you cap iterations; and modern function-calling APIs implement the same idea natively, so today you rarely hand-roll ReAct prompts — but the Thought-Action-Observation loop is still the mental model behind every tool-using agent, including LangGraph ones."

**Memory hook:** *Thought → Action → Observation → repeat. Grounds reasoning in tool results; cap the loop.*

### Q17. What is MCP and when do you use it?

> **Say this:** "Model Context Protocol is an open standard — JSON-RPC based — for connecting agents to tools and data sources. Before MCP, every agent-to-tool integration was custom glue code; MCP makes it plug-and-play: a server exposes tools and resources with typed schemas, and any MCP-compatible client can discover and call them. It's essentially 'USB for agent tooling.' You use it in two directions: when your agent needs third-party capabilities — databases, browsers, SaaS APIs — you consume existing MCP servers; when you want your service to be usable by other people's agents, you expose an MCP server. The security caveat worth mentioning: MCP tools are a prompt-injection surface, so tool descriptions and outputs need the same distrust as user input."

**Memory hook:** *MCP = USB for agent tools: discover + call over JSON-RPC. Two directions: consume or expose.*

### Q18. Walk me through multi-agent orchestration patterns.

> **Say this:** "Three main patterns. **Supervisor**: one orchestrator agent routes subtasks to specialist agents and integrates results — best when subtasks need different tools or prompts. **Parallel specialists**: independent agents work on separable pieces simultaneously — best for breadth, like researching multiple aspects of a question at once. **Sequential pipeline**: agent A's output feeds agent B — really a workflow with agent-shaped stages. The honest caveat interviewers want to hear: multi-agent is frequently *worse* than one well-prompted agent — coordination overhead, error propagation between agents, and token costs around 15× a single call. Anthropic's own research showed big quality gains on genuinely parallel research tasks, but at that cost multiple. My default: start single-agent, split only when one agent demonstrably can't hold all the context or tools."

**Memory hook:** *Supervisor, parallel, pipeline — and 'one good agent usually beats a committee.'*

## 3.4 Evaluation & observability

### Q19. How do you measure whether a prompt change actually improved your system?

> **Say this:** "Never by eyeballing three examples — that's the answer that fails the interview and fails in production. I keep a labeled eval set — a few hundred real cases with expected outputs — and every prompt change runs against it before merge. Scoring depends on the task: exact match or schema validation for structured outputs, rubric scoring for constrained text, LLM-as-judge for open-ended quality, calibrated against human labels. The deploy is gated: if the score drops below baseline minus a small tolerance, the change doesn't ship. In my HIPAA chat-assistant I ran baseline-versus-candidate comparisons with an LLM judge across models — that's the pattern: same eval set, two variants, head-to-head, with the judge's reliability itself checked against human spot-labels."

**Memory hook:** *Eval set + automated scoring + deploy gate. "I tested it manually" = instant junior flag.*

### Q20. What is LLM-as-judge and when does it work?

> **Say this:** "Using a strong LLM to score another model's outputs against a rubric — it scales human-like judgment to thousands of evaluations. It works well for **subjective dimensions**: helpfulness, tone, completeness, adherence to instructions. It does *not* work for factual accuracy unless you ground the judge — give it the reference answer or source context, or it'll happily approve fluent nonsense. And judges have known biases you must engineer around: **position bias** — favoring the first answer shown, so randomize order; **verbosity bias** — favoring longer answers, so instruct against it and check length correlation; **self-preference** — favoring outputs from its own model family. The non-negotiable step: calibrate the judge against a sample of human labels before trusting it — if judge-human agreement is low, fix the rubric first."

**Memory hook:** *Judge = scalable subjective scoring. Ground it for facts. Biases: position, verbosity, self-preference. Calibrate vs humans.*

### Q21. What do you monitor in production for an LLM feature?

> **Say this:** "Three dashboards. **Quality** — the one teams forget: a continuous LLM-judge score on sampled production traffic, user feedback like thumbs-down rate, refusal rate, and hallucination flags. This is critical because LLM features degrade *silently* — the API stays up while answers get worse. **System** — the classic golden signals: p50, p95, p99 latency, throughput, error rate, and timeout rate. **Cost** — dollars per query, tokens in and out per request, cache hit rate, and cost per tenant so one customer's usage spike is visible. Alert on regressions in all three, and trend them per release so you can attribute changes to specific deploys."

**Memory hook:** *Three dashboards: Quality (judge + feedback), System (p95, errors), Cost ($/query, cache hits).*

### Q22. How do you wire evals into CI/CD?

> **Say this:** "Any PR that touches prompts, models, retrieval config, or agent logic triggers the eval suite automatically — same as unit tests. The suite runs the labeled eval set, scores it, and compares against the stored baseline. The merge is blocked if a core metric — faithfulness, accuracy, whatever's primary for the feature — drops more than two to three points below baseline. Two practical details: cache eval LLM calls so the suite is fast and cheap enough that nobody's tempted to skip it, and version the baseline alongside the code so a deliberate quality-cost tradeoff is an explicit, reviewed decision rather than a silent drift. Post-merge, canary the change on a small traffic slice with the production quality monitors watching before full rollout."

**Memory hook:** *Eval suite = unit tests for AI behavior: trigger on PR, gate on baseline delta, canary after merge.*

---

# 4. Production Themes

The four themes that dominate AI-engineering loops. Each has a full speakable answer plus the reasoning behind it, so you can survive follow-ups.

## 4.1 Scalability

### Q23. "Your RAG/LLM feature grows from 100 to 100K users — what breaks first, and what do you do?"

> **Say this:** "The first thing that breaks is **cost and LLM throughput**, not your servers — the LLM call is both the latency bottleneck and 90%+ of the bill. So I work in four layers.
>
> **Layer 1 — the LLM calls themselves.** Add caching: a semantic cache that serves previous answers for near-duplicate queries — support and FAQ traffic is highly repetitive, so hit rates of 20–40% are realistic — plus provider prompt caching for the shared system-prompt prefix. Then routing: a model cascade where a cheap small model handles easy queries and escalates hard ones to the expensive model. Cap max output tokens; unbounded generation is unbounded spend.
>
> **Layer 2 — retrieval.** At 100 users you can brute-force search; at 100K you need a properly tuned ANN index — HNSW with sensible ef and M parameters — metadata pre-filtering so you search a fraction of the index, replicas for read throughput, and quantized embeddings if memory becomes the constraint.
>
> **Layer 3 — serving.** Async everything: FastAPI async endpoints, a queue between the API and the LLM workers so spikes buffer instead of timing out, horizontal scaling behind a load balancer, and streaming responses over SSE — streaming doesn't reduce real latency but it transforms perceived latency, first token in 500 milliseconds instead of a 10-second blank screen. Timeouts and backpressure so an overloaded downstream fails fast instead of cascading.
>
> **Layer 4 — the data pipeline.** Ingestion and embedding move fully offline and incremental — never embed at query time.
>
> And before optimizing anything, I'd measure: p95 per pipeline stage tells you which layer actually needs work."

**Memory hook:** *Four layers: Calls (cache/route/cap) → Retrieval (ANN/filter/replicas) → Serving (async/queue/stream) → Pipeline (offline).*
**Follow-up they'll ask:** "Which single change gives the most?" — usually the semantic cache plus model routing: they attack cost and latency simultaneously with no quality loss on cache hits.

### Q24. "How do you cut LLM cost 10× without killing quality?"

> **Say this:** "Stack six levers — no single one gives 10×, the combination does.
> **1. Model routing:** classify query difficulty and send easy ones to a model that's 10–20× cheaper; often 70% of traffic is easy.
> **2. Prompt caching:** restructure prompts static-first so the system prompt and tool definitions hit the provider's prefix cache — cached input tokens cost around a tenth of normal.
> **3. Trim the context:** rerank retrieved chunks and send the top 3–5 instead of 20 — halves input tokens and usually *improves* quality because there's less noise.
> **4. Semantic caching:** duplicate and near-duplicate queries never reach the model at all.
> **5. Fine-tune small for the high-volume narrow task:** a fine-tuned 8B model can match a frontier model on one specific task at a fraction of the cost — that's distillation economics.
> **6. Operational hygiene:** cap agent iterations, cap output tokens, batch non-interactive work through the provider's batch API at half price.
> The guardrail on all of it: the eval suite runs before and after every lever, so quality loss is measured, not guessed. If self-hosting, quantization — INT8 or 4-bit — buys another multiple."

**Memory hook:** *Route, cache-prefix, trim-context, cache-semantic, distill-small, cap-and-batch — evals guard every step.*

## 4.2 Reliability

### Q25. "Design for the day your LLM provider is down."

> **Say this:** "Assume it *will* happen — every major provider has had multi-hour outages. Five components.
> **Abstraction layer:** no application code calls a provider SDK directly; everything goes through an internal gateway, so switching providers is config, not a code change.
> **Circuit breaker:** after N consecutive failures the breaker opens and traffic instantly routes to the fallback provider — no per-request timeout suffering.
> **Degraded mode:** if all providers are down, degrade honestly — serve cached or semantically-similar previous answers where safe, fall back to deterministic heuristics where they exist, and show an honest 'AI features temporarily degraded' banner. An honest degradation beats a hallucinated fallback.
> **Kill switch per feature:** business-critical paths must work with AI features off — the AI enhances the product, it isn't a single point of failure.
> **Retry budget:** bounded retries with backoff, because infinite retries against a struggling provider amplify their outage and exhaust your own threads.
> The part interviewers reward: *test the fallback*. An untested fallback is a second outage waiting behind the first — fire drill it in staging regularly."

**Memory hook:** *Gateway → circuit breaker → honest degraded mode → kill switch → retry budget. And TEST the fallback.*

### Q26. "What's different about an LLM feature outage vs a normal software outage?"

> **Say this:** "Normal software fails *loud* — errors, 500s, alerts fire. LLM features fail *soft* — the API returns 200, latency looks fine, and the answers have quietly gotten worse. A provider model update, a prompt regression, or a drifting input distribution can degrade quality for weeks with zero traditional alerts. So reliability for LLM features means **quality monitoring, not just uptime monitoring**: continuous LLM-judge scoring on sampled production traffic, thumbs-down rate, refusal rate, tracked per release. And it changes on-call semantics: 'the model is unreachable' is pageable — wake someone up; 'the model is wrong' usually isn't — it feeds the eval backlog and gets fixed through the prompt/eval cycle. Confusing those two either burns out your on-call or lets quality rot."

**Memory hook:** *Software fails loud, LLMs fail soft. Unreachable = page; wrong = eval backlog.*

## 4.3 Hallucinations

### Q27. "How do you prevent hallucinations?" (the #1 AI interview question)

> **Say this:** "First, the honest framing: you can't *prevent* hallucinations — they're inherent to how autoregressive models work; the model always produces plausible text whether or not it knows the answer. What you can do is **reduce the rate, detect what slips through, and contain the impact.**
>
> **Reduce:** ground the model with RAG so it answers from retrieved evidence, not parametric memory. Require citations in the output — forcing the model to point at its source measurably reduces fabrication. Set a retrieval-confidence threshold: if nothing relevant was found, refuse rather than improvise. Lower temperature for factual tasks. In my Medical RAG, the system prompt enforces an exact refusal string — 'I cannot advise — no relevant content found' — when the retrieved excerpts don't support an answer, because in a medical context a confident wrong answer is worse than no answer.
>
> **Detect:** faithfulness scoring — an NLI model or LLM judge checks whether each claim in the answer is entailed by the retrieved context. Citation verification — do the cited chunks actually exist and say what's claimed? Self-consistency for high-stakes answers: sample multiple times; divergent answers signal low confidence. And an adversarial eval set that deliberately tries to elicit hallucinations — questions with no answer in the corpus, questions with misleading premises.
>
> **Contain:** show citations so users can verify; make escalation to a human one click; human-in-the-loop approval for any high-stakes action. The goal is that when a hallucination happens, it's caught cheaply instead of causing damage."

**Memory hook:** *Reduce (ground, cite, refuse, cool) → Detect (faithfulness, verify citations, self-consistency, adversarial evals) → Contain (UX, escalation, human gates).*
**Follow-up:** "What hallucination rate is acceptable?" — depends on stakes and fallback: near-zero for medical/legal with mandatory refusal-on-uncertainty; a few percent may be fine for brainstorming tools where the user verifies anyway. The point is you *choose* the bar per use case and measure against it.

## 4.4 Validation (models & data)

### Q28. "How do you validate an ML model before and after deployment?"

> **Say this:** "**Before:** the split must match reality — time-based splits for anything temporal, because random splits leak the future into training and inflate metrics; I used strict time-based backtesting in both my airline demand forecasting and stock prediction projects. Cross-validation with all preprocessing fitted *inside* each fold — fitting a scaler or feature selector on the full dataset is the most common leakage bug in the wild. Always a dumb baseline first — majority class, last-value carry-forward — because a model that can't beat the baseline is noise with extra steps. Then a held-out test set the model sees exactly once, as the deploy gate.
> **After:** deployment is where validation starts, not ends. Drift monitoring — PSI on input feature distributions, which I implemented for the Fibe credit scorecard — plus prediction-distribution monitoring, and delayed-label performance tracking once ground truth arrives. Canary or shadow deployment before full rollout: run the new model on live traffic without acting on it, compare against the incumbent."

**Memory hook:** *Before: honest splits, leakage-safe CV, baseline, one-shot holdout. After: PSI drift, prediction monitoring, shadow/canary.*

### Q29. "How do you validate data in a pipeline?"

> **Say this:** "Quality gates between transform and load — dirty data should fail loudly before it reaches consumers, not silently corrupt dashboards. Five check families: **schema** — columns, types, nullability match the contract; **completeness** — null rates per column under threshold; **uniqueness** — no duplicate primary keys; **volume** — row count within an expected band, because a 60% drop means an upstream break even if every row is individually valid; **freshness** — latest timestamp within SLA. I built exactly this as a reusable PySpark `DataQualityValidator` at ATCS — configurable rules, a structured quality report per run, and pipeline halt plus alert on failure. The same mindset applies to experiment data: my experimentation platform project runs SRM and logging-coverage diagnostics before any metric is read, because a measurement bug produces confident wrong conclusions — the worst kind."

**Memory hook:** *Five gates: schema, completeness, uniqueness, volume, freshness — fail loud, before load.*

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

Why this order works: interviewers fail candidates who draw boxes before asking questions (step 1) or who never mention evaluation (step 2). Front-loading both signals seniority in the first five minutes.

## 5.2 Worked example — "Design a RAG system for enterprise customer-support tickets" (the modal 2026 opener)

A full 30-minute answer you can adapt to almost any RAG design prompt. Learn the *flow*, not the exact words.

**Step 1 — Requirements (ask these out loud):**
> "Before designing — a few questions. Who uses this: support agents (assist mode) or customers directly (autonomous mode)? That changes the risk tolerance completely. What's the corpus — past tickets, help-center articles, internal runbooks — and how big? How fresh must answers be — do new product releases need to reflect same-day? Latency budget — is this an inline suggestion (sub-second) or a drafted reply (a few seconds is fine)? And any compliance constraints — PII in tickets, data residency?"
>
> *Assume the interviewer says: agent-assist first, 500K historical tickets + 5K help articles, same-week freshness, 3-second budget, tickets contain PII.*

**Step 2 — Success metrics (say early):**
> "Primary: agent adoption — % of suggestions used or lightly edited — and time-to-resolution. Quality gates before launch: faithfulness ≥ target on a labeled eval set, retrieval Recall@5 measured separately. Guardrails: escalation rate must not rise, and zero PII leakage across tenants."

**Step 3 — High-level architecture:**
> "Two planes. **Offline ingestion:** tickets and articles → PII redaction → chunking → embedding → vector index + BM25 index, running incrementally as new tickets close. **Online serving:** incoming ticket → query build (title + body + product metadata) → hybrid retrieval → rerank → prompt assembly → LLM drafts a suggested reply with citations → agent reviews, edits, sends. Agent's edit distance is my free feedback signal, feeding the eval set continuously."

**Step 4 — Data & retrieval detail:**
> "Chunking: help articles split by section headers (structure-aware); tickets kept as Q+resolution pairs — the resolution is the retrievable value. PII is redacted at ingestion with an NER pass (e.g., Presidio) — before anything hits the index or a third-party API. Retrieval: hybrid — BM25 catches error codes and product SKUs, dense catches paraphrase — fused with RRF, top-50 → cross-encoder rerank → top 5. Metadata filters on product line and language before search."

**Step 5 — Model choices:**
> "Buy, not build: a mid-tier frontier model for drafting (quality matters, 3s budget is generous), a small cheap model for query classification and routing. Structured output (JSON: draft + citations + confidence) via function calling with Pydantic validation. Provider abstraction + a second provider as fallback."

**Step 6 — Serving & scale:**
> "Async FastAPI, queue between API and LLM workers, semantic cache for recurring issues — support traffic is heavily repeated, expect high hit rates during incident spikes — prompt caching on the static system prompt. Cost math out loud: ~2K input + 300 output tokens per draft; at 10K tickets/day that's manageable, and the semantic cache cuts it further."

**Step 7 — Reliability & safety:**
> "Eval set: 200 real tickets with agent-approved gold replies; every prompt/retrieval change gates on it in CI. Production: LLM-judge sampling, thumbs-down tracking, drift alerts on retrieval scores. Kill switch: agents just write manually if the feature is down — graceful by design. Injection defense: ticket text is untrusted input; system prompt is isolated; no tools that act autonomously in v1."

**Step 8 — Roadmap:**
> "V1 ships in weeks: agent-assist with citations. V2: autonomous replies for the top-10 intents with confidence gating and auto-escalation. V3: multilingual + voice. I would *not* start autonomous — trust and evals first."

## 5.3 Prompts to drill (30 min each, out loud, whiteboard)

Run each through the same 8 steps. Half of these are systems you already built — practice presenting them as *designs*:

1. Design a RAG system for enterprise customer-support tickets (worked above — re-derive it without looking).
2. Add an LLM feature to an existing SaaS product without blowing the budget (§4.1 Q24 is the core).
3. Design a document-extraction pipeline for healthcare forms (you shipped this — Doc2Data; `doc2data/documents/PIPELINE_OVERVIEW.md`).
4. Design an agent that triages supply-chain risk alerts (AI Cargo).
5. Design semantic search over 100M product listings (`06_vector_databases.md` for ANN at scale).
6. Design an experimentation platform for a marketplace (P14 + `41_experimentation_platform_systems.md`).
7. Design a fraud/risk scoring service with real-time features (P01/P06 + `26_system_design_for_ml.md`).
8. Design evaluation infrastructure for a company's LLM features (evals-as-a-product; §3.4 + `14_evaluation_metrics.md`).

More worked designs (recommender, LLM app, fraud, search): `26_system_design_for_ml.md`.

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

## 6.2 Questions to prepare — with full answers

### Q30. "Why forward deployed / customer-facing work?"

> **Say this:** "Because the last mile is where AI projects actually succeed or die, and I've seen both sides. At ATCS I did consulting-style delivery — client data, client constraints, client skepticism — and the projects that worked were the ones where I sat close to the actual users. Doc2Data taught me the same thing from the product side: the technical extraction pipeline was maybe half the work; the other half was understanding what a claims processor actually does with the output. I want the role where the feedback loop with reality is shortest — you find out in days, not quarters, whether what you built solves the problem."

**Why this works:** it's specific, it shows you understand the role isn't pure engineering, and it names real experience.

### Q31. "A customer's team distrusts the model's outputs. What do you do?"

> **Say this:** "Trust is earned with *their* data, not with benchmarks. First, I'd sit with the skeptics — usually the people whose workflow the model touches — and collect their concrete failure examples. Skeptics are gold: they hand you your hardest test cases. Second, build an eval set from those cases plus a representative sample of their real workload, and show performance transparently — including the failures. Third, deploy in assist mode with human review, not autonomous mode: let the team accept, edit, or reject outputs, so they stay in control while accuracy proves itself in their own logs. Fourth, review the numbers with them weekly. Trust follows a visible track record; you can't argue people into it."

**Memory hook:** *Collect their failures → eval on their data → human-in-the-loop mode → weekly transparent review.*

### Q32. "You're on-site and the demo breaks in front of the client. What do you do?"

> **Say this:** "Stay calm — the client is watching how I handle failure more than the failure itself. First move: switch to the fallback I prepared — a recorded run, cached outputs, or a scripted path — because I never walk into a client demo without one. Acknowledge it honestly and briefly: 'that component is misbehaving; here's the same flow on this morning's run' — no bluffing, engineers in the room always know. Timebox any live debugging to a couple of minutes; a demo isn't a debugging session. Afterwards, root-cause it and send a short follow-up within 24 hours: what broke, why, what changed. Done right, that follow-up builds *more* trust than a flawless demo — it shows them what working with me during a real incident looks like."

**Memory hook:** *Fallback ready → honest one-liner → timebox → 24h root-cause note.*

### Q33. "The customer asks for X, but you realize they actually need Y."

> **Say this:** "Never just build X knowing it won't help, and never refuse X outright — both destroy the relationship. I'd first make sure I actually understand why they asked for X: what problem do they think it solves? Sometimes I'm the one missing context. If the analysis genuinely says Y, I bring evidence, not opinion: 'here's what the data shows about where the pain actually is.' Then I offer a path that respects their ask — often a thin slice of X that's cheap to deliver and builds trust, alongside a prototype or analysis of Y that lets the results argue for themselves. And the scope decision goes to the economic sponsor framed as options with costs and outcomes, not as me overriding their request. People rarely resist their own data."

**Memory hook:** *Understand why X → evidence for Y → thin slice of X + prototype of Y → sponsor decides between options.*

### Q34. "How do you know your deployed AI system is actually working?" (THE differentiator)

> **Say this:** "Three levels, and I set all three up *before* deploy. **Baseline:** measure the current process first — minutes per ticket, error rate, throughput — because 'improved by 40%' means nothing without a denominator. **System quality:** an eval set built from the customer's own data with their team's definition of correct, run continuously — not just at launch — plus production monitoring for drift and failure modes. **Business outcome:** the metric the sponsor actually cares about — resolution time, cost per document, revenue per rep — tracked against the baseline, reviewed with the customer on a regular cadence. If I can't draw the line from model output to that business number, I don't really know it's working — I know it's running. Those are different things."

**Memory hook:** *Baseline before → eval set + monitoring during → business metric always. "Running ≠ working."*

### Q35. "Tell me about translating a vague business ask into a shipped system."

> **Say this (pick one, told in this shape):** "At Fibe, the ask was essentially 'our risk scoring isn't working well' — no spec, a thousand-plus bureau variables, and a manual process nobody fully trusted. I turned that into: a defined target (delinquency within a window), a measurable baseline (the incumbent scorecard's approval-to-default tradeoff), a constraint list from compliance (interpretability, regulator-explainable features), and then a shipped system — automated pipeline, interpretable model, monitoring with PSI drift alerts. The lesson I carry: the translation step *is* the job — writing down the target, the baseline, and the constraints is what converts a complaint into an engineering problem."

**Alternates:** Doc2Data ("make document processing less painful" → lane-based extraction platform with quality gates) or the GPU benchmark ("which GPU should we use?" → reproducible benchmark harness + recommendation engine, validated at 80% winner-match).

### Technical depth rounds

Same material as §3–§5 (RAG, evals, agents, reliability), plus integration realities FDEs hit: customer auth (SSO/SAML), data access patterns, on-prem/VPC deployment constraints, data residency. → `23_cloud_mlops_deployment.md`, `28_terraform_iac_platform_engineering.md`.

---

# 7. Data Science / Classical ML Question Bank

Fundamentals get re-asked forever. The pure-concept questions are fully answered in the referenced docs (each has its own "Interview Questions with Strong Answers" section — drill those). The **scenario questions** that need a rehearsed narrative are answered in full below the table.

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
| 22 | "Your model's production performance dropped — walk me through debugging" | Answered in full below (Q36) |

For GenAI-flavored DS roles add: `38_genai_for_data_scientists.md`.

## 7.1 Scenario answers (rehearse these)

### Q36. "Your model's production performance dropped. Walk me through debugging it."

> **Say this:** "I debug in order of likelihood, and the model itself is the *last* suspect.
> **First — is the measurement right?** Did the metric drop, or did the metric's *computation* break? Check the eval pipeline, label sources, and any recent changes to how the metric is calculated. A dashboard bug looks identical to a model failure.
> **Second — did the input data change?** This is the usual culprit. PSI or KL divergence on each feature's distribution against the training baseline — at Fibe I had PSI monitoring on every scorecard feature with alert thresholds. A schema change, a new upstream default value, a fraction of nulls where there weren't any — any of these silently shifts predictions.
> **Third — did the data pipeline break?** A join dropping rows, a stale feature table, a timezone shift moving events across day boundaries. Compare feature values for identical entities week-over-week.
> **Fourth — did the world change?** Label shift or concept drift: new user segments, seasonality, a competitor action, a policy change. Segment the performance drop — is it everywhere, or concentrated in one geography, one cohort, one product line? Concentrated means something specific happened there.
> **Only then, the model:** if inputs are clean and the world changed, retrain on recent data, and consider whether the drift is one-off or ongoing — ongoing drift means scheduled retraining, not a one-time fix."

**Memory hook:** *Metric → inputs (PSI) → pipeline → world (segment it) → model last.*

### Q37. "How would you handle severe class imbalance?" (e.g., 1% fraud/default rate)

> **Say this:** "First fix the *metric*, then the data, then the model. **Metric:** accuracy is meaningless at 1% positive — 99% accuracy is the do-nothing classifier. I use PR-AUC as the primary (it focuses on the rare class), plus precision and recall at the operating threshold the business will actually use. **Data:** resampling on the training set only — SMOTE or undersampling — never on validation or test, or your metrics lie. **Model:** class weights are usually my first choice over resampling — cleaner, no synthetic artifacts; XGBoost's `scale_pos_weight` does this directly. **Threshold:** the default 0.5 is arbitrary; I pick the threshold from the precision-recall curve based on the cost asymmetry — at Axio, a missed default costs far more than a false alarm, so we tuned toward recall and let a human review the flagged cases. And I always sanity-check the model against the base rate: does the predicted positive rate roughly match reality?"

**Memory hook:** *Metric (PR-AUC) → weights over resampling → threshold from cost asymmetry → never resample the test set.*

### Q38. "You have 1,000+ candidate features. How do you get to a usable model?"

> **Say this:** "This was literally my Fibe scorecard problem — 1,000+ bureau variables. Staged filtering, cheap to expensive. **Stage 1, filters:** drop near-zero-variance features, drop features with excessive missingness, and among highly-correlated pairs keep one — correlation clusters mean redundancy. Univariate signal screens like IV/mutual information cut the field fast. **Stage 2, embedded:** train a regularized or tree model on the survivors and use importance — L1 zeros out weak features; gradient-boosting importance plus SHAP catches non-linear signal. **Stage 3, wrapper on the shortlist:** recursive feature elimination with cross-validation on the final few dozen, since it's too expensive earlier. Two disciplines throughout: selection happens *inside* CV folds — selecting on the full dataset leaks — and for credit specifically, every surviving feature had to be explainable to a regulator, so interpretability was a hard filter, not a nice-to-have."

**Memory hook:** *Filter (variance/missing/correlation) → embedded (L1, tree+SHAP) → wrapper (RFE last) — inside folds, always.*

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
| 15 | "Pipeline produced wrong numbers yesterday — debug it" | Answered in full below (Q40) |

SQL practice: write, don't read — top-N per group, dedup with ROW_NUMBER, funnel with LEFT JOINs, retention with self-join/window, running totals (`22` §window functions).

## 8.1 Scenario answers (rehearse these)

### Q39. "Design a daily ETL pipeline ingesting CSVs, an API, and a database."

> **Say this:** "I'd structure it as extract → transform → quality gate → load, config-driven so new sources are configuration, not code — that's the architecture I built at ATCS with PySpark.
> **Extract:** per-source readers — CSVs with explicit schemas and PERMISSIVE mode so corrupt rows are captured to a reject table instead of killing the run; the API with pagination, retry with backoff, and rate-limit respect; the database via JDBC with predicate pushdown so filtering happens on the source, not after a full table pull.
> **Transform:** schema validation against the contract, null strategies per column, type coercion with multi-format date parsing, dedup with ROW_NUMBER over the business key ordered by updated-at — keep latest.
> **Quality gate before load:** completeness, uniqueness, volume-within-band, freshness. Fail loud, alert, halt — dirty data never reaches consumers.
> **Load:** incremental by default using a high-water mark on updated-at; full-refresh only for small dimensions. Everything **idempotent** — reruns upsert rather than duplicate — because reruns are a fact of life.
> **Orchestration:** Airflow DAG, per-source tasks fanning into transform, retries with exponential backoff, SLA alerts, and backfill support — which idempotency makes safe."

**Memory hook:** *Extract (schema+rejects) → Transform (validate, dedup) → Gate (fail loud) → Load (incremental, idempotent) → Airflow.*

### Q40. "The pipeline produced wrong numbers yesterday. Debug it."

> **Say this:** "Follow the lineage from consumption back to source, checking the cheapest hypotheses first.
> **One — scope it:** which metrics are wrong, since when, how wrong? Everything 10× off means a units or join bug; one metric slightly off means logic; everything slightly off means partial data.
> **Two — row counts by stage:** compare yesterday's counts at each pipeline stage against the day before. A count explosion after a join means a fan-out — a supposedly-unique join key that now has duplicates on one side; that's the single most common cause. A count drop means a filter or a failed extract partition.
> **Three — time handling:** timezone shifts and late-arriving data move events across day boundaries; check whether the 'missing' data landed in the adjacent day's partition.
> **Four — schema drift:** did a source add a column, change a type, or change a default? Silent nulls from a renamed column flow straight into aggregates.
> **Five — the code:** only after data checks, look at yesterday's deploys — a changed filter or join condition.
> Then the fix has two halves: correct the data — rerun the idempotent pipeline for the affected window — and add the quality check that would have caught this class of bug automatically. Every incident should upgrade the gate suite."

**Memory hook:** *Scope → counts per stage (join fan-out!) → timezones/late data → schema drift → code last. Fix + add the missing check.*

### Q41. "Your Spark job is slow / keeps failing. What do you look at?"

> **Say this:** "The Spark UI first — it tells you which stage and why. The usual suspects in order: **Shuffle volume** — wide transformations like joins and groupBys move data across the cluster; I minimize by filtering and selecting columns *early*, and by broadcasting small dimension tables so the join happens map-side with no shuffle at all. **Skew** — one giant key (a null key, a default value) makes one task run forever while 199 finish; visible in the UI as a straggler task; fixes are salting the hot key or AQE's skew-join handling. **Partitioning** — too few partitions underuses the cluster, too many drowns it in scheduling overhead; and coalesce before writing so you don't produce ten thousand tiny files. **Memory** — spills to disk in the UI mean executors are undersized or a single partition is too big. At ATCS the combination of broadcast joins, early column pruning, and JDBC predicate pushdown took our worst pipeline from hours to minutes."

**Memory hook:** *Spark UI → shuffle (broadcast small tables) → skew (salt the hot key) → partitions (coalesce writes) → memory (spills).*

---

# 9. Analytics / Product DS Question Bank

| # | Question | Reference |
|---|---|---|
| 1 | "DAU dropped 10% — investigate" | Answered in full below (Q42) |
| 2 | Define a north-star metric for product X + guardrails | `37` §2–3 |
| 3 | Design metrics for a new feature launch | primary + secondary + guardrail; `37`, `14` §experiment metrics |
| 4 | Leading vs lagging metrics | `37` §2.3 |
| 5 | The experiment is flat/negative but PM wants to ship | Answered in full below (Q43) |
| 6 | SRM detected — what now? | Answered in full below (Q44) |
| 7 | Novelty & primacy effects | `24` (pitfalls), `43` |
| 8 | Simpson's paradox — real example | `24` Q&A; P14 scenario has one built in |
| 9 | Metrics for a marketplace / ads / SaaS business | `37` §9, `45_marketplace_science.md` §8, `46_ads_ranking_and_auctions.md` |
| 10 | Estimate/sizing questions ("how many X…") | structure > answer: decompose, state assumptions, sanity-check bounds |
| 11 | Communicate a technical result to execs | lead with the decision, one number, one caveat; `37` case framework |

## 9.1 Scenario answers (rehearse these)

### Q42. "DAU dropped 10% this week. Investigate."

> **Say this:** "Before any analysis, two framing questions: dropped relative to *what* — last week, same week last year, forecast? And is 10% outside normal weekly variance for this metric? Then four phases.
> **Phase 1 — is it real?** Instrumentation first, always. A logging change, an SDK release, a tracking-consent update, or a bot-filter change can produce exactly this signature. Check whether the definition or the collection of DAU changed. Measurement bugs cause a huge share of 'metric drops.'
> **Phase 2 — segment it.** Cut by platform (iOS vs Android vs web — an app release gone wrong shows instantly), by geography (one country = local event or holiday), by user cohort (new vs returning — a drop in *new* users points to acquisition channels; in *returning* points to product or notifications), and by entry point.
> **Phase 3 — timeline.** Exact start date and shape: cliff (something shipped or broke — check the deploy log and marketing calendar) versus slow slide (churn, seasonality, competition). Overlay known events.
> **Phase 4 — quantify and conclude.** Decompose the 10% into contributions per segment — 'two-thirds is Android in India following the vX release, the rest is a paused acquisition campaign' — and end with a recommendation and a monitoring change so the next occurrence is caught automatically."

**Memory hook:** *Real? (instrumentation) → Segment (platform/geo/cohort) → Timeline (cliff vs slide) → Quantify each piece.*

### Q43. "The experiment is flat, but the PM wants to ship anyway. What do you do?"

> **Say this:** "First, make sure 'flat' actually means 'no effect' rather than 'no power.' Check the achieved MDE: if the experiment could only detect a 5% lift and the plausible true effect is 1%, the test was never going to see it — that's an underpowered design, not a null result, and the honest options are run longer, use variance reduction like CUPED, or accept the uncertainty explicitly. Second, look at guardrails and segments: a flat average can hide a positive effect in one segment cancelled by a negative one elsewhere — heterogeneous effects worth understanding before shipping, with multiplicity caveats. Third, if it's genuinely a well-powered null: shipping can still be rational — strategic value, code simplification, alignment with a platform direction — but that's a *product* decision, and my job is to make sure it's made with accurate information: 'this is neutral on our metrics with a confidence interval of ±X%,' not 'the data supports shipping.' I'd document the decision rationale so the null result isn't retold as a win later."

**Memory hook:** *Power check (MDE!) → segments/guardrails → a true null can still ship, but label it honestly.*

### Q44. "You detect sample ratio mismatch in an experiment. What now?"

> **Say this:** "Stop reading the results — that's the first move. SRM means the randomization or logging is broken, so every downstream number is untrustworthy, no matter how exciting the lift looks. Then diagnose: chi-square confirms the imbalance isn't chance — with millions of users, even 50.2/49.8 can be wildly significant. Common causes, in the order I'd check: assignment bugs (a faulty hash, a targeting condition applied to one arm), logging asymmetry (the treatment changes page behavior so exposure events fire differently — e.g., faster page loads log more reliably), bot filtering interacting with one arm, and caching or redirect differences dropping users from one variant. Segment the SRM — if it only appears on one platform or browser, that localizes the bug. The critical discipline: fix the root cause and *rerun* the experiment. You cannot 'correct' SRM-contaminated data statistically, because the missing users are almost never missing at random. I built exactly this check into my experimentation platform project — SRM runs automatically before any metric is displayed."

**Memory hook:** *Stop → chi-square confirm → hunt: assignment, logging asymmetry, bots, caching → segment to localize → fix and RERUN, never patch.*

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
