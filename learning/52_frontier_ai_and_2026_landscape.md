# Frontier AI & the 2026 Landscape — What Changed, What's Asked, What to Say

**Purpose:** your other 51 docs teach the fundamentals. This one is the *delta* — what the field added recently, what interviewers in 2026 actually probe, and the vocabulary that signals you've been building lately rather than reading a 2023 tutorial. Read this last, but read it before every interview.

---

## Table of Contents

1. [What AI interviews actually test in 2026](#1-what-ai-interviews-actually-test-in-2026)
2. [The six-layer modern agent stack](#2-the-six-layer-modern-agent-stack)
3. [Context engineering (the skill that replaced prompt engineering)](#3-context-engineering)
4. [MCP — the tool-connectivity standard](#4-mcp--the-tool-connectivity-standard)
5. [Agent memory as a first-class primitive](#5-agent-memory-as-a-first-class-primitive)
6. [Retrieval in 2026: beyond embed-and-search](#6-retrieval-in-2026)
7. [GraphRAG and when graphs beat vectors](#7-graphrag)
8. [Agentic RAG](#8-agentic-rag)
9. [Reasoning models & test-time compute](#9-reasoning-models--test-time-compute)
10. [DSPy — compiling prompts instead of writing them](#10-dspy)
11. [RL fine-tuning: DPO, GRPO, RLVR](#11-rl-fine-tuning-dpo-grpo-rlvr)
12. [Evaluation as infrastructure](#12-evaluation-as-infrastructure)
13. [Agent economics](#13-agent-economics)
14. [Real production architectures worth citing](#14-real-production-architectures-worth-citing)
15. [MoE and what it means for serving](#15-moe-and-what-it-means-for-serving)
16. [The 2026 buzzword decoder](#16-the-2026-buzzword-decoder)
17. [Interview questions with strong answers](#17-interview-questions-with-strong-answers)
18. [Key takeaways](#18-key-takeaways)

---

## 1. What AI interviews actually test in 2026

The shift, stated plainly: **interviews moved from "do you know how a transformer works" to "can you build an AI system that is correct, cheap, and debuggable."** Model internals still appear, but they're no longer the main event.

Observed weighting across AI/GenAI engineer loops:

| Area | Share | What they're really checking |
|---|---|---|
| RAG, evals, agents | **~40%** | Can you build *and measure* a retrieval or agent system |
| Production systems | **~30%** | Cost, latency, failure modes, observability, rollout |
| LLM internals | **~20%** | Tokenization, context windows, sampling, attention, KV cache |
| Behavioral | **~10%** | Ambiguity, ownership, judgment under uncertainty |

**The single biggest filter is evaluation.** The pattern reported repeatedly by hiring teams: candidates who can *build* a RAG pipeline but can't say how they'd *measure* whether it works get rejected. Candidates who lead with evals pass.

> **The sentence that passes interviews:** *"Here's how I'd know if it's working, and here's how I'd know if a change made it worse."*

**Consequences for you:**
- Your `14_evaluation_metrics.md` + `49_llmops_and_observability.md` are now *more* important than your transformer internals.
- Every project answer should end with a measurement statement. Your PathWise eval harness (`P17`) and Bilbo labelled retrieval set (`P07`) are your assets here — use them.
- "How would you build a RAG system for X?" is the most-asked question in the field. Have a 3-minute canonical answer ready (see §17 Q1).

---

## 2. The six-layer modern agent stack

Three things redrew the map since 2024: **MCP standardized tool connectivity**, **reasoning models changed what a single call can do**, and **memory became architecture rather than an afterthought**.

```
┌──────────────────────────────────────────────────────────────┐
│ 6. EVAL & OBSERVABILITY   traces · golden sets · LLM-judge   │
│    LangSmith · Langfuse · Arize Phoenix · RAGAS · DeepEval    │
├──────────────────────────────────────────────────────────────┤
│ 5. FRAMEWORK / ORCHESTRATION   graphs · state · HITL          │
│    LangGraph · LlamaIndex Workflows · DSPy · provider SDKs    │
├──────────────────────────────────────────────────────────────┤
│ 4. CONTEXT ENGINEERING   compaction · caching · JIT retrieval │
│    what fills the window at each step                         │
├──────────────────────────────────────────────────────────────┤
│ 3. MEMORY & KNOWLEDGE   vector · graph · episodic · semantic  │
│    pgvector/Qdrant · Neo4j/GraphRAG · memory servers          │
├──────────────────────────────────────────────────────────────┤
│ 2. TOOLS & PROTOCOLS   function calling · MCP servers          │
│    MCP standardized this whole layer (it didn't exist in 2024)│
├──────────────────────────────────────────────────────────────┤
│ 1. MODEL   frontier API · open weights · reasoning · MoE      │
│    routed by task difficulty; self-hosted via vLLM/Ollama     │
└──────────────────────────────────────────────────────────────┘
```

Being able to draw and defend this — plus the five primitives (**RAG, model routing, guardrails, evaluation, agentic loops**) — covers most AI system-design rounds.

---

## 3. Context engineering

**The reframing:** prompt engineering is crafting the perfect question. **Context engineering is curating what information reaches the model at each step.** For agents that run many turns, the second matters far more.

### The failure modes you must be able to name

| Failure | What happens | Mitigation |
|---|---|---|
| **Context rot** | Quality degrades as the window fills with stale/irrelevant history | Compaction, summarization, eviction policies |
| **Lost in the middle** | Facts buried mid-context get ignored vs. start/end | Reorder by relevance; put critical facts at boundaries |
| **Context poisoning** | One bad retrieval or tool error propagates through all later turns | Validate tool outputs; quarantine failures |
| **Context distraction** | Too many tools/options degrade tool choice | Tool subsetting; retrieve tools like documents |
| **Token bloat → cost** | Agent loops re-send everything | Cache stable prefixes; JIT retrieval |

### The techniques

- **Compaction** — periodically replace raw history with a structured summary, keeping decisions and open threads while dropping transcript noise.
- **Prefix caching** — put stable content (system prompt, tool schemas, retrieved corpus) at the *front*, unchanged, so the provider or your server can cache the KV for it. This is one of the highest-leverage cost levers in agent systems. (See `50_llm_serving_and_inference_optimization.md` for vLLM automatic prefix caching.)
- **Just-in-time (JIT) retrieval** — don't pre-load everything; let the step fetch what it needs. Trades latency for relevance and cost.
- **Structured note-taking** — the agent writes findings to an external scratchpad/file rather than holding them in context.
- **Tool retrieval** — with 50+ tools, retrieve the relevant 5 instead of listing all in the prompt.
- **Sub-agent context isolation** — give a sub-agent a clean, narrow window; return only its conclusion to the parent. This is how multi-agent systems avoid context explosion.

> **Say this:** "I treat the context window as a scarce, managed resource, not a dumping ground. Stable content goes first so it caches, history gets compacted on a policy, retrieval happens just-in-time, and sub-agents get isolated windows so the parent only sees conclusions."

Cross-reference: `16_context_engineering.md`.

---

## 4. MCP — the tool-connectivity standard

**The problem it solves:** integrating $N$ models with $M$ tools used to require $N \times M$ bespoke integrations. MCP makes it $N + M$ — every model speaks one protocol, every tool exposes one interface.

### The primitives

| Primitive | What it is | Analogy |
|---|---|---|
| **Tools** | Callable functions the model can invoke (side effects allowed) | POST endpoints |
| **Resources** | Readable context the host can fetch | GET endpoints / files |
| **Prompts** | Reusable parameterized templates the server offers | Stored procedures |

Architecture: **host** (the app) ↔ **client** (protocol connector) ↔ **server** (exposes tools/resources). Transports: stdio for local, HTTP/SSE for remote.

### What interviewers probe

- **Why MCP over plain function calling?** Function calling is the *model capability*; MCP is the *integration standard* around it. MCP gives you discovery, reuse across hosts, and a security boundary — you write a tool server once and any MCP-capable client can use it.
- **Security.** This is the real question. Untrusted MCP servers are an injection surface: a malicious tool description can carry instructions; tool results enter your context. Mitigations: allowlist servers, scope credentials per server, sandbox execution, treat tool output as *untrusted data* not instructions, human approval for consequential actions, audit every call.
- **The confused-deputy problem** — your agent has credentials the user shouldn't be able to exercise indirectly. Scope tokens narrowly and check authorization at the tool boundary, not the prompt.

Cross-reference: `11_a2a_and_mcp_protocols.md`.

---

## 5. Agent memory as a first-class primitive

In 2024 "memory" meant "stuff a vector DB." In 2026 it's a designed subsystem with distinct types:

| Type | Horizon | Contents | Typical store |
|---|---|---|---|
| **Working** | current turn/step | active task state | in-context / graph state |
| **Episodic** | session | what happened, decisions, tool results | checkpointer, session store |
| **Semantic** | long-term | facts, preferences, entity knowledge | vector DB + knowledge graph |
| **Procedural** | long-term | learned how-to, successful trajectories | prompt/playbook store, fine-tune data |

**The hard problems** (and what to say): *when to write* (every turn is noisy and expensive — write on salience or on task completion), *conflict resolution* (new fact contradicts stored fact — recency vs. confidence vs. source authority), *forgetting* (TTL, decay, relevance-based eviction — unbounded memory degrades retrieval precision), and *privacy* (memory is a PII store; needs deletion paths and consent).

> **Say this:** "Memory is a write-path problem, not a read-path problem. Everyone can retrieve; the design decisions are what you commit, how you resolve contradictions, and how you forget."

---

## 6. Retrieval in 2026

Naive RAG (fixed chunks → cosine → stuff prompt) is now the *baseline you're expected to improve on*. The current toolkit:

| Technique | What it does | When it wins |
|---|---|---|
| **Hybrid search (BM25 + dense)** | Lexical catches exact terms/IDs/rare words; dense catches paraphrase | Almost always. This is table stakes — you did it at Bilbo |
| **Reciprocal Rank Fusion (RRF)** | Rank-based fusion, no score normalization needed: $\text{RRF}(d) = \sum_i \frac{1}{k + r_i(d)}$ | Combining retrievers with incomparable scores |
| **Cross-encoder reranking** | Joint query-doc scoring on a small candidate set | Big precision gain for small latency; retrieve-then-rerank cascade |
| **Listwise / LLM rerankers** | Rank a whole candidate list in one pass, modelling inter-document context | Highest quality reranking; higher cost |
| **Late interaction (ColBERT-style)** | Per-token embeddings, MaxSim at query time | Better than single-vector, cheaper than cross-encoder; storage-heavy |
| **Late chunking** | Embed the *long* document first, then pool per-chunk from contextualized token embeddings | Chunks keep document context — fixes "orphaned pronoun" chunks |
| **Contextual retrieval** | Prepend an LLM-generated context blurb to each chunk before embedding | Big recall gains on chunk-level ambiguity |
| **Query rewriting / HyDE / multi-query** | Rewrite or expand the query; generate a hypothetical answer and search with it | Vague, short, or under-specified queries |
| **Parent-document / small-to-big** | Retrieve on small precise units, return the larger parent for context | Precision of small chunks + context of big ones. *You do this at Bilbo: retrieve windows, cite paragraphs* |
| **Vision / page-as-image retrieval** | Embed rendered pages with a vision model; skip lossy text extraction | Complex layouts, tables, forms — relevant to your Doc2Data work |

**The chunking question** ("how do you choose chunk size?") — the honest senior answer: *it's determined by your retrieval unit versus your citation unit versus your generation context*. Small chunks = precise retrieval, poor context; large = the reverse. Decouple them: index small, return large (parent-document), or use late chunking so the small unit retains document context.

---

## 7. GraphRAG

**The core claim:** vector similarity answers "what's semantically near this?" A graph answers "what is *connected* to this, and how?" When precision and multi-hop reasoning matter, structured relationships beat embeddings.

| | Vector RAG | GraphRAG |
|---|---|---|
| Query it's good at | "What does the doc say about X?" | "How does X affect Y through Z?" |
| Multi-hop | Weak — each hop is a fresh fuzzy search | Strong — traverse explicit edges |
| Global questions | Poor ("summarize themes across the corpus") | Good via community summaries |
| Determinism | Similarity is fuzzy | Traversal is deterministic |
| Cost to build | Low | High — entity/relation extraction over the corpus |
| Freshness | Easy re-embed | Graph updates are harder |

**How it's built:** LLM extracts entities and relations → build knowledge graph → cluster into communities → pre-generate community summaries → at query time do local search (entity neighborhood) or global search (map-reduce over community summaries).

**The honest take for interviews:** GraphRAG is expensive to build and maintain, and it wins on a specific query class — multi-hop, relational, and corpus-global questions. Most production systems are hybrid: vectors for lookup, graph for relationships. *Don't* claim GraphRAG is a universal upgrade; that reads as hype.

---

## 8. Agentic RAG

The shift: retrieval stops being a fixed pipeline stage and becomes **a decision the agent makes**.

```
Classic RAG:   query → embed → retrieve k → generate            (one shot, fixed)
Agentic RAG:   query → agent decides:
                        need retrieval at all?
                        which source? (vector / graph / SQL / API / code search)
                        rewrite the query?
                        is what I got sufficient?  ── no ──► retrieve again, differently
                        conflicting sources? ──► reconcile or escalate
                      → generate → self-check groundedness → maybe re-retrieve
```

Patterns to name: **self-RAG** (model critiques its own retrieval and output), **corrective RAG / CRAG** (grade retrieved docs; fall back to web/another source if weak), **adaptive RAG** (route by query complexity — skip retrieval for chit-chat, multi-hop for hard questions), **router RAG** (classify to the right index).

**Cost warning to volunteer:** agentic retrieval multiplies LLM calls. Bound the loop (max iterations), cache aggressively, and use a small model for the routing/grading decisions and a large one only for synthesis.

---

## 9. Reasoning models & test-time compute

**The idea:** instead of scaling training, spend more compute *at inference* — the model generates extended internal reasoning before answering. This created a new axis: you can now buy accuracy with latency and tokens.

What this changed practically:
- **Some multi-step chains collapsed into a single call.** Workflows that used to need explicit planner→worker decomposition can sometimes be one reasoning-model call. Interviewers like candidates who know when *not* to build an agent.
- **A new cost/latency profile.** Reasoning tokens are billed and slow. You now *route by difficulty*: cheap fast model for easy traffic, reasoning model for the hard tail.
- **Prompting differs.** Reasoning models generally want less hand-holding — don't force chain-of-thought scaffolding they do internally; give them the goal and constraints.
- **Evaluation differs.** You care about the final answer, but for agents you also evaluate the **trajectory** (did it take a sane path?).

Related vocabulary: **best-of-n / self-consistency** (sample $n$, pick by majority or a verifier), **verifier models** (score candidate answers), **budget forcing** (cap or extend thinking), **process vs outcome supervision** (reward the reasoning steps vs. only the final answer).

> **Say this:** "Test-time compute gave me a third dial. I used to trade cost against quality by picking a model; now I also trade latency against quality within a model. So I route: the cheap path handles the bulk, and the reasoning path handles the hard tail where being wrong is expensive."

---

## 10. DSPy

**The pitch:** stop hand-writing prompts; define the *program* and let an optimizer compile the prompts (and optionally weights) against a metric.

The building blocks:
- **Signature** — a typed declaration of the task: `question -> answer`, or `context, question -> answer: str, confidence: float`.
- **Module** — a strategy: `Predict`, `ChainOfThought`, `ReAct`, `ProgramOfThought`.
- **Metric** — how you score an output (this is the whole game; a bad metric compiles a bad program).
- **Optimizer/compiler** — searches instructions and few-shot demonstrations:
  - **BootstrapFewShot** — mines successful trajectories from your data to use as demos.
  - **MIPROv2** — jointly searches instruction candidates *and* demonstration subsets.
  - **GEPA** — reflective/evolutionary prompt search building Pareto fronts of candidates.
  - **BetterTogether** — chains optimizers (e.g. prompt-optimize → fine-tune → prompt-optimize).
  - **Multi-module GRPO** — RL over a whole composed pipeline, treating the program as one policy.

**Why interviewers care:** it reframes prompting as an *optimization problem with a train/dev split*, which is a more engineering-mature stance than artisanal prompt tweaking. It also forces you to have a metric — which loops back to evals.

**The honest caveat:** you need a labelled dev set and a trustworthy metric, compilation costs API calls, and compiled prompts are less human-readable. It shines when you have a well-defined task with data; it's overkill for a one-off.

---

## 11. RL fine-tuning: DPO, GRPO, RLVR

The alignment/fine-tuning ladder, in the order you should reach for it:

| Method | What it needs | Use it when |
|---|---|---|
| **Prompting / context** | nothing | Always try first |
| **SFT (+ LoRA/QLoRA)** | input→output pairs | You need format, style, or domain adherence |
| **DPO** (Direct Preference Optimization) | preference pairs (chosen/rejected) | You have preferences, want alignment without a reward model or RL loop |
| **GRPO** (Group Relative Policy Optimization) | a reward signal, group of samples per prompt | Reasoning/verifiable tasks; no value network needed |
| **RLVR** (RL with Verifiable Rewards) | an automatic verifier (unit tests, exact match, checker) | Math, code, structured extraction — anywhere correctness is machine-checkable |

**Key distinctions to have crisp:**
- **PPO vs DPO** — PPO needs a separate reward model plus a value network and an RL loop; DPO derives a loss directly from preference pairs, no reward model, far simpler and more stable.
- **GRPO's trick** — sample a *group* of completions per prompt, compute advantage as each completion's reward relative to the group mean, drop the value/critic network entirely. Cheaper and well-suited to verifiable-reward settings.
- **RLVR's appeal** — the reward is a program, not a human or a learned model, so it doesn't drift and can't be gamed by pleasing a rater. Its limit is that it only applies where you can *verify*.
- **Reward hacking** — always name it: the model finds a way to score well without solving the task. Mitigate with held-out verifiers, KL penalty to the reference model, and human spot-checks.

Tooling to name: **TRL** (HF training library), **Unsloth** (efficient LoRA), **vLLM** for fast rollout generation during training.

Cross-reference: `07_fine_tuning_and_peft.md`, `09_llm_alignment_rlhf.md`.

> **Your Jet2 story is a distillation story, not an RL story** — keep the two clearly separated. Distillation transfers a big model's behavior into a small one; RL changes behavior against a reward. Confusing them is a common tell.

---

## 12. Evaluation as infrastructure

The 2026 stance: **evals are not a report, they're a system that runs continuously**.

The hierarchy:
1. **Deterministic assertions** — schema valid, citation exists, no PII, latency budget. Cheap, fast, run on every request. Use these first; most teams skip straight to fancy metrics and miss free wins.
2. **Reference-based metrics** — exact match, F1, recall@k, MRR/nDCG for retrieval.
3. **Reference-free / LLM-as-judge** — groundedness, relevance, helpfulness. Powerful and biased; must be calibrated.
4. **Trajectory evaluation** (agents) — did it call sane tools, in a sane order, without loops? Judge the *path*, not just the destination.
5. **Human review** — the calibration anchor for everything above.
6. **Online** — user feedback, task success, escalation rate, thumbs, edit-distance-to-accepted.

**RAGAS-style decomposition** (know these four): **faithfulness** (is the answer supported by retrieved context?), **answer relevancy** (does it address the question?), **context precision** (is retrieved context on-topic / well-ranked?), **context recall** (did we retrieve everything needed?). The value is diagnostic: faithfulness low → generation problem; context recall low → retrieval problem. That decomposition is exactly what interviewers want to hear.

**LLM-as-judge biases you must name:** position bias (order of candidates), verbosity bias (longer looks better), self-preference (a model prefers its own style), leniency drift. Mitigations: randomize order, pairwise not absolute scoring, rubric with explicit criteria, calibrate against a human-labelled subset, and use a *different/stronger* model as judge than the one under test.

**Statistical honesty:** on a 60-example eval set, a 4% move is noise. Report bootstrap confidence intervals; use paired comparisons on the same items (much tighter than unpaired); and pre-register what "better" means before you look.

**Eval-gated CI** — golden set runs on every PR; a regression blocks merge. This one sentence signals production maturity more than any framework name.

Cross-reference: `14_evaluation_metrics.md`, `49_llmops_and_observability.md`.

---

## 13. Agent economics

A number worth memorizing: **an agent task typically costs 5–30× a single LLM call**, because the loop calls the model repeatedly, each turn re-sends grown context, and tool results add tokens.

The levers, roughly in order of impact:
1. **Cap iterations** — hard max steps; agents that can loop must have a budget.
2. **Prefix caching** — stable prompt prefix = cached KV = large cost/TTFT win in multi-turn loops.
3. **Model tiering / routing** — small model for routing, grading, extraction; large model only for synthesis or hard cases.
4. **Context compaction** — stop re-sending the full transcript.
5. **Cache tool calls** — deterministic tools with identical args shouldn't re-execute.
6. **Distillation** — once a task is stable, train a small model on the big model's outputs (your Jet2 70B→7B is exactly this).
7. **Batching** for offline/bulk paths.

**The KPI to quote:** not cost per token or per call, but **cost per resolved task** (or per accepted suggestion). It's the only number that trades off correctly against quality.

---

## 14. Real production architectures worth citing

Naming one real system makes an abstract answer land. A few well-documented patterns:

| Company | Pattern | The lesson to cite |
|---|---|---|
| **Uber** | GenAI Gateway fronting 60+ use cases with a shared PII redaction layer | Centralize safety/observability once, don't reimplement per team |
| **Anthropic** | Multi-agent research: a strong orchestrator model delegating to cheaper subagents | Orchestrator/worker tiering + context isolation per subagent |
| **Slack** | Stateless RAG with models in an escrow VPC | Compliance-driven architecture; no customer data retained in the model layer |
| **Perplexity** | Very high query volume served on a hybrid search engine (Vespa) | At scale, retrieval infrastructure — not the LLM — is the hard part |
| **Cursor** | Continuously retrains its suggestion-acceptance model on accept/reject signal | Online eval as a product loop, not a quarterly report |
| **GitHub Copilot** | Aggressive context assembly from open files/neighbors + tight latency budget | Context engineering *is* the product for code assistants |

> Use these as *analogies for your own choices*: "similar to the gateway pattern Uber describes, I centralized provider routing and fallback in one layer in AI Cargo so every tool inherited it."

---

## 15. MoE and what it means for serving

**Mixture-of-Experts:** replace a dense FFN with many expert FFNs plus a router that activates only a few per token. So **total parameters** ≫ **active parameters per token**.

Consequences you should be able to state:
- **Quality per FLOP improves** — you get big-model capacity at small-model compute per token.
- **Memory does not shrink.** All experts must be resident even though few fire → VRAM is driven by total params, compute by active params. This is the trap question.
- **Serving is harder** — routing imbalance ("expert collapse"), the need for **expert parallelism**, and batch-dependent efficiency.
- **Fine-tuning is harder** — router instability, uneven expert updates.

> **Trap:** "MoE models are cheaper to run." Half-true — cheaper *per token in compute*, not cheaper in *memory footprint*. Say both halves.

---

## 16. The 2026 buzzword decoder

Quick definitions so no term catches you flat:

| Term | One-line meaning |
|---|---|
| **Context rot** | Degradation as the window fills with stale/irrelevant content |
| **Compaction** | Summarizing history to reclaim context budget |
| **JIT retrieval** | Fetch context at the step that needs it, not upfront |
| **Prefix caching** | Reusing KV cache for an unchanged prompt prefix |
| **Agent harness** | The scaffolding around a model: tools, loop, state, guards |
| **Trajectory eval** | Scoring the agent's path, not just its final answer |
| **RLVR** | RL where the reward comes from an automatic verifier |
| **GRPO** | Group-relative advantage RL; no critic network |
| **Late chunking** | Embed the long doc, then pool chunk vectors from it |
| **Contextual retrieval** | Prepend generated context to each chunk before embedding |
| **Late interaction** | Multi-vector per-token matching (ColBERT-style) |
| **Listwise reranker** | Rerank a whole list jointly rather than pairwise |
| **Computer use** | Agents driving a GUI/browser via screenshots and actions |
| **Sandboxing** | Isolated execution for agent-generated code/actions |
| **Confused deputy** | Agent misuses its own privileges on behalf of a user who lacks them |
| **Goodput** | Throughput that actually meets your latency SLO |
| **Speculative decoding** | Draft model proposes tokens, target verifies — lossless speedup |
| **Semantic caching** | Cache by query *meaning* rather than exact string |
| **Model routing** | Send each request to the cheapest model that can handle it |
| **Eval-gated deploy** | Golden-set regression must pass before merge/ship |

---

## 17. Interview Questions with Strong Answers

**Q1: "How would you build a RAG system for [domain]?"** *(the most-asked question in the field — have this memorized)*
> "I'd start with the data, not the model: what format are the documents, how often do they change, and what's the access-control model — because permissions decide the architecture more than anything else.
>
> Then ingestion: parse structure rather than flatten it, chunk with the retrieval unit and the citation unit chosen separately — I like indexing small precise units and returning the larger parent for context. Embed with a strong general model, store in something that supports hybrid search — pgvector, Qdrant, or Vespa at scale.
>
> Retrieval: BM25 plus dense, fused with reciprocal rank fusion, then a cross-encoder rerank on the top ~20. Hybrid matters because domain corpora are full of exact terms embeddings blur. Then context assembly with the citations preserved, and a prompt that requires grounded answers and permits refusal.
>
> Then the part candidates skip — evaluation. I'd build a golden set of 50 to 200 representative queries with the expected source documents, measure retrieval recall@k and answer faithfulness, and log every failure. That set goes into CI so a change can't silently regress it.
>
> Production concerns: caching, a latency budget, PII handling, and monitoring retrieval-miss rate and refusal rate as health metrics. And I'd start with the boring version — hybrid retrieval and a good eval set beat a clever architecture with no measurement."

**Q2: "What's the difference between a workflow and an agent, and which should I build?"**
> "A workflow follows branches you wrote; an agent decides its next step at runtime from context. The real question is whether you need runtime reasoning — and usually you don't. If the decision logic is expressible deterministically, a workflow is cheaper, faster, testable, and debuggable. I reach for an agent when the path genuinely depends on intermediate results in ways I can't enumerate. In my AI Cargo project I ended up with a bounded graph rather than a free-roaming agent for exactly that reason, and I hard-filtered which tools the model could propose — constrain in code what code can guarantee."

**Q3: "Your agent works in the demo and fails in production. Debug it."**
> "First I look at traces, not code — I want the failing trajectory: which tools were called, with what arguments, what came back, and where the context went wrong. The common causes in my experience are context problems, not model problems: a poisoned context from one bad tool result, context rot in long loops, or tool-choice degradation because too many tools are in the prompt. Then I check whether the failure is deterministic or sampling-dependent by replaying the same input. If it's grounded in retrieval, I check retrieval recall separately from generation quality, because those need different fixes. Finally I add the failing case to the golden set so it can't regress again."

**Q4: "Evals: how do you know your LLM app got better?"**
> "Layered. Deterministic assertions first — schema validity, citation existence, latency, no PII — those are cheap and catch most regressions. Then reference-based retrieval metrics, recall@k and nDCG, on a labelled set. Then LLM-as-judge for the subjective dimensions, but calibrated against a human-labelled subset and with position randomization and pairwise comparison to control bias. For agents I also evaluate the trajectory. And I hold the statistics honestly: on a small set a few points is noise, so I report bootstrap confidence intervals and use paired comparisons on identical items. The whole set runs in CI so a prompt change can't ship a regression."

**Q5: "LLM-as-judge is biased. Why should I trust it?"**
> "You shouldn't trust it unconditionally — you should calibrate it. I measure agreement between the judge and human labels on a subset; if agreement is poor the judge is useless and I fix the rubric or the judge model. Then I control the known biases: randomize candidate order for position bias, use pairwise preference rather than absolute scores, give an explicit rubric to reduce verbosity bias, and avoid using the same model family as judge and contestant to limit self-preference. It's a noisy instrument that scales — humans are the accurate instrument that doesn't. You use the cheap one continuously and the expensive one to keep it honest."

**Q6: "Why is MCP a big deal? Isn't it just function calling?"**
> "Function calling is the model's ability to emit a structured call. MCP is the integration standard around it — it turns an N-times-M integration problem into N-plus-M. I write a tool server once and any MCP-capable host can discover and use it, with a consistent notion of tools, readable resources, and reusable prompts. The trade-off worth raising is security: MCP widens your trust boundary. Tool descriptions and results enter the model's context, so they're an injection surface. I'd allowlist servers, scope credentials per server, sandbox execution, treat all tool output as untrusted data rather than instructions, and gate consequential actions behind a human."

**Q7: "When would you use GraphRAG over vector search?"**
> "When the question is relational rather than lookup. Vector search answers 'what does the corpus say about X'; a graph answers 'how does X connect to Y through Z' and supports corpus-global questions like 'what are the recurring themes.' Multi-hop is the clearest case, because with vectors each hop is a fresh fuzzy search and errors compound. The costs are real though — entity and relation extraction over the corpus is expensive and updates are harder than a re-embed. So in practice I'd go hybrid: vectors for lookup, graph for relationships, and I wouldn't pitch GraphRAG as a general upgrade."

**Q8: "How do you control the cost of an agentic system?"**
> "I measure cost per resolved task, not per call, because that's the number that trades off correctly against quality. Then: hard-cap iterations so a loop can't run away; structure prompts so the stable prefix caches; tier models so a small one does routing, grading, and extraction and a big one only synthesizes; compact context instead of re-sending transcripts; cache deterministic tool calls; and once a task stabilizes, distill it into a small model. That last one I've actually done — at Jet2 I distilled a 70B teacher into a 7B student to make intent classification deployable at scale while holding 94% recall."

**Q9: "Reasoning models — when do you use one, and when don't you?"**
> "I route by difficulty and by cost of being wrong. Reasoning models buy accuracy with latency and tokens, so they earn their place on the hard tail — ambiguous multi-step problems where a wrong answer is expensive. For high-volume, well-specified tasks like extraction or classification, they're waste; a small model with good context and validation wins on cost and latency. The interesting second-order effect is that reasoning models collapsed some agent workflows into single calls, so part of the job now is *not* building an agent that a single strong call would handle."

**Q10: "What's DSPy and would you use it?"**
> "It treats prompting as compilation: you declare a typed signature, pick a module like ChainOfThought or ReAct, define a metric, and an optimizer searches instructions and few-shot demonstrations against that metric. Optimizers range from BootstrapFewShot mining successful trajectories to MIPROv2 jointly searching instructions and demos. I'd use it when I have a well-defined task, a labelled dev set, and a metric I trust — because then prompt engineering becomes an optimization problem instead of a craft. I wouldn't use it for a one-off, and I'd be honest that its output is less human-readable and that compilation itself costs API calls. Also the metric is load-bearing: a bad metric compiles a confidently bad program."

**Q11: "Explain GRPO vs DPO vs PPO."**
> "PPO is classic RLHF: train a reward model, then optimize the policy with a value network and a clipped objective — powerful but heavy and finicky. DPO removes the reward model entirely; it derives a loss straight from chosen-versus-rejected preference pairs, which is far simpler and more stable and is why it became the default for preference tuning. GRPO removes the critic instead: you sample a group of completions per prompt and use each one's reward relative to the group mean as the advantage, which suits verifiable-reward settings like math or code where a program can score correctness. That pairing — GRPO with verifiable rewards, or RLVR — is what's driving current reasoning training. Whichever you use, I'd name reward hacking as the failure mode and keep a KL penalty to the reference model plus held-out verifiers."

**Q12: "Are MoE models cheaper to serve?"**
> "Cheaper in compute per token, not in memory. The router activates only a couple of experts per token, so FLOPs track the active parameters — but every expert has to be resident in VRAM, so your memory footprint tracks total parameters. So you get big-model quality at small-model compute, while paying big-model memory. Serving also gets harder: routing imbalance, the need for expert parallelism, and efficiency that varies with batch composition."

**Q13: "What's context engineering and why do people say it replaced prompt engineering?"**
> "Prompt engineering is optimizing one question. Context engineering is managing what information reaches the model across many steps — and for agents that's where quality and cost actually come from. The failure modes have names now: context rot as the window fills with stale content, lost-in-the-middle where buried facts get ignored, poisoning where one bad tool result contaminates every later turn. So the practices are compaction, ordering stable content first so it caches, just-in-time retrieval instead of preloading, retrieving tools when there are too many to list, and giving sub-agents isolated windows so the parent only sees conclusions. It's a resource-management discipline, which is why it feels more like engineering than prompting."

**Q14: "You've listed a lot of 2026 tooling. What would you actually pick for a greenfield project?"**
> "Deliberately boring: a strong provider API to start rather than self-hosting, hybrid retrieval on pgvector or Qdrant, a cross-encoder reranker, LangGraph if the control flow genuinely branches, tracing from day one with LangSmith or Langfuse, and a golden set in CI before I optimize anything. I'd add complexity only against evidence — GraphRAG when I see multi-hop failures, DSPy when prompt tuning becomes the bottleneck, self-hosted vLLM when cost or data residency demands it. The thing I'd never defer is observability and evals, because without them every later decision is a guess."

---

## 18. Key Takeaways

- **Evaluation is the differentiator.** In 2026 loops, the candidate who can measure beats the candidate who can only build. End every answer with how you'd know it works.
- **Context engineering > prompt engineering.** Know the failure modes by name: rot, lost-in-the-middle, poisoning, distraction.
- **MCP standardized tools** ($N \times M \to N + M$) — and widened the security boundary. Always mention both.
- **Memory is a designed subsystem**, and the hard part is the write path: what to commit, how to resolve conflicts, how to forget.
- **Retrieval has moved on** — hybrid + RRF + reranking is the *baseline*; late chunking, contextual retrieval, and late interaction are the current frontier.
- **GraphRAG is a specialist**, not an upgrade. Multi-hop and global questions; expensive to build.
- **Test-time compute added a third dial** — route by difficulty, and know when a single reasoning call replaces an agent.
- **DSPy makes prompting an optimization problem**, but only if your metric is trustworthy.
- **GRPO + verifiable rewards** is the current reasoning-training recipe; DPO is the simple preference default; name reward hacking.
- **MoE:** small-model compute, big-model memory.
- **Agents cost 5–30× a single call.** Cap iterations, cache prefixes, tier models, distill when stable. Measure cost per *resolved task*.
- **Prefer boring, instrumented, measured systems.** Add complexity against evidence — that's the answer that reads as senior.
