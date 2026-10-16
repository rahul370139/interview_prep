# Lam Research · Hiring manager call with Jatin Sasan (45 min)

Michelle's screen and the questionnaire are done. Next is a 45-minute conversation with Jatin Sasan, Director of Generative AI, in 6 days. He is a GenAI leader with a semiconductor equipment-software background, and he is building a new team. Expect more technical depth than a typical manager call: how you'd design, evaluate, and ship LLM systems, plus behavioral questions scored against Lam's values. The plan and study map come first, then the full question bank.

**Problem first, then the tech** · **Say when you would not use an LLM** · **One number per story** · Slow down, signpost, pause

| 45 min | 6 days | Director, GenAI | ~2.5 h/day | Very strong |
|---|---|---|---|---|
| With Jatin Sasan | To prepare | Who you're meeting | Study + speaking | Your match |

> **What Jatin is really deciding**
>
> (1) Is the depth real? Can you go two levels down on RAG, evaluation, serving, and agents without hand-waving? (2) Do you think like an engineer, not a demo builder: tradeoffs, failure modes, and how you measure? (3) Can you be one of the first people on a new team: own something end to end and explain it to process engineers? (4) Do you want Lam specifically? Your English does not need to be perfect. Clarity and structure are what get scored.

---

## Who you're talking to: Jatin Sasan

### His background (public LinkedIn)

- Director, Generative AI at Lam Research since January 2026.
- Before that, software engineering leadership at Applied Materials, and earlier, C/C++ software for a lithography tool. His whole career is semiconductor equipment software.
- Stanford LEAD executive program.
- What it means: he knows how tool software fails. He cares about reliability, testing, and safety as much as about models.

### What he has said he cares about

His post when he joined (Jan 15, 2026): "Generative AI in this space isn't about hype. It's about solving hard, specific problems: reducing time-to-solution in process recipes, augmenting engineering judgment with data-driven insights, making our tools smarter and more predictive."

He named process development, yield optimization, and equipment intelligence, and said: "I'll be building a team to tackle these challenges."

### How to impress him

- **Problem and measurement first,** technology second.
- **Say when you wouldn't use an LLM.** Your Jet2 "stats first" story is perfect for this.
- **Talk like an equipment engineer:** failure modes, fallbacks, human sign-off, logs. Your AI Cargo veto is a software safety interlock.
- **Show you can build foundations** for a new team: eval harness, tracing, guardrails.
- **Treat data as IP.** Bilbo's "data can't leave the box" maps to customer recipe data.

### Lam research to mention once: the Nature paper (2023)

Lam researchers published "Human–machine collaboration for improving semiconductor process development" in Nature ([link](https://www.nature.com/articles/s41586-023-05773-7)). In a virtual plasma-etch process game, expert engineers did best early, Bayesian optimization was more cost-efficient near the tight target, and a "human first, computer last" strategy cut the cost to reach the target by about half compared with engineers alone. The paper also names the cultural challenge of getting engineers to work with algorithms.

Lam also holds patents on ML-generated process recipes, and a recent SPIE talk by a Lam process engineer described Lam's internal AI/ML platform for optimizing etch recipes. So ML for recipes already exists there. GenAI is the layer that connects knowledge, people, and those tools.

### How to use it in the call

"I read Lam's Nature paper on human-machine collaboration in process development. The human first, computer last result is how I think about GenAI here too. The engineer sets the direction, the system does the narrow search and the paperwork, and the engineer makes the call."

"When you joined, you wrote that GenAI here isn't about hype, it's about hard, specific problems. That's how I work too. At Jet2 we started with statistics and rules before any model."

> **Your one-line positioning for Jatin**
>
> "I see GenAI at Lam as a layer that helps engineers find, reason over, and act on what Lam already knows. It sits on top of good data and classical models, it shows its evidence, and a human makes the call."

> **Don't**
>
> Don't go into detail about his LinkedIn activity. One reference to his post or the Nature paper is plenty. Don't claim fab or process experience you don't have. Say "I'd learn the domain from the process engineers first." Don't oversell vLLM or CAD experience. Say what you ran, then how you'd scale it.

---

## Your 6-day plan (about 2 hours of study and 30 minutes of speaking a day)

Each day is a study block from the map below plus a speaking block. Say every answer out loud. Reading an answer is not the same as being able to say it.

- [ ] Day 1 · Jatin + serving. Read 'Who you're talking to' and the Lam intel. Study map #1 (learning/50). Say A1 (LLM serving) out loud twice, record once. Warm-up: intro and why Lam.
- [ ] Day 2 · Hallucination + evaluation. Study map #2 and #5. Say A2 and B2 out loud. Practice the follow-ups 'RAG still hallucinates, debug it' and 'why trust LLM-as-judge'.
- [ ] Day 3 · Embeddings + RAG for specs. Study map #3 and #4. Say A3. Then draw B1 (RAG over manufacturing specs) on paper and talk through it in 6 minutes, as if screen-sharing.
- [ ] Day 4 · Agents + Jatin's vision questions. Study map #6, #7, #8. Say B3 (CAD multi-agent) and G1 to G5. Re-practice the AI Cargo walkthrough and PHANTOM TOOLS, since agent questions lead there.
- [ ] Day 5 · Equipment + coding + behavioral. Study map #9, #10, #11. Type D1 to D3 from memory in a plain editor. Say C1, C2, E1, and the Jet2 conflict story. Evening: full 45-minute mock using the prompt below. Record it and note the 3 weakest answers.
- [ ] Day 6 · Fix and rest. Re-record the 3 weakest answers. Pick 2 or 3 questions to ask Jatin. Skim the whole doc, no new material. Write a one-page sheet of your numbers. Sleep early.
- [ ] Call day · 10-minute warm-up: intro plus one technical answer out loud, 5 minutes of shadowing. Water, quiet room, headphones, resume and number sheet open. Join 3 minutes early.

> **Mock interview prompt (paste into any AI chat, use voice mode if you can)**
>
> "Act as Jatin Sasan, Director of Generative AI at Lam Research, interviewing me for Software Engineer Sys 2 (Gen AI). You came from semiconductor equipment software and you are building a new team. Ask one question at a time. Mix project deep dives, LLM system design, evaluation, agents, and Lam's values. Push back with 'why' at least twice per answer. At the end, score me 1 to 4 on depth, clarity, judgment, and Lam fit, and tell me my 3 weakest answers."

## Study map: what to read in interview_prep, in order

Section numbers refer to files in `interview_prep/learning` and `interview_prep/projects`. If time is short, do #1 to #6 and #11. Those cover 90% of what he is likely to ask.

| # | Topic | Read | Why for Jatin | Time |
|---|---|---|---|---|
| 1 | LLM serving + latency | 50 §2, §3, §4, §5, §6, §9, §12, §14 · Q9, Q10, Q11, Q13, Q14, Q22, Q24, Q25 | Question A1, maybe word for word | 2 h |
| 2 | Hallucination + grounding | 48 §4.3 Q27 · 49 §4.2, §4.3, §7.1, §7.2 · Q3, Q11, Q16, Q17 · 30 §4, Q2 | Question A2; trust in technical answers | 1.5 h |
| 3 | Embeddings for technical text | 18 §3, §4, §6 · 06 §3, §7, Q9 · 05 §2b, §2c, §3, §4 · P07 §2.2 | Question A3; jargon, part numbers | 1 h |
| 4 | RAG over manufacturing specs | 52 §6, §7, §8, §17 · 05 §2a to §2i, §9, Q1, Q8 · 48 §5.2 · 26 §2c | Question B1; Lam's core GenAI pattern | 1.5 h |
| 5 | LLM evaluation | 49 §4.1 to §4.6, §9 · Q2, Q4, Q9, Q10, Q21, Q22 · 52 §12 · 14 §4, §6, §7 | 'How do you know it works?' | 1 h |
| 6 | Agents + multi-agent | 10 §4, §5.2, §7, §10 · Q1, Q4, Q6 · 27 §6, §9.3, §10 · 52 §2, §4 · P15 | Question B3; agent traces | 1 h |
| 7 | Observability, cost, latency | 49 §3, §6 · Q6, Q12, Q14, Q15, Q20, Q23 | Production thinking | 45 min |
| 8 | Context engineering | 16 §1.3, §3, §9 · Q1, Q2 · 52 §3 | Prompt vs instruction vs skill follow-ups | 30 min |
| 9 | Monitoring, drift, scaling | 26 §1 Step 6, Q6 · 49 §8 · 50 §12, §13 · P03, P04 | Equipment intelligence; C1 to C3 | 1 h |
| 10 | Coding warm-up | 25 §0, §2, §5.3, §5.5 + D1 to D5 below | In case he asks for code | 1 h |
| 11 | Behavioral | 53 §2.1, §4, §6, §7, §12 + scripts in this doc | Lam values, scored with STAR | 1 h |

### Your numbers, in one place

| Story | Numbers to say |
|---|---|
| Bilbo (RAG) | +28% top-five relevance vs dense only · under 5 s · 4-bit Mistral-7B (~4.4 GB) on Ollama · MiniLM 384-d · top 50 → rerank top 20 |
| AI Cargo (agents) | First prize, Smith Agentic AI Challenge ($4,000) · solo · 6 weeks · deterministic veto + human gate |
| Jet2 forecasting | MAPE 18% → 7% · under 2% error variation for 18+ months · walk-forward validation |
| Jet2 event detection | Moving averages + z-scores (\|z\| > 3) + rules → grid-tuned → Isolation Forest · ~96% specificity |
| Jet2 intent (70→7) | 70B distilled to 7B · 94% recall · 1M+ call transcripts |
| ATCS ETL | 10M+ records · 4 h → 18 min · config-driven DQ on 4+ engagements · onboarding 2–3 weeks → 2–3 days |
| Fibe | Model cycle 3 days → under 5 hours (15×) · MLflow tracking + validation gates |

---

## Lam intel to know cold (so you sound informed, not generic)

### What Lam does (the domain)

Lam makes the equipment that chipmakers (TSMC, Micron, Samsung) use to build chips. Two core processes:

**Deposition** = adding ultra-thin layers of material onto the wafer (think "adding layers"). **Etch** = selectively removing material to carve the tiny patterns (think "cutting the pattern"). These repeat hundreds of times, at atomic-level precision, to build one chip.

This role sits in the Global Products Group, the org behind the etch & deposition tools.

### Lam's AI strategy (why this role exists)

Lam is building an enterprise GenAI platform on Azure / Microsoft Fabric / Databricks, using RAG, vector search, and knowledge graphs across R&D, manufacturing, service, supply chain, and compliance.

Concrete examples: a field-service AI trained on 30+ years of troubleshooting data (already used by ~2/3 of field engineers), and virtual-twin modeling via Semiverse Solutions to speed up development.

Their internal RAG platform is config-driven and multi-tenant, with LangGraph agents and RBAC. Your Bilbo + AI Cargo work is the same shape.

### Business context (one-liners)

CEO Tim Archer; focus on adding sensing + AI to tools, and US expansion (Arizona, California). Stock up ~75% this year on AI chip demand; the industry is "fundamentally sold out."

Lam's own filings call generative and agentic AI a governance risk that needs authorization, data controls, human oversight, and logging.

> **The connection to make out loud**
>
> Lam publicly treats agentic AI as something that must be governed: human oversight, access control, logging. That is exactly how you build. Your line: "The thing I care about most is making AI you can actually trust in production, grounded, bounded, and logged, and that's clearly how Lam thinks about it too." That one sentence tells the hiring manager you're already culturally aligned.

---

## Lam's 9 core values → your story for each

Lam uses Behavioral Event Interviewing: competency questions scored on STAR. These are their actual values. Have one story ready for each so no matter how the question is worded, you have an answer with a number.

| Lam value | What they want to hear | Your story (codename + number) |
|---|---|---|
| Achievement | Aggressive goals, delivered results | $4,000 win (solo, 6 wks) · 18→7 MAPE · 15× cycle |
| Agility | Speed, flexibility, quality | $4,000 (6-week solo build) · 4h→18min ETL |
| Honesty & integrity | Does the right thing, owns the truth | PHANTOM TOOLS (owned the bug) · governance-first design |
| Inclusion & diversity | Welcomes different perspectives | Jet2 ops and planning, plus risk and business stakeholders |
| Innovation & continuous improvement | New ideas, embraces change | +28% hybrid retrieval · EVAL HARNESS · 70→7 distillation |
| Mutual trust & respect | Treats others well in disagreement | Jet2 event detection: stats first, then Isolation Forest (96% specificity) |
| Open communication | Speaks candidly, listens, asks | Explaining alerts and forecasts to Jet2 ops and planning |
| Ownership & accountability | Takes responsibility, delivers | 18→7 live 18 months · PHANTOM TOOLS · 4h→18min |
| Teamwork | Collaborates to a shared goal | Stakeholder explanation at Jet2, in fintech, and on GenAI work |

---

## How to approach the 45 minutes with a GenAI director

| Phase | Roughly | What happens | Your job |
|---|---|---|---|
| Warm-up + intro | 5 min | Small talk, then 'tell me about yourself' | Warm 60-second intro, land on why this team |
| Projects, in depth | 10–15 min | AI Cargo and Bilbo, then 'why that choice?' | Problem, design, tradeoff, number. Expect two levels of 'why' |
| Technical discussion | 10–15 min | 'How would you design / evaluate / reduce X' (likely A1 to A3, B1 to B3) | Clarify first, structure out loud, end with how you'd measure it |
| Vision + judgment | 5 min | 'Where does GenAI help Lam?' 'When not to use an LLM?' | Recipes, knowledge, equipment intelligence; a human makes the call |
| Behavioral | 5–10 min | 'Tell me about a time…' on Lam's values | STAR-L, one number each, under 2 minutes |
| Your questions | 5 min | 'What questions do you have for me?' | 2 or 3 sharp ones about his team and first problems |

> **The single habit that wins this round**
>
> End every project and behavioral answer with how you knew it worked (a number) and what you learned. Most candidates end on a feeling. You end on evidence. That is the difference between a 2 and a 4 on their sheet.

> **How to handle a design question in the call (the 4-step shape)**
>
> (1) Clarify for 30 seconds: users, data, scale, latency, accuracy, who can see what. (2) Say the shape: "I'd break it into ingestion, retrieval, generation, evaluation, and operations." (3) Go through each part, naming one tradeoff per part. (4) Close with how you'd measure it and pilot it, and one line tying it to your own project. If you get stuck, say "Let me think about that for a second" and go back to the shape.

---

## Scripts to practice out loud (your voice, simple English)

Short sentences on purpose. Practice saying these until the rhythm is natural, then make them yours. Do not memorize word-for-word; memorize the shape and the numbers.

### 1 · Tell me about yourself (~60 seconds)

"Sure. I'm a data scientist and AI engineer with about five years of experience, and I just finished my master's in data science at Maryland in May. Most recently I was an AI engineer at Bilbo, where I built a healthcare AI assistant. It answers questions from medical documents and cites exactly where each answer came from, running on a private, self-hosted model because the data couldn't leave the environment. It made the search about 28% more accurate, with answers under five seconds."

"Before my master's, I was a data scientist at Jet2 in the UK for two years, working on forecasting and language models over call-center data, and earlier I did credit-risk and fraud work in fintech. On the side I build AI agents. I won first prize in an agentic-AI competition for a shipping-risk platform."

"What drew me to this role is that it's exactly the work I want to do: LLM applications, retrieval, and agents, with the reliability to trust them. And doing it for semiconductor engineering is genuinely interesting to me, because that's a domain with deep, high-stakes knowledge where this really matters."

*For Jatin, you can swap the last line for: "And what you wrote about solving hard, specific problems like process recipes, not hype, is exactly the kind of GenAI work I want to do."*

### 2 · Why Lam, why semiconductors? (deeper than the recruiter version)

"A few reasons. First, the work lines up almost exactly with what I've built, RAG, agents, and the observability that makes them reliable. Second, I like where Lam sits: you make the equipment that basically the entire chip industry depends on, so there's real weight to it. And I read that Lam is investing seriously in internal GenAI, across R&D, manufacturing, and field service, which is a huge, high-value problem."

"The part that really fits me is that Lam treats agentic AI as something you have to govern carefully, with human oversight and logging. That's exactly how I build. So it feels less like a stretch and more like the same values at a much bigger scale."

*Optional third line for Jatin: "I also read Lam's Nature paper on human-machine collaboration in process development. The idea of engineers first and the algorithm last is how I'd want to build GenAI tools here."*

### 3 · Walk me through Bilbo (your RAG project, ~2 min)

"At Bilbo I built a production assistant that answers questions from dense medical protocol documents. Two things made it hard: every answer had to be citable and trustworthy, and the data couldn't leave the environment for compliance. So the compliance constraint actually shaped the whole design."

"On the retrieval side, I used two kinds of search together, a vector index for meaning and keyword search for exact terms, and combined them, then re-ranked the top results for accuracy. That lifted top-five relevance about 28% over a vector-only baseline. Generation runs on a quantized model hosted locally, and the whole flow is a LangGraph workflow with guardrails, so it refuses when it doesn't have a good answer instead of guessing. End to end it's under five seconds, and I measured retrieval quality on a labeled set so the 28% is a real number, not a feeling."

### 4 · Walk me through AI Cargo (your agent project, ~2 min)

"AI Cargo is an agentic risk platform for pharmaceutical cold-chain shipments, and it won first prize in the Smith Agentic AI Challenge. It scores shipment risk in real time and flags problems before they become spoilage."

"The main idea is that the model shouldn't be the only decision-maker when safety and compliance are on the line. So I combined deterministic compliance rules with a machine-learning risk score, both under a LangGraph agent that reasons and calls tools. The rules get a hard veto, the agent can't override a real violation, and anything uncertain goes to a human for sign-off, with a full audit trail. I built the whole thing solo in about six weeks. It won because it was the one you could hand to a regulator, not the one with the prettiest demo."

### Behavioral STAR answers (one per competency)

### Failure / "a time you were wrong" — PHANTOM TOOLS

"On my agentic platform, the agent had a reflect-and-revise loop, and it started re-running tools that had already finished, and flagging optional tools as missing. The tricky part was the dashboard looked active and healthy, so the bug was basically invisible."

"What I did was instrument every step, log the input and output of each node, and I traced it to the model inventing extra work in the reflect step. I fixed it by constraining that step in code, so it could only ever emit tools that were mandatory or had actually failed, and I redesigned it to act first and review after. The phantom executions went to zero. What I took from it: an agent that looks busy is not the same as an agent that's correct, and if code can guarantee a constraint, I don't leave it to a prompt."

### Ownership / production — 18→7

"At Jet2 I owned a demand-forecasting system end to end. I ran a bake-off between XGBoost, LSTM, and GRU, tuned it, and used walk-forward validation instead of a random split, because a random split on time series leaks the future. That took the error, MAPE, from about 18% down to 7%."

"But the part I'm proud of is that I put it into production with scheduled retraining and validation checks, and it held under 2% error variation for more than eighteen months. Getting to 7% took about a week. Keeping it at 7% for eighteen months was the real job."

### Ambiguity / shipping fast — $4,000

"The clearest example is the agentic platform I built for the Smith challenge. There was no spec, I defined the problem myself: cold-chain pharmaceutical risk, where compliance really matters. I built the whole six-layer system solo in about six weeks, under a deadline. And what won it wasn't demo polish, it was the governance: rules that can veto the model, a human gate on anything consequential, and a full audit trail."

### Hard technical decision — 70→7

"At Jet2 I was classifying intent over a million-plus call transcripts. The most accurate model was a 70-billion-parameter one, but it just wasn't deployable at contact-center volume. So instead of compromising on accuracy or shipping something too slow, I distilled it, I trained a 7-billion model to inherit the behavior of the 70-billion one, and kept 94% recall at a size we could actually serve. The way I think about it: the best model and the shippable model were different models, so I made the shippable one inherit from the best one."

### Conflict — Jet2 event detection (stats first, then the model)

"This was at Jet2. We needed to find real operational events in the live chat data, so the ops team got an alert when something actually changed, not a pile of noise.

A teammate wanted to go straight to a machine learning model. I understood the point. We had a lot of data, and a model can pick up combinations you didn't write down. My pushback was that we didn't understand the data well enough yet. I wanted to look at it first, build the features, and see how far simple statistics could take us. Not every problem needs a complex model, and if you start there you often can't explain what fired, or why.

So we did it that way first. Moving averages for what normal looks like at that time of day, z-scores for a spike against recent history, and a small set of rules on top of those. That already caught a good share of the real events, and ops could follow it, which mattered because they were the ones acting on the alert.

Then we took those same signals and grid-tuned the thresholds. That cleaned up which signals were actually useful and which were just noise. Once we had that, we added Isolation Forest on top of those features. That was the part my teammate was right about. The rules catch a loud single signal. The forest catches the case where no one signal is loud, but the combination is. So the model wasn't thrown out. It came last, on features we already trusted.

The full setup landed at about 96% specificity, so a low false-alarm rate, and the ops team actually used the alerts. What I took from it is that both of us were partly right. Understanding the data first made the model better, and the model caught things the rules alone would miss."

*Scored on how you treated the other person, not on who won. You state his view fairly, you go first with the simple approach, and you say where he was right. If they ask a name, keep it as "a teammate" unless you want to name them. For Jatin this story does double duty: it is also your best answer to "when would you not use an LLM or a complex model?"*

### Teamwork / explaining to stakeholders

"Explaining the work to stakeholders has been part of almost every role I've had. The model is only useful if the person who has to act on it understands it.

At Jet2 that was constant, for more than two years. Operations and planning were not data scientists. On the event detection, I couldn't hand them an anomaly score. I had to say what changed, against what normal looks like for that time of day, and what they should do next. On the forecast, the error number only mattered if planning could use it, so I spent as much time on the explanation and the checks as on the model itself.

Same pattern earlier in fintech. Credit and fraud results went to risk and business people, so the reporting was built around the decision, not around the algorithm. And it's the same shape on the GenAI side. An answer only lands if the person using it can see where it came from and decide whether to trust it.

The habit is the same in all of them. I start with their question, not my method. I say what it means for them, what I'd want them to double-check, and I only go into the technique if they ask."

### Learning something fast

"For the agentic challenge I had about six weeks and had to get fluent fast in the agent stack, LangGraph, tool calling, multi-provider fallback. The way I learn fast is I build the smallest working version first, then break it on purpose to understand how it fails, which is actually how I found the phantom-tool bug. By the end I had a production-style system and a real understanding of where these agents break, not just how to make them run."

---

## Technical question bank for Jatin

> **Where these questions come from**
>
> Dataford and PracHub show public question titles for Lam AI and software roles, and keep the answers behind a paywall. Only those public titles were used here. Every answer is written from your own projects and your interview_prep notes. Questions tagged "Likely from Jatin" are predictions from his background and his post. Each question has a spoken answer, the follow-ups he may push on, and where to study it.

## A · The three questions you found (do these first)

These are the most likely to come up, close to word for word. Practice each until you can say it in about 3 minutes without notes.

### A1 · Describe the architecture for LLM serving with low latency in production

*Tag: You found · top priority*

**What he's testing:** Whether you know the request path end to end, the engine-level levers, and how you'd prove latency with numbers. He will also notice whether you're honest about what you have and haven't run.

```text
User
  |  HTTPS, streamed tokens (SSE)
Gateway ........ auth, rate limits, per-team quotas
  |
App service .... async orchestration, timeouts, tracing
  |-- Cache ...... exact match (Redis) + semantic, keyed by tenant + index version
  |-- Retrieval .. hybrid ANN + BM25, rerank only the top 20 to 50
  |
Model router ... small model by default, large for hard cases, fallback
  |
Inference pool . vLLM: continuous batching, PagedAttention, prefix cache,
  |              AWQ / FP8 quantization, chunked prefill, speculative decoding
  |
Autoscaler ..... scale on queue depth + KV-cache %, warm minimum replicas
Telemetry ...... queue time, TTFT, TPOT, tokens, cost per request
```

**Say it like this**

First I'd pin down what low latency means for this product, because there are two numbers. Time to first token is what the user feels as speed. Time per output token decides how long the full answer takes. I'd set targets at p95, not the average. For example, first token under one second at a stated number of concurrent users.

Then the request path. A gateway handles auth, rate limits, and per-team quotas. Behind it, an async service, FastAPI for example, does the orchestration. Before anything touches a GPU, it checks a cache, exact match in Redis and maybe a semantic cache. If it's RAG, retrieval runs in the same region with an ANN index plus BM25, and I rerank only the top 20 to 50.

Next, a model router. Most requests go to the smallest model that passes our evals. Only the hard ones go to the bigger model. The models run on an inference engine like vLLM, and tokens stream back to the user.

Inside the engine, the levers are well known. Continuous batching, so new requests join without waiting for a batch to finish. PagedAttention, so the KV cache doesn't waste memory and more users fit on one GPU. Prefix caching, so a long shared system prompt is computed once. Quantization like AWQ or FP8, to cut memory traffic. Chunked prefill, so one long prompt doesn't freeze everyone else. And speculative decoding when the GPU isn't fully loaded.

At the application level, the cheapest win is a shorter prompt. Fewer, better chunks, a cap on output tokens, and fewer LLM calls in a row in an agent chain.

For scale and reliability, I autoscale on queue depth and KV-cache usage, not CPU. I keep warm replicas on the interactive path, because loading a model takes minutes. Interactive and batch traffic get separate pools. If something degrades, I fall back to a smaller model or a cached answer instead of timing out.

And I trace every stage: queue time, retrieval, rerank, first token, total. When p95 moves, I know which stage moved. Before launch, I load test with real prompt lengths to find where latency starts to climb.

In my own work, Bilbo ran under five seconds end to end on a 4-bit Mistral-7B. When I traced it, generation was most of the time and retrieval was small. So the wins came from the quantized model, a short structured answer, and reranking only the top 20. That ran on Ollama on one machine. To serve many concurrent users, I'd move it to vLLM for continuous batching and prefix caching.

| Stage | p95 budget (illustrative) | Main lever |
|---|---|---|
| Gateway + auth | 10–30 ms | Same region, connection pooling |
| Cache check | 5–10 ms | A hit skips the GPU entirely |
| Hybrid retrieval | 50–150 ms | HNSW settings, metadata pre-filter |
| Rerank | 50–150 ms | Rerank 20–50 chunks, not 200 |
| Queue wait | under 100 ms | Autoscale on queue depth, priority lanes |
| Prefill (first token) | 200–600 ms | Shorter prompt, prefix caching, chunked prefill |
| Decode | 15–30 ms per token | Quantization, speculative decoding, cap output, stream |

**Follow-ups he may push on**

- **TTFT or TPOT, which matters more?** Depends on output length. Short chat answers: TTFT. Long reports: TPOT. Prefill sets TTFT and is compute-bound, so long prompts hurt it. Decode sets TPOT and is memory-bandwidth-bound, so big models and big batches hurt it.
- **We raised the batch size. Throughput went up, but users are angrier. Why?** Throughput per GPU went up, but each user waits longer in the queue and each token comes slower. The number to watch is goodput: the share of requests that meet the latency target. Cap concurrency at the point where p95 starts to bend.
- **How big is the KV cache?** Per token it's 2 for K and V, times layers, times KV heads, times head size, times bytes. Mistral-7B in FP16 with grouped-query attention: 2 × 32 × 8 × 128 × 2 bytes, about 128 KB per token. A 4,000-token conversation is about half a gigabyte. That's why memory, not compute, usually limits how many users fit.
- **Is speculative decoding lossy?** No. The big model checks every drafted token, so the output distribution is the same. The speedup depends on how often drafts are accepted. It helps most at low load. At high load the GPU is already busy, and it helps less on unpredictable outputs like tool-call arguments.
- **Would you use tensor parallelism to make a 7B faster?** Usually no. A 7B fits on one GPU, and splitting it adds communication overhead. For more users I add replicas. Tensor parallelism is for models that don't fit, or when a big model's per-token latency is too high.
- **Semantic caching sounds like free speed. What's the risk?** Serving an answer to a slightly different question, or an old answer after the documents changed. I key the cache by tenant, permissions, and index version, use a strict similarity threshold, and don't cache time-sensitive or personal answers.
- **How do you autoscale on Kubernetes?** Custom metrics, waiting requests and KV-cache usage, through KEDA or HPA. Cooldowns of a few minutes, minimum replicas of at least one on the interactive path, and readiness probes that wait until the weights are loaded.
- **Self-host or use an API?** Data sensitivity, volume, latency control, and quality on our own evals. For Lam, customer recipe data and process IP push toward private endpoints or self-hosted models. A gateway lets you route per use case and swap providers without rewriting the app.
- **Have you run vLLM in production?** Honest answer: not at fleet scale. Bilbo ran on Ollama on one machine. I know how vLLM works inside and how I'd migrate, and I'd prove each change with a load test before claiming a speedup.

*Study: learning/50 §2, §3, §4, §5, §6, §9, §12, §14 · Q9, Q10, Q11, Q13, Q14, Q22, Q24, Q25 · learning/49 Q14, Q15*

### A2 · What is hallucination, and what are 3 concrete techniques to reduce it before launch?

*Tag: You found · top priority*

**What he's testing:** Four things, the same ones the question lists: you understand hallucination as a failure mode, you can prompt for abstention and grounded answers, you use RAG as a factuality tool (and know its limits), and you measure faithfulness and bad-answer rate.

**Say it like this**

A hallucination is when the model gives a fluent, confident answer that isn't supported by its sources, or just isn't true. Two kinds matter. It contradicts the documents it was given, or it adds facts that aren't in them. It happens because the model is trained to produce likely text, not checked text. When it doesn't know, it tends to guess instead of saying so. In technical documentation that's the dangerous case. A wrong pressure limit said confidently is worse than no answer.

Before launch, I'd use three techniques together.

One. Ground it with RAG. Retrieve the authoritative passages with hybrid search, so exact part numbers and error codes are found. Rerank, and give the model only the few chunks that matter, with version and date in the metadata. RAG reduces hallucination but doesn't remove it. The model can still ignore the context or misread a table. So it's one layer, not the whole answer.

Two. Design the prompt and the output for grounding and abstention. The instruction is: answer only from the provided context, cite the passage for each claim, and say 'I don't have enough information' when the context doesn't cover it. I use a structured output with the answer, the citations, and an insufficient-evidence flag, at low temperature. Before generation, a relevance gate. If retrieval scores are weak, the system refuses or asks a clarifying question instead of guessing. After generation, a check in code. Every cited ID must exist, and every number in the answer must appear in the cited passage.

Three. Measure it, because you can't reduce what you don't measure. A golden set of real engineer questions with the correct answer and source, including questions that should not be answered. I track faithfulness, meaning every claim is supported by the retrieved context. Citation accuracy. Abstention correctness. And the bad-answer rate, meaning confident and wrong. That's the number I'd put a launch gate on. It runs in CI on every prompt, model, or index change, and after launch I keep scoring a sample of production traces.

At Bilbo I used this pattern. Answers had to cite the source passage, the output was validated against a schema, and the system refused when it didn't have good evidence instead of guessing.

```text
SYSTEM PROMPT (sketch)
You answer questions about equipment documentation.
Use ONLY the context passages below. Each passage has an ID like [P3].
1. Every factual claim must cite a passage ID.
2. If the context does not contain the answer, set "insufficient_evidence": true
   and say what is missing. Do not guess.
3. Copy numbers, units, and part numbers exactly as written.
Return JSON: {"answer": str, "citations": [str], "insufficient_evidence": bool}
```

```python
# Post-generation check in code
from re import sub, findall

def verify(answer: dict, passages: dict[str, str]) -> list[str]:
    problems = [f"unknown citation {p}" for p in answer["citations"] if p not in passages]
    cited = " ".join(passages.get(p, "") for p in answer["citations"])
    text = sub(r"\[P\d+\]", "", answer["answer"])
    for num in findall(r"\d+(?:\.\d+)?", text):
        if num not in cited:
            problems.append(f"number {num} not in cited passages")
    return problems   # not empty: block, retry once, or route to a human
```

**Follow-ups he may push on**

- **RAG is in place and it still hallucinates. Debug it.** First split retrieval from generation. Was the right passage in the top-k? If not, it's retrieval: chunking, embeddings, hybrid weights, query rewriting. If the passage was there and the answer is still wrong, it's generation: the prompt, too much context so the key chunk gets lost in the middle, conflicting documents, or the model. Fix the stage that failed.
- **How do you measure hallucination in production without labels?** An LLM judge scores groundedness on a sample of traces against the retrieved context. Plus claim-level checks, citation coverage, and user signals like thumbs-down, edits, and regenerations. I recalibrate the judge against human labels regularly.
- **Why should I trust LLM-as-judge?** Only after calibrating it. Experts label a set, I measure how often the judge agrees, I use a clear rubric, a different model family from the generator, and I swap answer order to catch position bias.
- **Refusals make it less helpful. How do you balance?** I track both, false refusals and bad answers, and tune the relevance threshold on that tradeoff. For safety-critical specs I lean toward refusing and routing to an expert. A clarifying question often beats both.
- **Two documents disagree, an old spec and a new one.** Version and effective-date metadata. Prefer the current one by default. If both are relevant, show both with dates and flag the conflict instead of blending them into one answer.
- **Doesn't temperature zero fix it?** It removes randomness, not missing knowledge. A deterministic model can be deterministically wrong.
- **Would you fine-tune to fix hallucination?** Not first. Fine-tuning is good for format and behavior, and a poor way to add facts. It can even make the model more confident. Knowledge goes in retrieval. Fine-tuning can help it learn to cite and to abstain.

*Study: learning/48 §4.3 Q27 · learning/49 §4.2, §4.3, §7.1, §7.2 · Q3, Q11, Q16, Q17 · learning/30 §4, Q2 · learning/14 §6 · projects/P07 §9*

### A3 · Explain the tradeoffs between embedding techniques for a domain-specific technical corpus

*Tag: You found · top priority*

**What he's testing:** Whether you understand that technical text breaks general embeddings (jargon, IDs, numbers, tables), and whether you'd decide with a benchmark instead of a favorite model.

**Say it like this**

For a technical corpus the big question is how well the embedding handles jargon, exact identifiers, and structure. It's a tradeoff between quality, cost, latency, and maintenance. I see five options.

Sparse lexical, BM25 or learned sparse like SPLADE. Great at exact matches: part numbers, error codes, chemical names, acronyms. Fast, and needs no training. But it misses paraphrases.

General-purpose dense models, like MiniLM, BGE, E5, or an API embedding. They capture meaning out of the box. The risk is that their idea of similarity comes from general language. In a general model, 'recipe' sits near cooking. Here it's a process recipe. Bigger models are more accurate but cost more memory and latency. Small ones also cut off long chunks. MiniLM stops around 256 tokens.

Domain-adapted dense. Fine-tune a bi-encoder with contrastive learning on in-domain question and passage pairs. If there are no labels, generate synthetic questions from the documents with an LLM and mine hard negatives with BM25. That's usually the biggest quality gain on jargon. You pay in data work, training, and re-embedding the whole corpus every time the model changes.

Late interaction, like ColBERT. It keeps a vector per token, so it matches exact technical terms better while staying semantic. The cost is a much bigger index.

And hosted API versus self-hosted. APIs are strong and easy. But the data leaves your environment, you pay per token, and when the vendor retires a model you re-embed everything. For Lam's process IP, I'd expect self-hosted or private endpoints.

What I'd actually do: hybrid. BM25 plus a strong dense model, fused with reciprocal rank fusion, then a cross-encoder rerank. I'd benchmark candidate models on a golden set of real engineer questions, recall at 10 and MRR, plus latency and memory. I'd only fine-tune if the gap is real. And I'd store the embedding model version with the index, so a switch is a controlled migration.

At Bilbo I used all-MiniLM, 384 dimensions, because it ran fast on a local machine, with BM25 alongside it. Adding BM25, fusion, and the cross-encoder gave about 28% better top-five relevance than dense alone. That tells you a general dense model misses domain terms. Fine-tuning MiniLM on the domain was the planned next step. I didn't get to do it there.

| Option | Strength | Weakness | Use when |
|---|---|---|---|
| BM25 / SPLADE | Exact IDs, codes, acronyms; no training | Misses paraphrase | Always, as half of hybrid |
| General dense (MiniLM, BGE, E5) | Meaning out of the box | General-language similarity; truncation | Baseline, quick start |
| Fine-tuned dense | Best on jargon | Data work, retraining, full re-embed | Golden set shows a real gap |
| Late interaction (ColBERT) | Precise term match plus meaning | Much bigger index, more complexity | Precision-critical search |
| API embeddings | Quality, no infra | Data leaves, cost, forced re-embeds | Non-sensitive data |

**Follow-ups he may push on**

- **There are no labels. How do you fine-tune?** Generate synthetic questions per chunk with an LLM, filter them with a cross-encoder, mine hard negatives with BM25, and train with in-batch negatives (MultipleNegativesRankingLoss). Later, add real queries and clicks from the logs.
- **How do you switch embedding models safely?** Build a second index with the new model, compare on the golden set, run shadow traffic, then flip an alias. Never mix two models' vectors in one index.
- **Cosine, dot product, or L2?** Whatever the model was trained with. With normalized vectors, cosine and dot product give the same ranking.
- **What about tables and numbers?** Embeddings are weak on numbers. Keep tables structured, use metadata filters or text-to-SQL for numeric lookups, and BM25 for exact values.
- **How many dimensions?** Do the memory math. 10 million chunks × 1024 dimensions × 4 bytes is about 41 GB. At 384 dimensions, about 15 GB. Int8 quantization cuts that by four with a small recall loss. Matryoshka models let you truncate dimensions.
- **Diagrams and schematics?** Text embeddings miss them. Caption figures with a vision model and keep the caption next to the text, or use page-image retrieval like ColPali for diagram-heavy manuals.
- **Should we use a science-pretrained model like SciBERT?** Domain pretraining helps vocabulary. But if it wasn't trained for retrieval, it often loses to a retrieval-tuned general model. Test both on the golden set.
- **How does chunking interact with this?** A better chunk often beats a better model. Keep sections and tables whole, add the heading path to each chunk, and retrieve small chunks but give the model the parent section.

*Study: learning/18 §3, §4, §6 · learning/06 §3, §7, Q9 · learning/05 §2b, §2c, §3, §4 · projects/P07 §2.2*

## B · More Lam AI-engineer questions (Dataford)

Design questions on RAG, evaluation, and agents. B1 is your best whiteboard answer, so practice drawing it.

### B1 · Design a RAG pipeline for engineers querying internal manufacturing specs

*Tag: Dataford · Lam AI Eng*

**What he's testing:** Your centerpiece system design. Clarify first, then five clear parts, then how you'd pilot and measure.

```text
Sources (specs, SOPs, manuals, service logs)
  -> layout-aware parsing (headings, tables, figures, OCR)
  -> structure-based chunks + heading path + metadata
     (doc id, version, effective date, tool / product line, access labels)
  -> BM25 index + vector index        (incremental re-index on change)

Query -> user permissions -> pre-filter -> hybrid search -> RRF -> cross-encoder top 20-50
      -> numeric question? -> structured table / text-to-SQL path
      -> LLM: answer only from context, cite, abstain -> citation + number check -> answer

Around it: golden-set evals in CI, tracing per stage, cache, model router
```

**Say it like this**

I'd start by clarifying. Who are the users, process or field engineers? What documents: specs, SOPs, manuals, service logs, mostly PDFs with tables and diagrams? How many, how often do they change, who is allowed to see what, and what latency and accuracy do we need? Then I'd design it in five parts.

Ingestion. Connectors to where the documents live. Layout-aware parsing, so headings, tables, and figures survive. OCR for scans. Tables kept as structured data. Chunks follow the document structure, a section or a table, with the heading path added to each chunk. Every chunk carries metadata: doc ID, version, effective date, tool or product line, and access labels. Updates are incremental, so a changed spec is searchable in minutes, not overnight.

Retrieval. The user's permissions become a filter applied before ranking. Never leave that to the prompt. Then hybrid search, BM25 for part numbers and codes plus dense vectors for meaning, fused with RRF. A cross-encoder reranks the top 20 to 50. If the question is numeric, like a limit or a setpoint, I route it to the structured table or a text-to-SQL path instead of embeddings.

Generation. The top few chunks go in. The model answers only from them, cites each claim, and abstains when the evidence is weak. A check in code verifies that the citations exist and the numbers match the source.

Evaluation and safety. A golden set built with engineers. Retrieval recall, faithfulness, citation accuracy, abstention, and a bad-answer-rate launch gate. Plus prompt-injection tests, since documents can contain text that looks like instructions.

Operations. Tracing on every stage, caching, model routing, and a latency budget of a few hundred milliseconds for retrieval and rerank, and one to three seconds for generation.

I'd pilot with one product line and a small group of engineers. I'd measure time to answer and how often they still had to escalate, then expand. It's basically Bilbo scaled up with permissions and versioning.

**Follow-ups he may push on**

- **Permissions change. How fast does that apply?** Access labels live in the index metadata and are checked per query against the user's live groups, so a group change applies right away. Relabeling a document is a metadata update, not a re-embed.
- **Stale documents?** Effective dates and 'superseded by' links. Superseded versions are filtered out by default. Freshness monitors on every connector.
- **Multi-hop questions, like which chambers use part X and what failures were reported?** That's where a knowledge graph helps: entities like tool, chamber, part, and failure mode, and the relations between them, combined with vector retrieval. I'd add it when those questions show up in the logs, not on day one.
- **How do you show business value?** Time to answer, escalation rate to experts, first-time-fix rate for field issues, and adoption, compared against a baseline group.
- **Prompt injection inside a document?** Retrieved text is data, never instructions. Clear system prompt hierarchy, flag instruction-like text at ingestion, no side-effect tools in the answer path, and red-team tests in CI.

*Study: learning/52 §6, §7, §8, §17 · learning/05 §2a to §2i, §9, Q1, Q8 · learning/48 §5.2 · learning/26 §2c · projects/P07*

### B2 · LLM evaluation strategies, especially for hallucination in technical documentation

*Tag: Dataford · Lam AI Eng*

**Say it like this**

I evaluate in layers, because a single score hides where it broke.

Retrieval: did the right passage come back? Recall at k and MRR, on a golden set of real questions with their source passages.

Generation: is the answer supported and correct? Faithfulness to the retrieved context, correctness against a reference answer, and citation precision. For technical docs, numeric accuracy as its own check, because a wrong unit or digit is the failure that matters most.

Behavior: does it refuse when it should, and answer when it can? I put unanswerable and out-of-scope questions in the set on purpose.

System: latency, cost per answer, and refusal rate.

For scoring, deterministic checks wherever possible: numbers, units, citation IDs, format. An LLM judge with a rubric for faithfulness, calibrated against subject-matter experts on a sample. And human review on the high-risk slice.

Two things people skip. Statistics: with 60 questions, a 4% change can be noise, so I report confidence intervals and grow the set. And segmentation: I score by document type and product line, because the average can hide one area that's badly wrong.

This runs in CI on every prompt, model, or index change, with a regression gate. After launch, sampled production traces get scored by the judge, and thumbs-down and escalations go into a review queue that feeds new golden examples.

**Follow-ups he may push on**

- **Reference-based or reference-free?** Reference-based, comparing to a gold answer, for the golden set. Reference-free, groundedness against the retrieved context, for production, where there's no gold answer.
- **Offline evals pass, but users say it got worse.** Production questions drifted away from the golden set, or one segment broke. Sample recent traces, label them, add them to the set, and check that slice.
- **Your eval set has 60 examples and the score moved 4%. Is that real?** Probably not. Bootstrap a confidence interval and compare per example before believing it.
- **How do you evaluate a multi-turn conversation?** Score the outcome of the whole conversation, plus turn-level checks, and test that context carries over correctly between turns.

*Study: learning/49 §4.1 to §4.6, §9.2, Q2, Q3, Q4, Q9, Q10, Q22, Q26 · learning/52 §12 · learning/14 §4, §6, §7*

### B3 · Develop a multi-agent system for multi-step CAD automation

*Tag: Dataford · Lam AI Eng*

**What he's testing:** Can you design agents for actions that have consequences? Be honest that you haven't done CAD, then show the pattern.

```text
Request -> Planner (breaks into steps)
            |-> Rules agent ..... RAG over design rules and standards
            |-> Geometry agent .. edits through the CAD API (typed tools, MCP)
            |-> Validator ....... deterministic checks: dimensions, tolerances,
                                  interference, design rules
          errors -> structured feedback -> Geometry agent (max N attempts)
          -> new model revision (rollback possible) -> human approval -> commit

State graph (LangGraph) with checkpoints · budgets on steps, tokens, time · full trace
```

**Say it like this**

I'll be honest that I haven't automated CAD myself. But the pattern is the same as the agent systems I've built. The key thing about CAD is that the actions are consequential and checkable. So I'd use agents only where judgment is needed, and keep everything checkable in code.

A planner takes the request and breaks it into steps. Specialist workers handle each step. One retrieves design rules and standards with RAG. One generates or edits geometry through the CAD tool's API or a script. And a validator runs deterministic checks: dimensions, tolerances, interference, design rules. Validator errors go back to the generator as structured feedback, in a bounded loop with a maximum number of attempts.

All of it runs as a state graph, like LangGraph, with checkpoints. The CAD API is exposed as typed tools, and MCP is a good fit for that. Every step creates a new model revision, so we can roll back. Operations are idempotent. And a human approves before anything is committed or released.

For failures: tool errors come back as structured messages. There are timeouts and budgets on steps and tokens, so it can't loop forever. When it's stuck, it escalates to an engineer with the trace.

I'd measure it on a benchmark of real tasks: completion rate, iterations needed, and how much the engineer had to edit afterwards.

This is the lesson from my AI Cargo agent. A reflect step kept re-running tools that had already succeeded. The fix was to constrain it in code. In CAD that matters even more, because re-running an edit changes the part.

**Follow-ups he may push on**

- **Why multi-agent and not one agent?** Only if the roles need different tools, context, or models. If the steps are fixed, a deterministic pipeline with one LLM step is cheaper and easier to debug.
- **Two agents edit the same model at once?** One writer per model, or optimistic concurrency with revision numbers, so a conflicting write retries on the latest revision. See coding answer D5.
- **How do you evaluate an agent?** Task success, plus the trajectory: right tools, right order, number of steps, cost, and any rule violations, scored from the traces.
- **How do you stop runaway cost?** Budgets per task on steps, tokens, and time. Cheap models for routine steps. When the budget is hit, it stops and escalates.

*Study: learning/10 §4, §5.2, §7, §10.1, §10.2, Q1, Q4, Q6 · learning/27 §6, §9.3, §10 · learning/52 §2, §4 · learning/49 Q6, Q12, Q21 · projects/P15*

## G · Likely from Jatin (director-level judgment)

A director building a team asks about value, judgment, data safety, and how you'd start. These are short on purpose. Say them in 60 to 90 seconds.

### G1 · Where can GenAI create the most value at Lam?

I'd think about it in three places, close to what you've written about.

First, knowledge. Lam has decades of specs, service history, and troubleshooting notes. A grounded assistant with citations helps an engineer find what's already known in minutes instead of hours. That's the most proven pattern, and it's measurable: time to answer and first-time-fix rate.

Second, process recipe development. GenAI won't replace the physics or the experiments. But it can shorten the loop: pull up similar past recipes and experiments, summarize what happened, and propose next experiments alongside classical optimization, with the engineer deciding. That's augmenting judgment, not replacing it. Lam's own Nature paper found that engineers first and the algorithm last cut the cost to reach the target by about half.

Third, equipment intelligence. Sensor data is mostly a classical ML problem, anomaly detection and drift. The LLM layer is the interface on top: explain an anomaly in plain language, and link it to similar past cases and their fixes.

In all three, the LLM is the interface and reasoning layer on top of good data and classical models, with a human making the call.

*Study: 'Who you're talking to' above · learning/52 §1, §14*

### G2 · When would you NOT use an LLM?

When a rule, a statistic, or a classical model solves it reliably. When the answer has to be exactly right every time, like a control decision or a safety limit. When it's a real-time control loop where latency and determinism matter. And when there's no way to evaluate it yet.

My Jet2 event detection is the example. We started with moving averages, z-scores, and rules, and the model came last, on features we already trusted. And in AI Cargo, the compliance rules had a veto over the LLM.

- **So why GenAI at all?** For unstructured knowledge, language as the interface, and reasoning across sources. Things rules can't do well. Used for the right part of the problem, it's very strong.

*Study: learning/49 Q25 · Jet2 conflict story above*

### G3 · How would you build an assistant that reduces time-to-solution in process recipe development?

**What he's testing:** This is literally the first problem he named in his post. Clarify, keep the physics with classical methods, keep the engineer in charge, and measure iterations saved.

I'd start by sitting with process engineers to see where the time actually goes. It might be finding past experiments, not designing new ones.

Data: past recipes with their parameters, experiment results and metrology, and engineer notes and reports, all linked by experiment ID. Structured parameters stay in tables. Notes and reports go into a retrieval index.

The assistant: an engineer describes the goal, say a target etch profile on a new material. It retrieves similar past experiments by both parameters and text. It summarizes what worked and what didn't, with citations. And it suggests the next experiments. The suggestions come from a classical optimizer, like Bayesian optimization over the parameter space, and the LLM explains them. The engineer approves, the results come back, and the loop learns.

Guardrails: never suggest parameters outside the tool's safe operating window, enforced in code. Every suggestion links to its evidence. Customer data stays isolated.

Success metric: fewer experiment iterations to reach the target, and time to a qualified recipe, compared to a baseline.

This matches Lam's Nature paper: engineers are best early, the algorithm is more efficient near the tight target. And I'd be upfront that I'm not a process engineer, so the first month is learning the domain from the people who are.

- **Engineers don't trust it. Now what?** Start as a search and summary tool that saves them time, with citations, before it suggests anything. Trust comes from being right and showing evidence. The Nature paper calls out this culture challenge too.
- **Very little data per new process?** That's why the optimizer has to be sample-efficient, and why similar past processes and the engineer's starting point matter so much.

*Study: Nature paper summary above · learning/52 §8 · learning/05 §8*

### G4 · How do you handle sensitive data and IP?

I treat it as the first design constraint, not an add-on. At Bilbo the data couldn't leave the environment, so the whole stack ran locally, the model included.

At Lam I'd expect three kinds of sensitive data: Lam's own IP, customer data like recipes and tool logs, and export-controlled technical data. So models run on private endpoints or self-hosted inside Lam's network. Access control is applied at retrieval, so the user's permissions filter what can be retrieved before the model sees anything. Customer data is isolated per customer. Nothing goes into training or fine-tuning without an explicit agreement.

Logs help debugging, but they can also leak. So traces are access-controlled, sensitive fields are redacted, and there's an audit trail of who asked what and what was returned.

- **Can't the prompt just tell the model not to reveal restricted documents?** No. If the model can see it, a clever prompt can get it out. Restricted content must never reach the context in the first place.

*Study: learning/49 §7.4 · learning/52 §17 · projects/P07 (local-only design)*

### G5 · How do you evaluate and debug an agent? How would you detect errors in its traces?

I trace every step under one trace ID: the input, the tool called and its arguments, the output, tokens, latency, and the final result.

Then I evaluate at two levels. The outcome: did it complete the task correctly? And the trajectory: did it call the right tools in a sensible order, how many steps, how much it cost, and did it break any rule?

For error detection I group failures into types: wrong tool, bad arguments, repeated calls, ignored tool errors, stopping early, and made-up results. Some are caught by deterministic checks on the trace, like the same tool called twice with the same arguments. The fuzzy ones get an LLM judge that reads the trace with a rubric, calibrated against human labels.

That's exactly how I found PHANTOM TOOLS in AI Cargo. Every step was logged, and the trace showed the reflect step re-running tools that had already succeeded. The dashboard looked healthy. Only the trace showed it was wrong.

- **How do you build a test set for an agent?** Real tasks with a known good end state, plus hard ones: tools that fail, ambiguous requests, missing data. Replay them on every change and compare trajectories, not just final answers.

*Study: learning/49 §3, Q12, Q21, Q23 · learning/10 §9.4, §10.4 · projects/P15*

### G6 · Build or buy? Hosted models or open-source?

I decide per use case, with our own eval set as the judge. Four questions. How sensitive is the data? What's the volume and cost at scale? Do we need control over latency or the model version? And do we have the people to run it?

My default is to start with a strong managed model on a private endpoint, to learn fast and build the eval set, behind a gateway so we're not locked in. When a use case has high volume, strict privacy, or a narrow task, I move it to a smaller self-hosted or fine-tuned model, but only if it matches quality on the evals.

At Jet2 that's what the 70B to 7B distillation was. The big model set the quality bar, and the small one was what we could run at contact-center volume, with 94% recall.

*Study: learning/50 §13.3 · learning/48 Q24 · learning/49 §6*

### G7 · Fine-tuning versus RAG versus prompting?

Prompting first, because it's cheap and fast to iterate.

RAG when the model needs knowledge it doesn't have or that changes, like specs and service history. It also gives citations, and you update it by re-indexing, not retraining.

Fine-tuning when the problem is behavior, not knowledge: a consistent format, a domain style, a narrow classification, or making a smaller model as good as a big one on one task. Often it's both: RAG for facts, and a light fine-tune like LoRA for format.

I'd only fine-tune with a stable task and a good eval set, because otherwise you can't tell whether it helped.

*Study: learning/48 Q8 · learning/52 §11*

### G8 · Engineers want to ask questions over equipment and yield data. How?

A lot of those questions are numeric, and embeddings are bad at numbers. So text-to-SQL is often the right tool.

Steps. First, a curated semantic layer: clean views with clear names and column descriptions, not the raw schema. Then schema linking: retrieve only the relevant tables, columns, and a few example queries for this question. The model writes SQL. Before it runs, checks in code: it must parse, be read-only, use allowed tables only, and have a row limit and a timeout. If it errors, I feed the error back once or twice.

The answer shows the result, the SQL, and a plain explanation, so an engineer can check it. Permissions are enforced by the database user, not the prompt. And I evaluate by execution accuracy, comparing results against a golden set of question and SQL pairs.

- **Ambiguous questions like 'yield last month'?** Ask a clarifying question, or state the assumption in the answer: which fab, which product, which definition of yield.

*Study: learning/52 §6 · learning/49 §7.2*

### G9 · GraphRAG or vector RAG?

Vector RAG is great for 'find the passage that answers this.' It struggles with questions that connect many facts, like which tools use this part and what failures were reported on them, or summarize all issues across these chambers.

A knowledge graph stores the entities, tools, chambers, parts, failure modes, recipes, and the relations between them, so you can follow the links instead of hoping one chunk has it all. The cost is building and maintaining the graph: entity extraction, matching names that mean the same thing, and keeping it fresh.

So I'd start with hybrid vector search, look at which questions fail in the logs, and add a graph for those multi-hop patterns. Lam already uses knowledge graphs in its enterprise AI work, so part of the data model may already exist.

*Study: learning/52 §7, §17 · learning/05 §8.2*

### G10 · How do you defend against prompt injection?

Retrieved documents, emails, and tool outputs are data, not instructions. That's the core rule. Untrusted text is wrapped and labeled as content, under a clear system prompt hierarchy.

The bigger protection is limiting what the model can do. Least-privilege tools, no side-effect tools in a plain Q&A path, and human approval for anything consequential. At ingestion I flag text that looks like instructions. Outputs are validated against a schema. And a red-team set of injection attempts stays in the eval suite.

You can't make a model fully immune, so you design so that a successful injection can't do much.

*Study: learning/49 §7.3 · learning/30 §6.2, Q5*

### G11 · What would you do in your first 90 days?

First month: learn. Sit with the users, process and field engineers, and see where their time actually goes. Learn the platform, the data, and how things are evaluated today. Ship one small fix to learn the deploy path.

Second month: own one measurable improvement. For example an evaluation set for a key use case, a tracing gap, or a retrieval fix, with a before-and-after number.

Third month: own a feature end to end, from design to production, and bring a short proposal for what to do next, based on what I've seen in the data and the logs.

*Study: learning/53 §7.6*

### G12 · How do you keep up with GenAI without chasing hype?

I read a few good sources, but mostly I learn by building small things. When something new comes out, I try it on a real problem and compare it to what I have, on my own eval set. If it doesn't beat the current setup, it doesn't go in.

New models and frameworks come every month. A good eval set is what lets you switch quickly when something is actually better. Day to day, I use Cursor heavily for coding, with rules and skills, and I treat its output like a junior engineer's PR. I read it and I test it.

*Study: learning/52 §1, §16*

### G13 · Why should I hire you for a team I'm building from scratch?

Three reasons. I've built both halves of what this team needs, retrieval systems and agents, end to end, not just one part of them.

I build for production. Evaluation, tracing, and guardrails come first, and my Jet2 model ran for eighteen months.

And I'm used to starting from zero. I built AI Cargo alone in six weeks with no spec. On a new team someone has to set up the foundations and ship the first use case, and that's the work I like most.

*Study: learning/53 §7.5 · 00_RESUME_MASTER_MAP §6*

## C · System design: monitoring, drift, scale (Dataford)

This is where 'equipment intelligence' shows up. Tie it back to Jet2.

### C1 · Design real-time monitoring for an AI model in a semiconductor fab

I'd monitor four layers. The input data: sensor values, missing data, ranges, schema, and drift against a baseline. The model's outputs: the prediction distribution and confidence. The model's accuracy: when labels arrive later, like metrology results, I join them back and track the error. And the service itself: latency, errors, uptime.

The model service logs every prediction with its input features, model version, tool and chamber ID, and timestamp to a stream, like Kafka or Event Hubs. A stream processor computes windowed statistics per tool and per chamber, because each chamber behaves differently. Those go to a time-series store for dashboards and alerts.

Alerts need thresholds and cooldowns, or people stop reading them. When the model looks unhealthy, the system falls back to the rule-based path. New model versions run in shadow mode before they take over.

At Jet2 I tuned an event-detection system to about 96% specificity, because in alerting, false alarms kill adoption faster than a missed event.

- **Labels arrive days later. What do you do until then?** Input drift and output drift are the early warning. Delayed labels confirm it.
- **Which drift metric?** PSI or a KS test per feature, per tool, with thresholds set from normal historical variation.

*Study: learning/26 §1 Step 6, observability section, Q6 · learning/49 §8 · projects/P03*

### C2 · How would you detect and handle model drift with time-series sensor data?

With sensors there are a few kinds of drift. The data shifts. The relationship between inputs and outcomes changes. Or the sensor itself degrades or gets recalibrated. In a fab, a lot of drift has a known cause: preventive maintenance, a part swap, chamber seasoning, a recipe change. So I'd log those events and use them to segment the data and reset baselines, instead of treating every shift as a mystery.

Detection: distribution tests per feature per tool, change-point methods like CUSUM on the model's residuals, and model error once labels arrive.

Response: retrain on a recent sliding window with walk-forward validation, normalize per tool, and use scheduled plus triggered retraining behind validation gates. Champion versus challenger before swapping, and a human sign-off. One trap: don't retrain on a fault period, or the model learns the fault as normal.

At Jet2 my forecasting model ran 18 months with under 2% error variation, with walk-forward validation and scheduled retraining. Same discipline.

- **Retrain on a schedule or on a trigger?** Both. A schedule for gradual drift, a trigger for sudden shifts, and the same validation gate either way.

*Study: learning/26 §1 Step 6, §2g · learning/49 §8.3 · projects/P04, P03*

### C3 · Scale an inference service for peak loads during a design cycle

Design-cycle peaks are predictable, so I'd pre-scale before them and autoscale on top, on queue depth rather than CPU. I'd keep a warm pool, because GPU replicas take minutes to load a model.

Interactive and batch traffic are separated. Interactive gets priority. Batch work goes to a queue that returns a job ID. Per-team quotas and rate limits, so one team can't starve the others. Under heavy load, degrade gracefully: route to a smaller model, serve cached answers, or shorten outputs, instead of timing out.

Capacity comes from a load test: tokens per second per GPU at our real prompt lengths, at the latency target. That gives the replica count for the peak, plus headroom, and burst capacity in the cloud if on-prem is full.

*Study: learning/50 §12, §13, Q11, Q29 · learning/26 §1 Step 5*

## D · Coding (Dataford)

Unlikely in a 45-minute manager call, but possible as a quick 'how would you write this'. Say the approach and complexity first, then the code.

### D1 · Implement efficient vector search across a large dataset

Say the plan first. Normalize the vectors, then a matrix multiply gives cosine scores. argpartition finds the top k in linear time without a full sort. That's exact search, fine up to about a million vectors in memory.

If the data doesn't fit in memory, stream it in blocks and keep only the best k between blocks. Past that scale, I'd switch to approximate search, HNSW or IVF-PQ with FAISS, and tune recall against latency.

```python
import heapq; import numpy as np

def normalize(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def top_k(index, query, k=10):
    """index: (N, d) normalized vectors. Exact search, O(N*d)."""
    scores = index @ normalize(query)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]     # O(N), no full sort
    idx = idx[np.argsort(-scores[idx])]           # sort only the k winners
    return idx, scores[idx]

def top_k_streaming(blocks, query, k=10):
    """Data larger than memory. blocks yields (ids: np.array, vectors)."""
    q, best = normalize(query), []
    for ids, vecs in blocks:
        s = normalize(vecs) @ q
        j = np.argpartition(-s, min(k, len(s)) - 1)[:k]
        best = heapq.nlargest(k, best + list(zip(s[j], ids[j])))
    return best
```

- **Complexity?** O(N·d) to score, O(N) to partition, O(k log k) to sort the winners.
- **How do you add metadata filters?** Pre-filter when the filter is selective, so you never return zero results. Post-filter when it's broad, and over-fetch to compensate.
- **HNSW or IVF-PQ?** HNSW for high recall and low latency when memory allows. IVF-PQ when memory is the limit, accepting some recall loss. Tune efSearch or nprobe against a recall target.

*Study: learning/06 §3, §6, Q1, Q4, Q5, Q8 · learning/25 §6.1*

### D2 · Write a function to find anomalies in real-time log data

Say the plan first. Parse each line, bucket counts per key per minute, for example per tool and error code. Keep a rolling baseline per key, flag a point when its z-score is above a threshold, and alert with a cooldown.

Two details that show experience. Updates are O(1) with running sums. And anomalies are kept out of the baseline, so a spike doesn't hide the next spike. If the anomaly lasts long enough, treat it as the new normal.

This is the same signal layer I built at Jet2 for event detection, before Isolation Forest went on top.

```python
from collections import defaultdict, deque; import math

class RollingZ:
    """O(1) per update. One instance per key, e.g. (tool_id, error_code)."""
    def __init__(self, window=60, threshold=3.0, min_points=20, rebaseline_after=10):
        self.buf, self.window = deque(), window
        self.threshold, self.min_points = threshold, min_points
        self.rebaseline_after, self.streak = rebaseline_after, 0
        self.total = self.total_sq = 0.0

    def _push(self, x):
        self.buf.append(x); self.total += x; self.total_sq += x * x
        if len(self.buf) > self.window:
            old = self.buf.popleft(); self.total -= old; self.total_sq -= old * old

    def update(self, x: float) -> bool:
        n, anomaly = len(self.buf), False
        if n >= self.min_points:
            mean = self.total / n
            std = math.sqrt(max(self.total_sq / n - mean * mean, 1e-12))
            anomaly = abs(x - mean) / std > self.threshold
        self.streak = self.streak + 1 if anomaly else 0
        if not anomaly or self.streak >= self.rebaseline_after:
            self._push(x)   # spikes stay out of the baseline unless they become the new normal
        return anomaly

detectors = defaultdict(RollingZ)

def on_minute(key, count):          # called once per key per minute bucket
    if detectors[key].update(count):
        alert(key, count)            # with a cooldown, so one incident is one alert
```

- **Daily or weekly patterns?** Compare against the same hour on previous days, or remove seasonality first, so a normal morning peak doesn't fire.
- **Numerical stability of running sums?** For long streams or large values, use Welford's method for the variance.
- **Millions of keys?** Shard by key across workers, and expire keys that have been idle.
- **Several signals together?** That's where a multivariate model like Isolation Forest helps, on top of these features. That's what I did at Jet2.

*Study: learning/26 Q6 · learning/25 §2.4 · projects/P03*

### D3 · Use dynamic programming to optimize resource allocation

Frame it as 0/1 knapsack: jobs need some GPUs and have a value, and we want the most value within capacity. The state is the best value using the first i jobs with capacity c. Each job is either skipped or taken.

Time and space are O(n × capacity). If I only need the value, a 1D array going backwards gets space down to O(capacity). I'd also mention that real schedulers with several resources use bin packing heuristics or an integer program, but DP is the right answer for one resource.

```python
def allocate(jobs, capacity):
    """jobs: list of (gpus_needed, value). Returns (best_value, chosen_indexes)."""
    n = len(jobs)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i, (need, value) in enumerate(jobs, 1):
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]                       # skip job i
            if need <= c:
                dp[i][c] = max(dp[i][c], dp[i - 1][c - need] + value)   # take it
    chosen, c = [], capacity
    for i in range(n, 0, -1):                             # walk back to find the choices
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(i - 1)
            c -= jobs[i - 1][0]
    return dp[n][capacity], chosen[::-1]

# allocate([(2, 3), (3, 4), (4, 5), (5, 6)], 5)  ->  (7, [0, 1])
```

*Study: learning/25 §7 patterns*

### D4 · Refactor inefficient Python code for performance

My order: measure first with cProfile, so I fix the real hotspot. Then fix the data structure before micro-optimizing. Keep the tests green, and state the complexity before and after.

Common fixes: list lookups to sets or dicts, string += in a loop to join, list.pop(0) to deque, repeated work to a cache, pandas apply to vectorized operations, and I/O-bound loops to async or batching.

```python
# Before: O(n*m). Both 'in' checks scan a list.
def common_ids(a, b):
    out = []
    for x in a:
        if x in b and x not in out:
            out.append(x)
    return out

# After: O(n + m). Same output, same order.
def common_ids(a, b):
    b_set, seen, out = set(b), set(), []
    for x in a:
        if x in b_set and x not in seen:
            seen.add(x)
            out.append(x)
    return out
```

*Study: learning/25 §2.1 to §2.4, §6.1, §6.2*

### D5 · Handle concurrency when multiple agents update a shared data store

Two cases. Inside one process with asyncio, a lock per key around any read-modify-write that has an await in the middle. Without an await between the read and the write, asyncio is already safe, because only one coroutine runs at a time. The race shows up at the await.

Across processes or machines, optimistic concurrency. Each record has a version. A write only succeeds if the version hasn't changed. If it has, re-read and re-apply. Plus idempotency keys, so a retried tool call doesn't apply twice. That's the same failure I fixed in AI Cargo, where tools were re-run after they'd already succeeded.

```python
import asyncio
from collections import defaultdict

class SharedStore:
    """One process: a lock per key, so agents on different keys don't block each other."""
    def __init__(self, backend):
        self.backend, self.locks = backend, defaultdict(asyncio.Lock)

    async def update(self, key, mutate):
        async with self.locks[key]:
            current = await self.backend.get(key)    # another agent could run here
            await self.backend.put(key, mutate(current))

class VersionConflict(Exception):
    pass

def update_with_retry(store, key, mutate, retries=5):
    """Many processes: optimistic concurrency with a version number."""
    for _ in range(retries):
        record = store.get(key)                       # {"version": 7, "data": {...}}
        new_data = mutate(dict(record["data"]))
        if store.compare_and_set(key, record["version"], new_data):
            return new_data                            # version is now 8
    raise VersionConflict(key)
```

- **Why not a distributed lock?** It works, but a lock with a timeout can expire while the holder is still writing. Optimistic concurrency doesn't block and fails safely.
- **Threads instead of asyncio?** threading.Lock. The GIL doesn't make read-modify-write atomic.
- **Deadlocks?** Always take locks in the same order, and use timeouts.

*Study: learning/25 §5.5, §5.6, §5.7 · projects/P15 (idempotent tools)*

## E · Equipment-software questions (his background)

From public Lam SWE question titles. With his equipment background, E1 is the one most worth practicing.

### E1 · Design a state machine to safely shut down a plasma chamber on an unexpected pressure drop

**What he's testing:** Jatin wrote equipment software for years. He's checking whether you think about safety, fail-safe defaults, and confirmation, not just the happy path.

```text
Idle -> PumpDown -> Ready -> Processing -> Venting -> Idle
                         \          |
                          \         | pressure out of band, sensor fault,
                           \        | lost comms, watchdog timeout
                            v        v
                         SAFE_SHUTDOWN  (entered from any active state)
                           entry actions in a fixed order, each confirmed by a
                           sensor reading with a timeout; no confirmation -> alarm
                           + hardware interlock
                         exit only by operator acknowledgment -> Idle
```

First principle: software is not the only safety layer. There are hardware interlocks underneath, and the software state machine sits on top of them.

States like Idle, PumpDown, Ready, Processing, Venting, and a SafeShutdown state. Transitions only happen on defined events with guard conditions. A pressure drop outside the allowed band, from any active state, moves straight to SafeShutdown.

SafeShutdown runs its entry actions in a fixed order that's agreed with the process and safety engineers, for example turn off RF power, stop the gas flows, and move the valves to their safe positions. Each action is confirmed by a sensor reading with a timeout. If the confirmation doesn't come, it raises an alarm and relies on the hardware interlock.

Defaults are fail-safe. Lost communication or a watchdog timeout also goes to the safe state. Leaving SafeShutdown needs an operator acknowledgment, never an automatic reset. Every transition is logged with timestamps for root cause. And I'd test it in a simulator by injecting faults in every state.

It's the same idea as the deterministic veto in my agent work. The rule that protects safety can't be overridden by the smarter component.

- **How do you avoid false trips from a noisy sensor?** Debounce: the reading must be out of band for a short, defined time, or two sensors must agree. But the safety margin decides the filter, not convenience.
- **How do you test it?** A table of every state and event with the expected result, simulator fault injection, and tests for the transitions that must never happen.

*Study: state pattern, enums, guard conditions (learning/25 §3)*

### E2 · Architect a system that handles high-frequency data from many sensors

Keep the control path separate from the data path, so analytics can never slow down the tool.

Each sensor reader timestamps at the source and writes into a bounded ring buffer. An aggregator reads in batches and computes window summaries, like min, max, and mean per window, and keeps raw data around events. The backpressure policy is explicit. For analytics we can decimate or drop with a counter. For safety signals, never.

Data goes to a time-series store in batches, with sensor ID, unit, and calibration version on every record. And I'd monitor the pipeline itself: lag, dropped samples, and clock skew.

*Study: learning/26 batch vs streaming · learning/25 §2.4 (deque as ring buffer)*

### E3 · What are Python decorators, and when would you use one?

A decorator is a function that takes a function and returns a wrapped version of it. It's for cross-cutting concerns: timing, retries, caching, auth, and logging, without repeating that code everywhere.

A practical example is retry with exponential backoff for LLM calls. functools.wraps keeps the original name and docstring. And I only retry operations that are safe to repeat. Retrying a tool that already changed something is how you get duplicate actions.

```python
import functools, random, time

def retry(times=3, base_delay=0.5, exceptions=(TimeoutError, ConnectionError)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times - 1:
                        raise
                    time.sleep(base_delay * 2 ** attempt + random.uniform(0, 0.1))
        return wrapper
    return decorator

@retry(times=4)
def call_model(prompt: str) -> str:
    ...
```

*Study: learning/25 §3.6, §5.3*

## F · Behavioral (Dataford)

Extra behavioral prompts on top of your STAR scripts above. Same rule: one number and one lesson at the end.

### F1 · Explain a complex ML concept to a non-technical stakeholder

At Jet2 our event-detection system flagged unusual hours in the live chat data, and I had to explain Isolation Forest to the ops team. I didn't explain trees.

I said: imagine you keep splitting the chat data at random, by volume, by keywords, by topic. A normal hour looks like lots of other hours, so it takes many splits to separate it. A strange hour stands out, so it gets separated in just a few. The model counts how fast an hour gets isolated.

Then I always added what they actually needed: which signals were unusual for that time of day, and what to check first. They didn't need the algorithm. They needed to trust the alert and know what to do. The setup ran at about 96% specificity, so the alerts stayed credible and they kept using them.

*Study: learning/53 §6 · projects/P03*

### F2 · Your initial approach failed and you had to pivot

At Bilbo I started with dense vector search alone, which is where most people start. When I tested it on real protocol questions, it missed exact terms, specific medical terms and protocol names. Semantically similar isn't the same as exactly right.

I didn't keep tuning the same approach. I added BM25 keyword search alongside it, fused the two with reciprocal rank fusion, and put a cross-encoder rerank on top. On the labeled set that gave about 28% better top-five relevance than dense alone.

The lesson: test the default on real queries early, before building everything on top of it.

*Study: learning/53 §6 · projects/P07 §2*

### F3 · Technical debt versus a fast release cadence

I sort it into what can be cut and what can't. On AI Cargo I had six weeks, alone. I cut UI polish and some nice-to-have features. I didn't cut the things that are expensive to add later: the audit log, the deterministic rules, tests on the risk scoring, and tracing. And I wrote down what I skipped, so it was a list, not a surprise.

The other side is paying debt down once instead of many times. At ATCS, each client engagement used to build its data processing from scratch. I built a configuration-driven data-quality framework, with the rules in YAML, so onboarding a new client meant writing a config, not a pipeline. Onboarding went from two or three weeks to two or three days, and it ran on four or more engagements.

*Study: learning/53 §6 · projects/P05b §6*

### F4 · Mentoring someone, or improving a team process

I haven't had a formal mentoring role yet, so I'll give you the closest real examples.

The ATCS data-quality framework was a process change for the team. Because the rules lived in config, other people could add checks for a new client without writing new pipeline code. It was used on four or more engagements.

At Fibe I set up MLflow tracking and automated validation gates, so model runs were reproducible and anyone could see what changed between versions. That took the model cycle from about three days to under five hours.

On a new team I'd do the same: write down how things work, make the common path easy, and sit with people the first time they use it.

*Study: learning/53 §6, §7 · projects/P05b, Fibe notes*

---

## Speaking toolkit — for clear, confident delivery

Your English is good enough; the goal is clarity and calm, not a perfect accent. These techniques make simple English sound polished.

### Delivery techniques

**Slow down ~20%.** Rushing is the #1 clarity killer. Speak slower than feels natural.

**Short sentences.** One idea per sentence. Full stop. Next idea. Long sentences are where you trip.

**Pause, don't fill.** Replace "um / uh / like" with a short silent pause. Silence sounds confident; fillers don't.

**Signpost.** Use "So, the situation was…", "What I did was…", "The result was…". Structure makes simple words sound organized and buys you thinking time.

**Let STAR carry you.** The structure keeps you from rambling when you're nervous.

### In-the-moment tactics

**Buy time cleanly:** "That's a good question, let me think for a second." Totally normal.

**Ask to clarify:** "Just to make sure I answer the right thing, do you mean X or Y?" This shows communication, not weakness.

**Lost a word?** Describe it ("the part that checks the output") instead of freezing. Keep moving.

**Didn't catch it?** "Sorry, could you repeat that last part?" is completely fine and better than guessing.

**Energy carries.** Smile and sound interested. Warmth covers a lot of small grammar slips.

> **Daily fluency drill (20–30 min, every day until the call)**
>
> (1) Record yourself answering 2–3 questions from the scripts above on your phone. (2) Listen back once, note only two things: where you rushed, and where you used fillers. (3) Re-record the same answers fixing just those two. (4) Shadowing: watch 5 minutes of a clear English speaker (a tech talk), pause each sentence and repeat it matching their rhythm. This trains pace and intonation faster than anything else. Do a 10-minute warm-up of this on call day.

---

### Questions to ask Jatin (pick 2 or 3)

1. "When you joined, you wrote about process recipes, engineering judgment, and equipment intelligence. Which of those is the team going after first?"
2. "Since you're building the team now, what do you want in place in the first six months: the evaluation setup, observability, the platform, or a first use case in production?"
3. "Who are the first users, process engineers or field service? How closely would this role work with them?"
4. "How do you decide when a problem needs GenAI and when a classical model or a rule is the better answer?"
5. "How does your team work with the enterprise AI platform group? What do you build, and what do you reuse?"
6. "What would make someone in this role a clear success after a year?"

1 and 2 are the strongest. They show you read what he wrote and that you think like an early team member. General backups: "How mature is the GenAI platform today, and where are the biggest reliability or evaluation gaps?" and "How do you measure whether a GenAI feature is working before it ships?"

### How to close the call

End with genuine interest and a clear ask: "This has made me more excited, honestly. The work is a strong match and I'd really like to move forward. Is there anything about my background I can clarify, or anything you'd want me to prepare for the next round?"

This signals enthusiasm, invites them to raise any doubt now (so you can address it), and sounds confident without being pushy.

Same day, send Michelle a short thank-you note for Jatin, with one specific thing from the conversation in it.

> **One-page mental model for the whole round**
>
> You are not trying to prove you're the smartest person on the call. You are proving three things: you've genuinely built RAG and agents (your projects), you think clearly about tradeoffs and failures (STAR with a number, and knowing when not to use an LLM), and you actually want Lam (his post, the Nature paper, real questions). Say less, say it slowly, end every answer on evidence. That is a strong-hire signal in any language.
