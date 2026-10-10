# Lam Research — Hiring Manager call (45 min)

Next step after Michelle's recruiter screen: a 45-minute conversation with the hiring manager, ~9–10 days out. This is the round that decides whether you go to the full loop. It is a conversation, not a coding test: background, projects, motivation, how you think, team fit, and behavioral questions scored against Lam's core values. The deep technical Q&A bank lives in your companion canvas.

**Clear thinking beats fancy words** · **One story per competency, each with a number** · **Slow down, signpost, pause** · Be specifically excited about Lam

| 45 min | ~9–10 days | Make-or-break | STAR + fit | Very strong |
|---|---|---|---|---|
| With the hiring manager | To prepare | Gate to the full loop | How it's scored | Your match |

> **What the hiring manager is really deciding**
>
> Four things, all at once: (1) Can you actually do this job — is the depth in your projects real? (2) How do you think — your judgment and tradeoffs, not just what you built? (3) Will you fit and communicate on the team — can you explain a hard idea simply? (4) Do you genuinely want Lam, or is it just another application? Your English does not need to be perfect. What's scored is clarity of thinking and ownership. A simple, well-structured answer said slowly beats a fast, tangled one every time.

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
| Inclusion & diversity | Welcomes different perspectives | Working with clinicians / ops / exec stakeholders |
| Innovation & continuous improvement | New ideas, embraces change | +28% hybrid retrieval · EVAL HARNESS · 70→7 distillation |
| Mutual trust & respect | Treats others well in disagreement | CONFLICT story (how you handled it, not who won) |
| Open communication | Speaks candidly, listens, asks | Explaining citations to clinicians · exec dashboards · speaking up |
| Ownership & accountability | Takes responsibility, delivers | 18→7 live 18 months · PHANTOM TOOLS · 4h→18min |
| Teamwork | Collaborates to a shared goal | Cross-functional at Jet2 / Bilbo · DQ framework adopted by 4+ teams |

---

## How to approach the 45 minutes

| Phase | Roughly | What happens | Your job |
|---|---|---|---|
| Warm-up + intro | 5 min | Small talk, then 'tell me about yourself' | Warm, ~60s intro, land on why Lam |
| Background + projects | 15–20 min | Walk through Bilbo / AI Cargo, how and why | Explain simply, show judgment and tradeoffs |
| Behavioral | 10–15 min | 'Tell me about a time…' on Lam's values | STAR-L, one number each, under 2 min |
| Technical conversation | 5–10 min | 'How would you approach X' (not coding) | Reason out loud; it's fine to say tradeoffs |
| Your questions | 5 min | 'What questions do you have for me?' | Ask 3–4 sharp ones; close warmly |

> **The single habit that wins this round**
>
> End every project and behavioral answer with how you knew it worked (a number) and what you learned. Most candidates end on a feeling. You end on evidence. That is the difference between a 2 and a 4 on their sheet.

---

## Scripts to practice out loud (your voice, simple English)

Short sentences on purpose. Practice saying these until the rhythm is natural, then make them yours. Do not memorize word-for-word; memorize the shape and the numbers.

### 1 · Tell me about yourself (~60 seconds)

"Sure. I'm a data scientist and AI engineer with about five years of experience, and I just finished my master's in data science at Maryland in May. Most recently I was an AI engineer at Bilbo, where I built a healthcare AI assistant. It answers questions from medical documents and cites exactly where each answer came from, running on a private, self-hosted model because the data couldn't leave the environment. It made the search about 28% more accurate, with answers under five seconds."

"Before my master's, I was a data scientist at Jet2 in the UK for two years, working on forecasting and language models over call-center data, and earlier I did credit-risk and fraud work in fintech. On the side I build AI agents. I won first prize in an agentic-AI competition for a shipping-risk platform."

"What drew me to this role is that it's exactly the work I want to do: LLM applications, retrieval, and agents, with the reliability to trust them. And doing it for semiconductor engineering is genuinely interesting to me, because that's a domain with deep, high-stakes knowledge where this really matters."

### 2 · Why Lam, why semiconductors? (deeper than the recruiter version)

"A few reasons. First, the work lines up almost exactly with what I've built, RAG, agents, and the observability that makes them reliable. Second, I like where Lam sits: you make the equipment that basically the entire chip industry depends on, so there's real weight to it. And I read that Lam is investing seriously in internal GenAI, across R&D, manufacturing, and field service, which is a huge, high-value problem."

"The part that really fits me is that Lam treats agentic AI as something you have to govern carefully, with human oversight and logging. That's exactly how I build. So it feels less like a stretch and more like the same values at a much bigger scale."

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

### Conflict — fill this in with your real details before the call

Draft (grounded in your Bilbo decision — confirm the real people and what they actually said): "At Bilbo there was a push to just use a frontier API like GPT-4 for generation, which is a fair instinct, it's the easy high-quality option. I disagreed, because the data was PHI and couldn't leave our environment. Instead of just saying no, I first made sure I understood the real concern, which was answer quality. So I showed that with strong retrieval and reranking plus a local model, we could hit the quality we needed and stay compliant. We went with the local model, and compliance ended up shaping the whole architecture. What I do earlier now is bring the compliance constraint into the design on day one, so it isn't framed as me blocking a better option."

Conflict answers are scored on how you treated the other person, not on whether you won. Keep it respectful, state their view fairly, and it's fine, even better, to say where you partly conceded. Confirm the details actually happened; never invent a conflict.

### Teamwork / explaining to non-technical people

"A lot of my work is with people who aren't technical. At Bilbo I worked closely with clinical staff who needed to trust the assistant, so I made citations a first-class feature, every answer points to the exact source paragraph, and I walked them through how to check it. Earlier, in fintech, I built executive dashboards that turned model output into decisions leaders could act on. I lead with the outcome and the 'why can you trust this,' and keep the jargon out unless someone asks for it."

### Learning something fast

"For the agentic challenge I had about six weeks and had to get fluent fast in the agent stack, LangGraph, tool calling, multi-provider fallback. The way I learn fast is I build the smallest working version first, then break it on purpose to understand how it fails, which is actually how I found the phantom-tool bug. By the end I had a production-style system and a real understanding of where these agents break, not just how to make them run."

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

### Questions to ask the hiring manager

1. "What are the first problems this person would work on in the first few months?"
2. "How mature is the GenAI platform today, and where are the biggest reliability or evaluation gaps?"
3. "How does the team work with the process engineers whose knowledge you're encoding?"
4. "How do you measure whether a GenAI feature is actually working before it ships?"
5. "What does success look like for someone in this role after a year?"

### How to close the call

End with genuine interest and a clear ask: "This has made me more excited, honestly. The work is a strong match and I'd really like to move forward. Is there anything about my background I can clarify, or anything you'd want me to prepare for the next round?"

This signals enthusiasm, invites them to raise any doubt now (so you can address it), and sounds confident without being pushy.

---

## Your 9–10 day plan (study + speak every day)

Each day has a learning block and a speaking block. The speaking block is non-negotiable, it's the thing you're nervous about, so it gets daily reps. Keep it to ~60–90 min a day.

- [ ] Day 1 — Absorb this canvas end to end. Burn in the Lam intel (etch vs deposition, AI strategy, the governance-fit line) and the 9 values. Speaking: read the intro and 'why Lam' out loud 3x, record once.
- [ ] Day 2 — Make the intro and 'why Lam' your own words. Speaking: record both, listen for rushing + fillers, re-record. Target a calm 60s intro.
- [ ] Day 3 — Project deep-dives: Bilbo and AI Cargo. Be able to explain each in 2 min to a non-expert and defend one follow-up. Speaking: record both walkthroughs.
- [ ] Day 4 — Behavioral batch 1: PHANTOM TOOLS (failure), 18→7 (ownership), $4,000 (ambiguity). WRITE your real CONFLICT story (confirm the Bilbo GPT-4 details). Speaking: say each STAR in under 2 min.
- [ ] Day 5 — Behavioral batch 2: 70→7 (technical judgment), conflict, teamwork, learning-fast. Speaking: record all four; cut each to under 2 min with a number at the end.
- [ ] Day 6 — Technical conversation: review the companion canvas (RAG, agents, observability, cost/latency). Practice saying 3 explainers out loud plainly: how RAG works, how you'd monitor an LLM app, how you handle agent failures.
- [ ] Day 7 — Full mock interview (a friend, or an AI: 'act as a strict hiring manager at Lam, ask me one at a time, push back'). Record the whole thing. Note the 3 weakest answers.
- [ ] Day 8 — Fix the 3 weak answers. Lock your 5 questions to ask. Speaking: fluency drill + shadowing (15 min).
- [ ] Day 9 — Second short mock (intro + 2 projects + 3 behavioral + your questions). Light review only; don't cram new material.
- [ ] Day 10 (call day) — 10-min warm-up: read intro + one STAR out loud, do shadowing, hydrate. Re-skim this canvas. Quiet room, headphones, resume + a one-page cheat sheet of codenames and numbers. Rest the night before; sleep beats cramming.

> **One-page mental model for the whole round**
>
> You are not trying to prove you're the smartest person on the call. You are proving three things: you've genuinely built RAG and agents (your projects), you think clearly about tradeoffs and failures (STAR with a number), and you actually want Lam (the governance-fit line + real questions). Say less, say it slowly, end every answer on evidence. That is a strong-hire signal in any language.
