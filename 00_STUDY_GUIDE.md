# Study Guide — How to Use This Folder to Crack Interviews

**Purpose:** You have 52 learning docs + 20 project docs. This is the map: what to read for which role, in what order, how to practice (not just read), and what to do in the 48 hours before an interview.

> ## 🔴 START HERE — read these two first, before any learning doc
> 1. **[`00_RESUME_MASTER_MAP.md`](00_RESUME_MASTER_MAP.md)** — your new resume (`Rahul Sharma_edited 18th`) bullet-by-bullet, with the metric, the likely follow-up, and the answer. **§1 lists numbers that CHANGED from your old resume** — if you rehearse the old ones you'll contradict yourself in an interview. Fix those first.
> 2. **[`learning/52_frontier_ai_and_2026_landscape.md`](learning/52_frontier_ai_and_2026_landscape.md)** — what the field added recently and what 2026 loops actually test.
>
> **The single most important fact about 2026 AI interviews:** the weighting is roughly **40% RAG/evals/agents, 30% production systems, 20% LLM internals, 10% behavioral** — and **evaluation is the top filter.** Candidates who can build a RAG pipeline but can't say how they'd measure it get rejected. Every technical answer you give should end with *"and here's how I'd know it's working."*

---

## 1. The folder at a glance

```
interview_prep/
├── 00_STUDY_GUIDE.md            ← you are here
├── 00_RESUME_MASTER_MAP.md      ← ⭐ resume → proof → drill (read first)
├── learning/
│   ├── 01–21  Deep learning & ML core (activations, losses, transformers,
│   │          RAG, vector DBs, fine-tuning, agents, CV, DL architectures)
│   ├── 22–32  Applied engineering (SQL/DE, MLOps, stats, Python, system
│   │          design, agentic patterns, Terraform, orchestration)
│   ├── 33–47  DS depth (causal inference, time series, recsys, product
│   │          sense, GenAI-for-DS, experimentation platform, CUPED,
│   │          switchback, measurement debugging, econometrics,
│   │          marketplace, ads, statistical reasoning)
│   ├── 48     Practical interview question bank  ← the drill sheet
│   ├── 49     LLMOps & observability (LangSmith, Langfuse, evals in CI)
│   ├── 50     LLM serving & inference optimization (vLLM, speculative decoding)
│   ├── 51     Credit risk & scorecard modeling (WOE/IV, KS/Gini, PSI, OOT)
│   ├── 52     Frontier AI & the 2026 landscape  ← the "what's new" doc
│   └── 53     Behavioral interview mastery (rubrics, story bank, AI screens)
└── projects/
    ├── 00_FLAGSHIP_PROJECTS_STUDY_GUIDE.md  ← how to master the 3 flagships
    ├── P01–P14  One doc per resume project (STAR + architecture + Q&A)
    ├── P15–P17  Flagship deep-dives: AI Cargo · Doc2Data · PathWise
    └── FusionSpan_Python_Coding_Questions.md  (coding drills)
```

**Your flagship (deep-dive round) projects** — an interviewer will pick ONE and drill for 30–45 min:

| Priority | Project | Why |
|---|---|---|
| **#1** | [`P07_bilbo_medical_rag`](projects/P07_bilbo_medical_rag.md) | **Your current role and the first thing on your resume.** RAG + hybrid retrieval + reranking + LangGraph agents + guardrails = exactly what 2026 AI interviews test. Non-negotiable. |
| **#2** | [`P15_ai_cargo_agentic_risk_platform`](projects/P15_ai_cargo_agentic_risk_platform.md) | The only project on your resume; prize-winning; agentic + explainable ML + HITL |
| #3 | [`P16_doc2data_healthcare_extraction`](projects/P16_doc2data_healthcare_extraction.md) | Best *systems / cost-aware inference* story |
| #4 | [`P17_pathwise_career_simulator`](projects/P17_pathwise_career_simulator.md) | Best *evaluation* story — and evals are the 2026 differentiator |
| #5 | [`P02`](projects/P02_jet2_call_transcript_classification.md)/[`P04`](projects/P04_jet2_risk_forecasting.md) Jet2, [`P01`](projects/P01_fibe_behaviour_scorecard.md) Fibe | Your depth in *employed* work — distillation, forecasting, credit risk |

Start with [`projects/00_FLAGSHIP_PROJECTS_STUDY_GUIDE.md`](projects/00_FLAGSHIP_PROJECTS_STUDY_GUIDE.md) for the 4-layer mastery model and a 2-week plan.

**Two kinds of files, two kinds of study:**
- **Learning docs** = reference depth. Read actively (see §4), don't binge.
- **Project docs** = your ammunition. Every conceptual answer should end with "…and I did this in <project>." These are rehearsal scripts, not reading material.

---

## 2. Reading paths by role

Read in order; ★ = must, ○ = if time. `48` is always first — it tells you what gets asked, then you backfill depth.

### AI Engineer / GenAI Engineer  ← **your primary target**
1. ★ `00_RESUME_MASTER_MAP` — fix the changed numbers, learn your own bullets
2. ★ `52_frontier_ai_and_2026_landscape` — what's asked now; memorize the canonical "build me a RAG system" answer (§17 Q1)
3. ★ `48` §2–§5 (landscape, question bank, production themes, system design)
4. ★ `05_rag_systems` → `06_vector_databases` → `16_context_engineering`
5. ★ `10_agentic_ai_and_multi_agent` → `27_agentic_patterns_deep_dive` → `17_langchain_langgraph`
6. ★ `14_evaluation_metrics` + **`49_llmops_and_observability`** — **the differentiator. Do not skip.**
7. ★ **`50_llm_serving_and_inference_optimization`** — you list vLLM + speculative decoding on your resume; you must be able to defend them
8. ★ `26_system_design_for_ml` + `15_fastapi_and_backend`
9. ★ `04_transformers_and_attention` → `07_fine_tuning_and_peft` → `08_knowledge_distillation` (your Jet2 70B→7B story)
10. ○ `13_quantization`, `31_ai_safety_guardrails`, `23_cloud_mlops_deployment`, `11_a2a_and_mcp_protocols`, `30_prompt_engineering_patterns`
11. ○ **`25_python_for_interviews` §0, §7.0–§7.3, §10–§12** if the loop has a live coding round (pattern recognition, not language trivia)
12. **Projects to rehearse, in order:** **P07 (Bilbo — your job)**, P15 (AI Cargo), P16 (Doc2Data), P17 (PathWise), P02 (distillation)

### Forward Deployed Engineer
1. ★ `48` §6 (decomposition script) — practice the case study format 3× out loud
2. ★ Everything in the AI Engineer path (FDE = AI engineer + customer judgment)
3. ★ `22_data_engineering_and_sql` (Palantir-style loops lean on data)
4. ○ `28_terraform_iac`, `23_cloud_mlops` (on-prem/VPC constraints come up)
5. **Projects:** pick ONE flagship (P16 Doc2Data, P15 AI Cargo, or P17 PathWise) for the deep-dive round; P05b for the "messy client data" story

### Data Scientist (product / experimentation)
1. ★ `48` §7, §9
2. ★ `24_statistics_and_ab_testing` → `47_deep_statistical_reasoning`
3. ★ `19_classical_ml_algorithms` → `14_evaluation_metrics` → `03_regularization`
4. ★ `33_causal_inference` → `40_cuped` → `37_product_sense`
5. ○ `41_experimentation_platform`, `42_switchback`, `43_debugging_measurement`, `44–46` (econometrics/marketplace/ads — only if the company is a marketplace/ads business)
6. **Projects:** P14 (experimentation platform), P01 (scorecard), P04 (forecasting), P06

### Data Scientist (risk / credit / fintech)  ← newly viable with the updated resume
1. ★ **`51_credit_risk_and_scorecard_modeling`** — WOE/IV, monotonic binning, KS/Gini, PSI, OOT, reject inference
2. ★ `projects/P01_fibe_behaviour_scorecard` (0.94 AUC, 1,000+ variables) → `projects/P06` (Amazon Pay fraud + feature selection)
3. ★ `19_classical_ml_algorithms` → `14_evaluation_metrics` → `03_regularization`
4. ★ `24_statistics_and_ab_testing` (hypothesis testing) + `36_explainability_and_interpretability`
5. ○ `23_cloud_mlops_deployment` (MLflow sections — you claim MLflow on the resume)
6. **Be ready for:** "0.94 AUC on credit is suspiciously high — where's the leakage?" and "why logistic regression instead of XGBoost?"

### Data Engineer
1. ★ `48` §8
2. ★ `22_data_engineering_and_sql` (the core doc) → `32_data_pipeline_orchestration`
3. ★ `P05b_atcs_pyspark_etl_pipeline` (your shipped ETL story) → `39_oop_etl_concepts`
4. ○ `23_cloud_mlops`, `26_system_design_for_ml`, `28_terraform`
5. **Practice:** write SQL daily (window functions, top-N per group, funnels) — reading SQL ≠ writing SQL

### Analytics / Product Analyst
1. ★ `48` §9 → `37_product_sense_and_business_metrics`
2. ★ `24_statistics_and_ab_testing` (§6 A/B testing) → `43_debugging_measurement_systems`
3. ○ `33_causal_inference`, `40_cuped`, `22` (SQL sections)
4. **Projects:** P14, P03 (event detection = anomaly investigation story)

---

## 2b. Minimum-time first coverage (do this before any deep reading)

You cannot read 52 docs before your next interview, and you shouldn't try. **First pass = your own resume + vocabulary. Depth comes on revision.** In order:

| # | Task | Time | Why it's first |
|---|---|---|---|
| 1 | `00_RESUME_MASTER_MAP` §1 (changed numbers) + §5 (your 60s intro) | 45 min | Contradicting your own resume is the fastest way to fail |
| 2 | `P07` Bilbo: §0 bullet map, §8 LangGraph, §10 trick questions | 2 h | Your current role — guaranteed to be drilled |
| 3 | `52_frontier_ai` §1, §17 Q1 (canonical RAG answer), §16 decoder | 1.5 h | Tells you what's asked and gives you the most-asked answer |
| 4 | `49_llmops` — evaluation sections only | 1.5 h | The #1 filter in 2026 loops |
| 5 | `P15` AI Cargo: bullet map + 3 deep-dives | 1.5 h | The only project on your resume |
| 6 | Skim `48` section *titles* only | 20 min | A map of what gets asked; don't read the answers yet |
| 7 | One number per remaining resume stint (Jet2 94%/96%/MAPE, Fibe 0.94, ATCS 92%/18min, Amazon 95%) | 30 min | So no bullet is a blank |

That's roughly **9–10 hours to a defensible baseline.** Everything after that is revision and depth.

**Then, and only then**, backfill by role using §2. When low-energy, use videos/audio (see `00_FLAGSHIP_PROJECTS_STUDY_GUIDE.md` §6 drills and re-listen to your own recorded pitches) — never first-pass new material while tired.

---

## 3. The practice system (what actually moves the needle)

Reading feels productive; **retrieval practice is what sticks.** Weekly cadence:

| Day | Activity (60–90 min) |
|---|---|
| Mon | Pick 10 questions from `48` for your target role. Answer out loud, 90s each, **before** checking references. Grade yourself 0–3 (the rubric in `48` §2). |
| Tue | Backfill: read the referenced sections for every question you scored ≤1. |
| Wed | One system-design prompt from `48` §5.2 — 30 min whiteboard, out loud, phone recording. |
| Thu | Coding: start `25_python_for_interviews.md` **§0 then §11**, then 3 problems from §7 / §10 (two-sum both ways, window, RRF or entity-split). Extra bank: `FusionSpan_Python_Coding_Questions.md`. DE roles: 3 SQL problems instead. |
| Fri | One project deep-dive rehearsal: present a P-doc for 10 min, then answer its own Q&A section without looking. |
| Weekend | One full mock (see §5) or rest. |

Rules:
- **Out loud, always.** Knowing and articulating are different skills; interviews test the second.
- **Record yourself weekly.** Listen for filler, missing signposts, answers without endings.
- **Every answer ends with a project tie-in or an evaluation statement.** Build the habit now so it's automatic under pressure.
- **Never claim a resume skill you can't survive two follow-ups on.** If `vLLM` and `Speculative Decoding` stay on your resume, `50_llm_serving_and_inference_optimization` is mandatory reading — an interviewer who serves models will ask, and "I listed it but haven't used it" is a much better answer than a bluff that unravels.

---

## 4. How to read the learning docs (active, not passive)

1. Read the Table of Contents → try to explain each section title from memory first.
2. Read the section → close it → write a 3-line summary from memory.
3. Do the doc's own "Interview Questions with Strong Answers" section **cold** before reading the model answers (nearly every doc 19–47 has one).
4. Skip what you already know confidently — these docs are references, not a syllabus.

---

## 5. Mock interview protocol

- **Behavioral rounds:** every loop has one, and it's the round technical candidates lose without knowing why. Use **[`learning/53_behavioral_interview_mastery.md`](learning/53_behavioral_interview_mastery.md)** — scoring rubrics, your codenamed story bank with real numbers, the hard questions (gap, weakness, salary), company variants, and AI-screen mechanics. Its §13 is a 7-day out-loud rehearsal plan.
- **Self-mock:** pick 6 questions from `48` (2 conceptual, 2 production, 1 design, 1 behavioral), 45 min timer, record, review.
- **AI mock:** paste a question list from `48` into ChatGPT/Claude with: *"Act as a strict senior interviewer for a <role> role. Ask me these one at a time, push back with follow-ups, grade each answer 0–3 with one sentence of feedback."* This doubles as practice for AI-conducted screens (`48` §10).
- **Human mock:** at least one before a real onsite — a friend reading follow-ups from a P-doc's Q&A section is enough.

---

## 6. 48 hours before an interview

1. **Re-read the JD**, map every bullet to a doc + a project story (15 min).
2. **Re-read `48`** sections for that role only.
3. **Rehearse 3 things cold:** your intro (90s), your flagship project walkthrough (5 min + defense), your "production incident" story.
4. Skim the **Key Takeaways** boxes at the end of the relevant learning docs — they're built for this.
5. Prepare **5 questions to ask them** (about their evals, their data quality, their deployment cadence — signals you've shipped).
6. Stop studying the night before. Sleep is worth more than one more doc.

---

## 7. Maintenance

- After every real interview, add the questions you were asked to `48` §1 (that's how that section started).
- When you build something new, add/update its P-doc within a week while details are fresh.
- Once a month, re-run the Mon drill on random questions across roles — flags what's decaying.
