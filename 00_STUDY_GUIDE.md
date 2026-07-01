# Study Guide — How to Use This Folder to Crack Interviews

**Purpose:** You have 48 learning docs + 15 project docs. This is the map: what to read for which role, in what order, how to practice (not just read), and what to do in the 48 hours before an interview.

---

## 1. The folder at a glance

```
interview_prep/
├── 00_STUDY_GUIDE.md            ← you are here
├── learning/
│   ├── 01–21  Deep learning & ML core (activations, losses, transformers,
│   │          RAG, vector DBs, fine-tuning, agents, CV, DL architectures)
│   ├── 22–32  Applied engineering (SQL/DE, MLOps, stats, Python, system
│   │          design, agentic patterns, Terraform, orchestration)
│   ├── 33–47  DS depth (causal inference, time series, recsys, product
│   │          sense, GenAI-for-DS, experimentation platform, CUPED,
│   │          switchback, measurement debugging, econometrics,
│   │          marketplace, ads, statistical reasoning)
│   └── 48     Practical interview question bank  ← the drill sheet
└── projects/
    ├── P01–P14  One doc per resume project (STAR + architecture + Q&A)
    └── FusionSpan_Python_Coding_Questions.md  (coding drills)
```

**Two kinds of files, two kinds of study:**
- **Learning docs** = reference depth. Read actively (see §4), don't binge.
- **Project docs** = your ammunition. Every conceptual answer should end with "…and I did this in <project>." These are rehearsal scripts, not reading material.

---

## 2. Reading paths by role

Read in order; ★ = must, ○ = if time. `48` is always first — it tells you what gets asked, then you backfill depth.

### AI Engineer / GenAI Engineer
1. ★ `48` §2–§5 (landscape, question bank, production themes, system design)
2. ★ `05_rag_systems` → `06_vector_databases` → `16_context_engineering`
3. ★ `10_agentic_ai_and_multi_agent` → `27_agentic_patterns_deep_dive` → `17_langchain_langgraph`
4. ★ `14_evaluation_metrics` (LLM/RAG sections) — **the differentiator topic**
5. ★ `26_system_design_for_ml` + `15_fastapi_and_backend`
6. ★ `04_transformers_and_attention` → `07_fine_tuning_and_peft` → `30_prompt_engineering_patterns`
7. ○ `13_quantization`, `31_ai_safety_guardrails`, `23_cloud_mlops_deployment`, `11_a2a_and_mcp_protocols`
8. **Projects to rehearse:** Doc2Data, AI Cargo, P07 (Medical RAG), P08 (SkillSpring)

### Forward Deployed Engineer
1. ★ `48` §6 (decomposition script) — practice the case study format 3× out loud
2. ★ Everything in the AI Engineer path (FDE = AI engineer + customer judgment)
3. ★ `22_data_engineering_and_sql` (Palantir-style loops lean on data)
4. ○ `28_terraform_iac`, `23_cloud_mlops` (on-prem/VPC constraints come up)
5. **Projects:** pick ONE flagship (Doc2Data or AI Cargo) for the deep-dive round; P05b for the "messy client data" story

### Data Scientist (product / experimentation)
1. ★ `48` §7, §9
2. ★ `24_statistics_and_ab_testing` → `47_deep_statistical_reasoning`
3. ★ `19_classical_ml_algorithms` → `14_evaluation_metrics` → `03_regularization`
4. ★ `33_causal_inference` → `40_cuped` → `37_product_sense`
5. ○ `41_experimentation_platform`, `42_switchback`, `43_debugging_measurement`, `44–46` (econometrics/marketplace/ads — only if the company is a marketplace/ads business)
6. **Projects:** P14 (experimentation platform), P01 (scorecard), P04 (forecasting), P06

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

## 3. The practice system (what actually moves the needle)

Reading feels productive; **retrieval practice is what sticks.** Weekly cadence:

| Day | Activity (60–90 min) |
|---|---|
| Mon | Pick 10 questions from `48` for your target role. Answer out loud, 90s each, **before** checking references. Grade yourself 0–3 (the rubric in `48` §2). |
| Tue | Backfill: read the referenced sections for every question you scored ≤1. |
| Wed | One system-design prompt from `48` §5.2 — 30 min whiteboard, out loud, phone recording. |
| Thu | Coding: 3 problems from `FusionSpan_Python_Coding_Questions.md` or `25_python_for_interviews`; for DE roles, 3 SQL problems instead. |
| Fri | One project deep-dive rehearsal: present a P-doc for 10 min, then answer its own Q&A section without looking. |
| Weekend | One full mock (see §5) or rest. |

Rules:
- **Out loud, always.** Knowing and articulating are different skills; interviews test the second.
- **Record yourself weekly.** Listen for filler, missing signposts, answers without endings.
- **Every answer ends with a project tie-in or an evaluation statement.** Build the habit now so it's automatic under pressure.

---

## 4. How to read the learning docs (active, not passive)

1. Read the Table of Contents → try to explain each section title from memory first.
2. Read the section → close it → write a 3-line summary from memory.
3. Do the doc's own "Interview Questions with Strong Answers" section **cold** before reading the model answers (nearly every doc 19–47 has one).
4. Skip what you already know confidently — these docs are references, not a syllabus.

---

## 5. Mock interview protocol

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
