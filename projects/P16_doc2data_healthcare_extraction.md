# P16: Doc2Data — Healthcare Document-to-JSON Extraction Platform

> **Project Type:** Flagship self-project | **Live:** [thinkray.space](https://thinkray.space)
> **GitHub:** [github.com/rahul370139/doc2data](https://github.com/rahul370139/doc2data)
> **Resume line:** *"Multimodal Document Intelligence (Doc2Data): healthcare form-to-JSON platform using FastAPI, LangGraph, and OCR, with quality-aware extraction lanes and targeted VLM rescue for accurate structured output."*

**One-line pitch:** *A LangGraph-orchestrated pipeline that turns a messy healthcare form (CMS-1500 / UB-04) — fillable PDF, digital, or a crooked phone scan — into validated structured JSON in ~40 seconds, by picking the cheapest extraction lane that will work, gating expensive OCR behind an alignment check, and spending a vision-language model only on the handful of fields that actually need rescuing.*

---

## Table of Contents

1. [STAR Summary](#1-star-summary)
2. [Architecture](#2-architecture)
3. [The Three-Lane Strategy (the core idea)](#3-the-three-lane-strategy)
4. [OCR v2 — Batched Florence-2 + Adaptive Blank Detection](#4-ocr-v2)
5. [Reflect → Targeted VLM Rescue](#5-reflect--targeted-vlm-rescue)
6. [Alignment, Validation & Output Contract](#6-alignment-validation--output-contract)
7. [The Performance Story (4 min → 40 s)](#7-the-performance-story)
8. [Topics You Must Know](#8-topics-you-must-know)
9. [Interview Q&A](#9-interview-qa)
10. [Red Flags & How to Handle](#10-red-flags--how-to-handle)
11. [Key Takeaways](#11-key-takeaways)

---

## 1. STAR Summary

| Component | Detail |
|-----------|--------|
| **Situation** | Healthcare claim forms (CMS-1500, UB-04) arrive as a chaotic mix: clean fillable PDFs, digital-text PDFs, and low-quality handwritten scans. Naively running a heavy OCR/VLM model on every field of every form is slow (3–5 min/page), expensive, and *still* wrong on scans because a misaligned template makes every zone read garbage. |
| **Task** | Build a reliable, fast, self-hostable document-to-JSON extractor with per-field confidence, validation, and a full debug trace — accurate enough for downstream claim systems, cheap enough to run on one GPU. |
| **Task (constraints)** | On-prem/DGX (PHI-sensitive, so no cloud OCR APIs), page-1 scope, 86-field CMS-1500 / 98-field UB-04 schemas, sub-minute latency. |
| **Action** | Built a **LangGraph** state machine (`load → identify → plan → extract → validate → reflect → rescue → finalize`) with a **three-lane** extraction strategy that routes each form to the cheapest method that works; an **OCR v2** module that replaces a per-field Florence-2 loop with one *batched* call gated by an adaptive structural **blank detector**; an **alignment gate** that short-circuits heavy OCR on badly-registered scans; and a **reflect step** that hands only ≤12 suspect fields to a targeted VLM rescue. Next.js 14 frontend with a live SSE progress stream; FastAPI backend; deployed via Docker + Cloudflare Tunnel. |
| **Result** | End-to-end extraction in **~40 s** (down from 3–5 min); one pathological scan dropped from **258 s → 41 s** once the alignment gate landed. Structured JSON with `extracted_fields`, per-field confidence + bbox + source, typed validation (NPI/ICD/date/money/…), and a per-node timing/trace debug object. Live public demo on `thinkray.space`. |

---

## 2. Architecture

```
PDF / Image
   │
   ▼  LangGraph StateGraph  (src/pipelines/graph/{graph,nodes,state}.py)
 load ── render page @300 DPI, extract digital words
   │
 identify ── form fingerprint (PaddleOCR header/footer + CMS strong-tokens)
   │            → cms-1500 | ub-04 | ncpdp | generic
   │
 plan ── choose the cheapest viable lane
   ├── Lane A (widgets)   : fillable PDF, ≥10 filled CMS / ≥3 UB-04
   ├── Lane B (digital)   : digital text layer → zone match
   └── Lane C (scan OCR)  : scan/handwritten/Lane-B downgrade
   │
 [Lane B guard] sparse match OR >30% duplicate text → downgrade to Lane C
   │
 align (Lane C) ── AKAZE/ORB + RANSAC homography (dropout-red mask)
   ├── quality ≥ 0.5 → extract_scan (OCR v2)
   └── quality < 0.5 → alignment_failed  (skip heavy OCR, emit qa_note)
   │
 extract_* ── widgets | digital zones | OCR v2 batched Florence-2
   │
 validate ── typed validators (NPI, ICD, date, phone, money, zip, tax_id) + qa_notes
   │
 reflect ── score every field (error, low-conf, uncertain-blank, token mismatch)
   │          keep ≤12 rescue candidates
   ├── candidates → rescue (targeted VLM) → revalidate → finalize
   └── none       → finalize
   │
 finalize ── business-schema mapping + reducto export + debug(node_trace, timings)
   │
   ▼
Structured JSON  +  SSE node_start/node_end/final events
```

### Tech stack

| Layer | Technology |
|-------|-----------|
| **Orchestrator** | LangGraph state machine (nodes + conditional edges + typed `GraphState`) |
| **Primary OCR** | Florence-2-large (batched, `<OCR>` task) |
| **VLM rescue / tables** | MiniCPM-V / MiniCPM-o4.5 (via Ollama) |
| **Form ID / general OCR** | PaddleOCR; TrOCR (optional handwriting); YOLOv8 (optional layout) |
| **Alignment** | OpenCV AKAZE/ORB feature matching + RANSAC homography |
| **Backend** | FastAPI (`/extract/graph`, `/extract/graph/stream` SSE) |
| **Frontend** | Next.js 14 + TypeScript + Tailwind (drag-drop, live progress, multi-tab explorer) |
| **Deploy** | Docker Compose on DGX; Cloudflare Tunnel (no inbound ports) → `thinkray.space` |
| **Runtime proxy** | Next.js Route Handler proxies `/api/backend/*` to FastAPI at request time |

---

## 3. The Three-Lane Strategy

The whole design philosophy is **"don't pay for OCR you don't need."** Each form is routed to the cheapest method that will actually work.

| Lane | Trigger | Method | Cost |
|------|---------|--------|------|
| **A — widgets** | Fillable PDF with ≥10 filled CMS widgets (≥3 UB-04) | Read AcroForm values → map to schema IDs. **No OCR.** | Cheapest, most reliable |
| **B — digital text** | Non-structured form with a usable digital text layer | `page.get_text("words")` → match words to schema zones | Cheap |
| **C — scan OCR** | No widgets, scan/handwriting, or a Lane-B downgrade | Template alignment → schema zones → **batched Florence-2 (OCR v2)** → adaptive blank detection → optional VLM rescue | Expensive (gated) |

**Two downgrade guards (added after real failures):**
1. `lane_b_downgraded_to_c` — Lane B chosen but <10 CMS / <5 UB-04 blocks matched, **or >30% of matched blocks share identical text** (a template-mismatch guard: identical repeated text means we matched the pre-printed template, not the filled values).
2. `lane_c_alignment_failed` — Lane C chosen but alignment quality < 0.5, so heavy OCR is skipped and a `qa_note` is emitted instead of burning minutes on a page that will read as noise.

For CMS-1500/UB-04 the planner *prefers Lane C over Lane B* because the schema bounding boxes are calibrated to the canonical template — so alignment + zones beats trusting an inconsistent embedded text layer.

---

## 4. OCR v2

The single biggest engineering win. The original pipeline ran Florence-2 **once per field** (~90 fields × sequential inference = 3–5 min). OCR v2 (`src/pipelines/ocr_v2/`) replaces that with:

| Component | Role |
|-----------|------|
| `BlankDetector` | Per-form **calibration** on known-blank zones (compute ink_mean/std, connected-component count, text density) → adaptive thresholds. Then classify every zone using a **center-70% crop** (avoids padding/edge bleed) into `blank` / `filled` / `uncertain`. |
| `BatchedFlorence2` | Load Florence-2-large **once**, run a **single batched forward pass** over all `filled`+`uncertain` crops. |
| `FieldOCRBatch` | Glue: schema zones → blank detector → batched inference → per-field `{value, confidence, blank_status, signals}`. |

**Why this matters:** blank fields (the majority of a claim form) are *skipped entirely* — they emit an empty value with no model call. Only genuinely non-blank crops hit the model, and they hit it in one batch. That's what takes end-to-end OCR from ~4 min to ~20–30 s.

Per-field post-processing then cleans model output: template-keyword filter (blank the field if >50% of tokens are pre-printed labels like "CARRIER"/"PICA"), hallucination filter (reject garbled/repeated text), CCA self-consistency check at 2× upscale, and template-bleed stripping.

---

## 5. Reflect → Targeted VLM Rescue

This is the "spend the expensive model surgically" idea, and it's a clean agentic pattern.

- After validation, `reflect_node` **scores every field** by four signals: has a validation error, low confidence, `uncertain` blank status, or token mismatch. It keeps the **top ≤12** as rescue candidates.
- `rescue_node` sends **only those fields** to a VLM (MiniCPM-V) via `MultiAgentPipeline._apply_targeted_vlm_rescue` — not the whole page.
- Then `revalidate` re-runs validators on the rescued values before `finalize`.

So a VLM (slow, powerful) is used as a **precision instrument on a dozen suspect fields**, not a blunt tool on all 86. It's the same economics as the three-lane routing, applied at field granularity.

Tables (CMS Box 24, the service-line grid) are their own path: a VLM (`MiniCPM-o4.5`, temperature 0.0) extracts rows, deduped by `(cpt_code, charges)`, with a cell-OCR fallback if the VLM returns nothing.

---

## 6. Alignment, Validation & Output Contract

**Alignment (`cms1500_register.py`):** builds a dropout-red mask (CMS-1500 templates are printed in red so the red channel *is* the template), runs AKAZE/ORB feature matching + RANSAC homography to register the scan to the canonical template, with a 4-point "quad fallback" when feature matching is weak. The resulting **alignment quality** is the gate for whether Lane C spends OCR budget.

**Validation:** typed validators for NPI (checksum), ICD/HCPCS codes, dates, phone, SSN, zip, money, tax_id → structured `errors`/`warnings`/`qa_notes`.

**Output contract (stable JSON):**
```json
{
  "success": true, "form_type": "cms-1500",
  "extracted_fields": { "2_patient_name": "..." },
  "field_details": [{ "id": "...", "value": "...", "confidence": 0.91,
                      "bbox": [x0,y0,x1,y1],
                      "metadata": {"source":"schema_zones","ocr_engine":"florence2"} }],
  "business_fields": { "patient_name": "..." },
  "validation": { "errors": [], "warnings": [], "qa_notes": [] },
  "debug": { "alignment_used": true, "alignment_quality": 0.88,
             "vlm_rescue_count": 1, "node_trace": ["load","identify",...],
             "node_timings_ms": {...} },
  "reducto_format": { "result": { "chunks": [...] } }
}
```
Everything is traceable: which lane, which OCR engine per field, alignment quality, rescue count, and per-node wall-clock. That's what makes it *debuggable*, not a black box.

---

## 7. The Performance Story

| Fixture | Plan reason | Alignment Q | Wall-clock | Non-blank fields |
|---|---|---:|---:|---:|
| `cms1500.pdf` | Lane B → C downgrade | 0.62 | 41 s | 22 |
| `cms1500_1.pdf` | Lane A widgets + rescue | n/a | 38 s | 37 |
| `cms1500_2.pdf` | Lane C scan + OCR v2 | 0.74 | 47 s | 29 |
| `cms1500_3.pdf` | Lane C scan + OCR v2 | 0.58 | **41 s** (was **258 s**) | 24 |
| `cms1500_6.pdf` | Lane C scan + OCR v2 | 0.81 | 70 s | 41 |

The headline: `cms1500_3.pdf` used to burn **258 seconds** of Florence-2 time on a page whose alignment had *silently failed* — every zone was reading garbage from a misregistered template. The alignment gate now catches that in **under one second** and emits a QA note instead. Pre-refactor these fixtures took 3–5 min each; post-refactor ~40 s.

**The two levers, restated:** (1) *route to the cheapest lane*, (2) *gate the expensive model behind a cheap check* (alignment quality for the page; blank detection for each field; reflect for rescue). Both are "spend compute only where it changes the answer."

---

## 8. Topics You Must Know

- **LangGraph state machines** — typed `GraphState`, conditional edges, why a graph beats a linear script here (branching lanes, gates, optional rescue). (`learning/17_langchain_langgraph.md`.)
- **VLMs & OCR** — Florence-2 (`<OCR>` grounded tasks), MiniCPM-V, batched inference, when a VLM beats classic OCR. (`learning/21_computer_vision.md`.)
- **Classic CV** — feature matching (AKAZE/ORB), RANSAC homography, template subtraction, connected-component analysis, ink-ratio thresholds.
- **Cost-aware inference** — routing, gating, batching; "blank detection so you never OCR empty fields."
- **Structured extraction & validation** — schema-driven zones, typed validators, confidence + provenance per field.
- **Streaming UX** — SSE (`node_start`/`node_end`/`final`) for a live progress timeline.
- **Self-hosted / PHI constraints** — why no cloud OCR APIs; DGX + Docker + Cloudflare Tunnel.
- **Eval** — F1 vs gold labels per field; per-node timing harness (`scripts/test_graph_pipeline.py`).

---

## 9. Interview Q&A

**Q1: Walk me through Doc2Data.**
> A form comes in and a LangGraph state machine takes over: load renders the page and grabs any digital words, identify fingerprints the form type, and plan picks the cheapest extraction lane — read the fillable-PDF widgets if they exist, else match the digital text layer, else fall back to scan OCR. For scans I align the image to the canonical template and only OCR if alignment quality clears a threshold; otherwise I skip the expensive pass and flag it. OCR itself is one batched Florence-2 call over only the non-blank fields, decided by an adaptive blank detector. Then I validate every field with typed validators, reflect to find the ≤12 most suspicious fields, and send just those to a VLM for rescue before finalizing to structured JSON with per-field confidence, bounding boxes, and a full debug trace. It runs in about 40 seconds on one GPU.

**Q2: Why LangGraph instead of a linear pipeline?**
> Because the flow genuinely branches and loops: three mutually exclusive lanes, a downgrade edge from B to C, an alignment gate that can skip the OCR node, and a conditional rescue loop. A graph with typed shared state and conditional edges expresses that cleanly and — crucially — gives me a node trace and per-node timings for free, which is what made the whole thing debuggable. A pile of if-statements would have hidden exactly the control flow I most needed to see.

**Q3: What was the single biggest optimization?**
> The alignment gate plus batched blank-aware OCR. The original ran Florence-2 once per field, sequentially, on every zone including blanks — 3 to 5 minutes a page. One scan spent 258 seconds because its template alignment had silently failed, so every field was OCR'ing noise. I added: (1) an alignment-quality gate that short-circuits heavy OCR in under a second when registration is bad, and (2) OCR v2 — a per-form blank detector that skips empty fields and a single batched Florence-2 call for the rest. That's what took it to ~40 seconds.

**Q4: How does blank detection work and why does it matter?**
> A claim form is mostly empty, and OCR'ing empty boxes is both wasted compute and a hallucination risk. For each form I calibrate on known-blank zones to learn adaptive ink/texture thresholds, then classify every schema zone using the center 70% of the crop (to avoid padding/border bleed) into blank, filled, or uncertain. Blanks are emitted empty with no model call; only filled and uncertain crops go into the batched Florence-2 pass. It cuts model calls dramatically and reduces false extractions.

**Q5: Where do you use a VLM, and why not everywhere?**
> Surgically. After validation, a reflect step scores each field by validation error, low confidence, uncertain-blank status, and token mismatch, and keeps at most 12 candidates. Only those go to MiniCPM-V for rescue, then I revalidate. VLMs are accurate but slow and expensive, so I treat one like a precision instrument on the dozen fields that are actually doubtful, not a blunt instrument on all 86. Tables (Box 24) are the one place I use a VLM directly, because grid structure is exactly what VLMs are good at.

**Q6: How do you know it's accurate — how do you evaluate?**
> Field-level F1 against gold labels via `scripts/test_graph_pipeline.py`, which also records per-node timings, validation errors, and non-blank field counts per fixture. Confidence and provenance are attached per field (which lane, which OCR engine), so I can see *why* a field is what it is. Typed validators (NPI checksum, ICD codes, date/money formats) catch structurally-invalid values before they ever leave the pipeline.

**Q7: Why self-host instead of using a cloud OCR/Document AI API?**
> PHI. Claim forms carry protected health information, so shipping them to a third-party cloud API is a compliance problem. Everything runs on a DGX box with local models (Florence-2, MiniCPM via Ollama) and is exposed through a Cloudflare Tunnel with no inbound ports open. It also means predictable cost and no per-page API fees.

**Q8: How would you scale this to 100-page documents / high throughput?**
> Today it's page-1 scoped and synchronous. I'd move extraction behind a job queue (workers pull pages), parallelize pages across GPUs, cache form-identification and alignment per template, and stream partial results per page over the existing SSE channel. The lane routing already keeps per-page cost low; the win is horizontal workers plus batching across pages, not just within a page.

**Q9: What's the failure mode you're proudest of handling?**
> Silent misalignment. Before the gate, a crooked scan didn't error — it produced confident-looking garbage after wasting minutes. Now a low alignment quality is treated as a first-class signal: skip the heavy OCR, emit an explicit `alignment_failed` QA note, and tell the user *why* rather than returning junk. "Fail loud and cheap" beat "fail slow and silently."

---

## 10. Red Flags & How to Handle

| Red flag | How to handle |
|----------|---------------|
| "Only page 1?" | Yes — deliberate scope for the demo. The graph is page-agnostic; multi-page is a worker-per-page fan-out, not a redesign. |
| "Florence-2 hallucinations?" | That's why there's a multi-stage filter chain: template-keyword filter, hallucination filter, CCA self-consistency at 2×, template-bleed strip, then typed validation, then targeted VLM rescue on the doubtful ones. |
| "Hardcoded thresholds (0.5 alignment, 30% duplicate)?" | Tuned via grid search (`tune_cms1500_thresholds.py`) and recomputed from a corrections log (`recompute_thresholds.py`); they're calibrated, not guessed, and per-form calibrated at runtime for blanks. |
| "No accuracy number in the headline?" | I report per-field F1 vs gold and per-node timings on fixtures; accuracy is field- and scan-quality-dependent, so I show the eval harness rather than a single vanity number. |
| "Ollama/VLM is a heavy dependency." | Rescue and tables degrade gracefully — VLM off still gives widget/digital/Florence-2 extraction; the VLM only sharpens the doubtful subset. |

---

## 11. Key Takeaways

- **What it demonstrates:** cost-aware multimodal extraction — routing to the cheapest lane, gating expensive models behind cheap checks (alignment, blank detection, reflect), and surgical VLM rescue; all inside an inspectable LangGraph state machine with per-field confidence, validation, and full trace.
- **Signals:** you optimize systems by *not doing work* (skip blanks, skip misaligned OCR, rescue only the doubtful), you care about *debuggability and provenance*, and you respect real constraints (PHI → self-hosted).
- **Best-fit roles:** Applied AI Engineer, ML/AI Engineer (multimodal/CV/OCR), Forward-Deployed Engineer.
- **30-second pitch:** *"Doc2Data turns messy healthcare claim forms into validated JSON in ~40 seconds. A LangGraph pipeline routes each form to the cheapest lane that works, gates heavy OCR behind a template-alignment check, runs one batched Florence-2 call over only the non-blank fields, and spends a vision-language model only on the dozen fields a reflect step flags as doubtful. It self-hosts on a DGX for PHI safety, and cut one pathological scan from 258 seconds to 41."*
