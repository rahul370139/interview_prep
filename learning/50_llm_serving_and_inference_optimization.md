# LLM Serving & Inference Optimization — vLLM, Speculative Decoding, and Production Model Serving

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** AI Engineer / LLM Engineer — vLLM, speculative decoding, KV-cache, continuous batching, Kubernetes GPU serving

**Purpose:** Everything you need to defend the resume skills **vLLM**, **speculative decoding**, **Kubernetes**, and **model serving** — prefill vs decode, TTFT/TPOT/throughput/goodput, KV-cache math, PagedAttention, continuous batching, engine choice, parallelism, capacity, and cost. Doc `49` is *how you know the app is working*. This doc is *how the tokens actually get produced*.

**Honesty box (read this before any interview):**
- **Bilbo** served **4-bit Mistral-7B via Ollama**, not a multi-GPU vLLM cluster. **AI Cargo** routes Groq / Ollama / OpenAI / Anthropic with a deterministic fallback.
- You **list** vLLM and speculative decoding on the resume. That means you must survive two follow-ups on the **mechanics**. It does **not** mean you should claim you owned production vLLM at Bilbo.
- Spoken line if they press production ownership: *"I have served locally with Ollama and I know vLLM's scheduler, PagedAttention, prefix cache, and speculative decoding well enough to size and operate a replica. I have not run a multi-node vLLM fleet in production — I would not invent that."*
- Cross-ref: `13_quantization.md` (weights / KV precision), `49_llmops_and_observability.md` (TTFT as an SLO, cost, traces), `23_cloud_mlops_deployment.md` (K8s, KEDA), `04_transformers_and_attention.md` (why decode is sequential).

---

# Table of Contents

1. [Why Serving Is a Different Problem](#1-why-serving-is-a-different-problem)
2. [Prefill vs Decode — The Two Phases](#2-prefill-vs-decode--the-two-phases)
3. [The Metrics That Matter](#3-the-metrics-that-matter)
4. [KV-Cache Math](#4-kv-cache-math)
5. [PagedAttention](#5-pagedattention)
6. [Continuous Batching](#6-continuous-batching)
7. [vLLM — The Engine You Listed](#7-vllm--the-engine-you-listed)
8. [Engine Comparison](#8-engine-comparison)
9. [Speculative Decoding](#9-speculative-decoding)
10. [Quantization for Serving](#10-quantization-for-serving)
11. [Tensor, Pipeline, and Expert Parallelism](#11-tensor-pipeline-and-expert-parallelism)
12. [Kubernetes GPU Serving](#12-kubernetes-gpu-serving)
13. [Capacity Planning & Cost](#13-capacity-planning--cost)
14. [Reference Architecture](#14-reference-architecture)
15. [Interview Questions with Strong Answers](#15-interview-questions-with-strong-answers)
16. [Key Takeaways](#16-key-takeaways)

---

# **1. Why Serving Is a Different Problem**

---

## **1.1 Training vs Serving vs "I called OpenAI"**

| | Training | Calling a provider API | Self-hosting (vLLM / TGI / TRT-LLM) |
|---|---|---|---|
| **You own** | Weights, optimizer, data | Prompt, tools, retries | Weights **and** the GPU scheduler |
| **Bottleneck** | Compute (GEMMs), data pipeline | Dollars and p95 of *their* fleet | **Memory** (KV-cache) + **memory bandwidth** (decode) |
| **Unit of work** | Step / batch | Request | Token, then request, then GPU |
| **Failure mode** | Loss doesn't drop | 429, silent model swap, cost spike | OOM, KV fragmentation, queue blow-up, cold start |
| **When you pick it** | You are changing weights | Privacy / latency / cost don't force you on-box | Data residency, cost at scale, prefix reuse, LoRA multi-tenant, tail control |

Self-hosting is not "faster ChatGPT." It is **an operating system for tokens**: a scheduler, a paged memory manager, a batcher, and an SLO.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WHAT "SERVING" ACTUALLY IS                           │
│                                                                         │
│   Client ──HTTP / gRPC──► Gateway (auth, quota, route)                  │
│                              │                                          │
│                              ▼                                          │
│                    ┌───────────────────┐                                │
│                    │  Scheduler        │  waiting queue, priorities     │
│                    │  Continuous batch │  mix prefill + decode          │
│                    └─────────┬─────────┘                                │
│                              ▼                                          │
│                    ┌───────────────────┐                                │
│                    │  KV block manager │  PagedAttention / Radix        │
│                    │  Prefix cache     │  share system prompts          │
│                    └─────────┬─────────┘                                │
│                              ▼                                          │
│                    ┌───────────────────┐                                │
│                    │  GPU kernels      │  prefill GEMM + decode GEMV    │
│                    │  (+ draft model)  │  speculative verify            │
│                    └───────────────────┘                                │
│                                                                         │
│   The model forward pass is ~30% of the story. The other 70% is         │
│   memory layout, scheduling, and not wasting the GPU on idle bubbles.   │
└─────────────────────────────────────────────────────────────────────────┘
```

## **1.2 The one-sentence mental model**

> **Prefill is compute-bound. Decode is memory-bound. The KV-cache is the working set. Everything that looks like a serving trick — paging, continuous batching, prefix cache, speculative decoding, KV quantization — is a way to keep the GPU busy without running out of HBM.**

If you remember only that, you can derive the rest of this document in an interview.

---

# **2. Prefill vs Decode — The Two Phases**

---

## **2.1 What happens on one request**

A decoder-only Transformer does **not** generate the whole answer in one shot. There are two different kernels, with different bottlenecks.

| Phase | What it does | Parallelism | Bound by | What the user feels |
|---|---|---|---|---|
| **Prefill** | Run the prompt (all $N_{\text{in}}$ tokens) through the model; write the initial K/V; emit token 1 | Tokens in the prompt are parallel (one big GEMM) | **Compute** (FLOPs) | **TTFT** |
| **Decode** | For each new token: read *all* existing K/V, do one step, append one K/V pair, emit token $t$ | Sequential in time; batched across requests | **HBM bandwidth** (weights + KV) | **TPOT / ITL**, then total time |

$$
T_{\text{e2e}} \approx T_{\text{queue}} + T_{\text{prefill}}(N_{\text{in}}) + T_{\text{decode}} \times (N_{\text{out}} - 1)
$$

$T_{\text{prefill}}$ grows roughly with $N_{\text{in}}$ (and worse with $N_{\text{in}}^2$ attention if you have not chunked). $T_{\text{decode}}$ is almost independent of how "smart" the prompt is — it is "how many tokens you asked for" times "how loaded the GPU is."

## **2.2 Why decode is memory-bound**

At decode, batch-size-per-request is 1 token. You load **the entire model weights** plus **the growing KV** to produce a handful of FLOPs per parameter.

Arithmetic intensity (FLOPs / byte) for decode is low. The GPU's tensor cores sit waiting on HBM. That is why:

- A 70B FP16 model can be **slower per token** than a 7B even when both "fit."
- **Quantizing weights** (AWQ/GPTQ/FP8) often speeds decode *even if* compute is not the limit — you move fewer bytes.
- **Speculative decoding** helps: you pay one target-forward to accept several tokens, which raises work per weight-load.

## **2.3 Chunked prefill**

A 32k-token prompt can monopolize the GPU for hundreds of milliseconds and stall every decode in the batch (TTFT for *other* users spikes). **Chunked prefill** splits the prompt into blocks (e.g. 512 tokens) and interleaves those chunks with decode steps so interactive users keep getting tokens.

vLLM flag: `--enable-chunked-prefill` (on by default in recent versions for many configs). Spoken line: *"I chunk long prefills so a RAG dump cannot starve the decode batch."*

## **2.4 Streaming is not a serving optimization — it is a UX one**

Streaming does **not** reduce $T_{\text{e2e}}$. It reduces **time-to-useful-pixels**. For chat, SLO on **p95 TTFT** plus a smooth ITL. For batch summarization or an agent tool JSON blob the user never sees incrementally, stream less and optimize **goodput**.

---

# **3. The Metrics That Matter**

---

## **3.1 Definitions (say these exactly)**

| Metric | Definition | Typical chat target | What actually moves it |
|---|---|---|---|
| **TTFT** | Request accepted → first output token | p50 < 400 ms, p95 < 2 s | Queue + prefill ($N_{\text{in}}$) + cold start |
| **TPOT** | Mean time per output token after the first | 20–50 ms (~20–50 tok/s) | Model size, batch, quant, decode load |
| **ITL** | Inter-token latency (distribution of gaps) | Smooth; no 400 ms "stalls" | Scheduler preemption, chunked prefill, GC |
| **E2E latency** | Request → last token / `finish_reason` | $\text{TTFT} + \text{TPOT}\times(N_{\text{out}}-1)$ | **Output length** dominates |
| **Throughput** | Tokens/s **or** requests/s leaving the engine | Hardware-dependent | Batch size, utilization, prefix hit rate |
| **Goodput** | Tokens/s that **met the SLO** | The number you should optimize | Throughput that violates p95 is fake capacity |
| **Concurrency** | In-flight requests (prefill + decode) | Until KV or `max-num-seqs` binds | KV memory, not CPU |
| **Queue time** | Arrival → first scheduled token | << TTFT budget | Replica count, admission control |

$$
L_{\text{total}} \approx L_{\text{queue}} + \text{TTFT}_{\text{compute}} + \text{TPOT} \times (N_{\text{out}} - 1)
$$

Same identity as `49` §6.5. This doc is *why* TTFT and TPOT come from different kernels.

## **3.2 Throughput vs goodput (the trick question)**

A replica at 100% GPU util pumping 4k tok/s with p99 TTFT of 12 s has **high throughput and low goodput**. You optimized the wrong objective. Admission control (reject or shed) plus another replica can *lower* raw tok/s and *raise* SLO-compliant tok/s.

> Interview line: *"I would rather run at 70% KV utilization with a p95 TTFT under two seconds than 95% KV and a queue that looks like a denial-of-service."*

## **3.3 What to put on the dashboard**

| Panel | Series |
|---|---|
| Latency | p50/p95/p99 TTFT, ITL, e2e — **split by prefill vs decode vs queue** |
| Load | `num_running`, `num_waiting`, KV cache % used, GPU SM util, HBM used |
| Quality-adjacent | `finish_reason` (stop / length / abort), prefix-cache hit rate, speculative **acceptance rate** |
| Cost | tok/s per GPU-hour, $ per 1M tokens (input vs output separately) |

Prefix-cache hit rate and speculative acceptance rate are **serving** metrics, not app metrics. App traces (LangSmith) will not show them unless you export engine stats.

---

# **4. KV-Cache Math**

---

## **4.1 What is stored**

For every layer, every token seen so far, you keep **keys and values** so you do not recompute attention over the whole prefix at every decode step. That cache **grows linearly with sequence length and with concurrency**.

Per **token**, per **request**:

$$
\text{bytes/token} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times b
$$

| Symbol | Meaning |
|---|---|
| $2$ | Key **and** value |
| $L$ | Number of layers |
| $n_{\text{kv}}$ | **KV heads** (not query heads — GQA/MQA shrink this) |
| $d_{\text{head}}$ | Head dimension |
| $b$ | Bytes per element (FP16/BF16 $= 2$, FP8 $= 1$, INT8 $= 1$, INT4 $\approx 0.5$) |

Total live cache:

$$
\text{KV} = (\text{bytes/token}) \times \sum_{\text{requests}} \text{seqlen}_i
$$

## **4.2 A number you should be able to compute on a whiteboard**

**Llama-3-class 8B, GQA:** $L=32$, $n_{\text{kv}}=8$, $d_{\text{head}}=128$, FP16 ($b=2$):

$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} \approx 128 \text{ KB per token}
$$

| Context | KV for **one** request (FP16) |
|---|---|
| 2k tokens | ~256 MB |
| 8k tokens | ~1 GB |
| 32k tokens | ~4 GB |

A 24 GB L4 holding a 4-bit 8B (~4–5 GB weights) has **~16–18 GB** left for KV + activations + fragmentation. At 8k average context that is on the order of **a dozen concurrent** full-context chats — not hundreds. **Concurrency is a memory problem**, not a "how many CPU threads" problem.

**GQA is a serving feature.** Llama-2-7B (full MHA, 32 KV heads) is ~4× the KV of Llama-3-8B (8 KV heads) at the same $d_{\text{head}}$. That is why "7B vs 8B" is not the interesting comparison — **KV heads** are.

## **4.3 Why naive allocation dies**

If you **pre-reserve** `max_model_len` × `max_batch` contiguous KV, you pay for the worst-case prompt on every slot. A 2k-token chat sitting in a 32k slot wastes ~15/16 of that reservation. Fragmentation + reservation is why older HF `generate()` servers OOM at low concurrency. **Paging** (next section) is the fix.

## **4.4 KV quantization**

Weights in INT4 and KV in FP16 is a common first deploy — then KV becomes the limiter. FP8/INT8 KV (and increasingly 4-bit KV with outlier handling) cuts the working set ~2–4× with a quality hit you must **eval** (doc `49`), especially on long-context retrieval. Spoken line: *"I quantize KV only after I have a long-context golden set; needle-in-haystack and citation tasks die first."*

---

# **5. PagedAttention**

---

## **5.1 The OS analogy (use this)**

vLLM's PagedAttention treats the KV-cache like **virtual memory**:

| OS | vLLM |
|---|---|
| Physical pages | **KV blocks** (e.g. 16 tokens × all layers) |
| Virtual address space | A request's logical sequence |
| Page table | Block table: logical token range → physical block |
| Internal fragmentation | Last partially filled block per request (bounded) |
| External fragmentation | Almost gone — any free block can be used by any request |
| Copy-on-write / fork | Shared prefix blocks for beam search or parallel samples |

You no longer allocate one giant `max_len` tensor per request. You allocate **blocks on demand** as tokens are generated, and you **free** them when the request finishes.

## **5.2 What it buys you**

1. **Higher concurrency** on the same GPU — the usual 2–4× vs reserved-max-len allocation.
2. **Prefix sharing** — identical leading blocks (system prompt, tool schemas, retrieved-and-pinned corpus) are stored once and pointed at by many requests. That is **automatic prefix caching** when enabled.
3. **No giant memcpy** when a sequence grows — append a block.

## **5.3 What it does *not* do**

- It does not make decode compute-bound. You still stream weights.
- It does not remove the **attention compute** that is $O(N)$ per new token (or $O(N^2)$ in prefill).
- Block size is a trade-off: too small → more table overhead; too large → more internal fragmentation. Defaults are fine until you measure.

## **5.4 RadixAttention (SGLang) — same idea, tree-shaped**

SGLang's **RadixAttention** keeps a prefix **tree** of KV blocks across requests. Agents that resend the same system prompt + tool JSON + last-k turns hit the tree constantly. If the interviewer says "why would anyone use SGLang over vLLM?": *shared-prefix / multi-turn / constrained-decoding workloads.* vLLM's prefix cache covers a large fraction of the same value in 2025–2026; the honest answer is "measure prefix hit rate on *your* traffic."

---

# **6. Continuous Batching**

---

## **6.1 Static batching (the thing you must not ship)**

Wait until $B$ requests arrive (or a timeout), run them together until **the longest** finishes. Short requests pad and wait. Throughput looks good on a benchmark script. Interactive p95 looks like a queueing disaster.

## **6.2 Iteration-level (continuous) batching**

At **every decode step**:

1. Drop sequences that just emitted EOS or hit `max_tokens`.
2. Admit waiting requests (their **prefill**, possibly chunked).
3. Run one fused step on the new mixed batch.

The GPU stays full. A 20-token "yes" does not wait for a 800-token essay in the same batch.

```
 time →
 static:     [ req A .......... ][ req B .... ][ req C ............... ]
             GPU idle-ish while A is the only one left in the batch

 continuous: A A A A A|
             B   B B B B B B|
             C C C C C C C C C C C|
             (new D starts the instant a slot + KV blocks exist)
```

This is the default in vLLM, TGI, SGLang, TensorRT-LLM in-flight batching. **Saying "we use batching" is not enough — say iteration-level / continuous.**

## **6.3 Prefill/decode interference**

A newly admitted 12k-token prefill can stall ITLs for everyone. Mitigations: chunked prefill, separate **prefill replicas** vs **decode replicas** (disaggregated serving — emerging production pattern for big fleets), priority queues (interactive vs batch).

---

# **7. vLLM — The Engine You Listed**

---

## **7.1 Two APIs you must not confuse**

| API | Use | Blocking? |
|---|---|---|
| `LLM` + `llm.generate()` | Offline / batch jobs, notebooks, eval sweeps | Yes — a batch in, a batch out |
| `AsyncLLMEngine` / OpenAI HTTP server | Online product traffic | No — stream, abort, continuous batch |

If you only show `LLM()`, an interviewer will ask how you serve HTTP. The product path is the **OpenAI-compatible server**.

```python
# Offline / eval — fine for a notebook, not a user-facing API
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    quantization="awq",          # or gptq / fp8 / bitsandbytes
    dtype="half",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enable_prefix_caching=True,
    tensor_parallel_size=1,
)
params = SamplingParams(temperature=0.2, max_tokens=256, stop=["</s>"])
outputs = llm.generate(["Cite the contraindication in the retrieved note."], params)
print(outputs[0].outputs[0].text)
```

```bash
# Online — this is what "we serve with vLLM" means
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --quantization awq \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 64 \
  --port 8000

# Newer CLI equivalent:
# vllm serve mistralai/Mistral-7B-Instruct-v0.3 --quantization awq ...
```

Clients use the same OpenAI SDK you already use for Groq/OpenAI — swap `base_url` to `http://vllm:8000/v1`. That is how you A/B a provider vs self-host without rewriting the app (Bilbo/AI Cargo already think in "provider behind a URL").

```python
# Same client you already use — only the base_url changes
from openai import OpenAI

client = OpenAI(base_url="http://vllm:8000/v1", api_key="not-needed")
stream = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "List three contraindications. JSON only."}],
    max_tokens=200,
    temperature=0.0,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

If you embed vLLM in a FastAPI process instead of the standalone server, use `AsyncLLMEngine` so one request's decode does not block another:

```python
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        quantization="awq",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )
)

async def generate(request_id: str, prompt: str):
    params = SamplingParams(temperature=0.0, max_tokens=256)
    async for out in engine.generate(prompt, params, request_id=request_id):
        yield out.outputs[0].text
        # engine.abort(request_id)  # user navigated away — free KV blocks
```

**Abort is a serving feature.** If the user hits stop or the gateway times out and you do not abort, those KV blocks stay allocated until `max_tokens`. That is a silent concurrency leak.

## **7.2 Flags you should be able to defend**

| Flag | What it actually does | If you set it wrong |
|---|---|---|
| `--gpu-memory-utilization 0.90` | Fraction of free GPU memory vLLM may take for the **KV block pool** after weights load | 0.98 → OOM on fragmentation / CUDA graphs; 0.6 → you left capacity on the table |
| `--max-model-len` | Hard cap on prompt + generation | Too high → fewer concurrent slots; too low → RAG prompts 400 |
| `--max-num-seqs` | Cap on concurrent sequences | Scheduler admission control; pair with KV % |
| `--enable-prefix-caching` | Hash leading token blocks; reuse KV | Worthless if every request has a unique dump *first* — **put the stable prefix first** (see `52` / `16`) |
| `--enable-chunked-prefill` | Interleave long prefills with decode | Off: one fat RAG prompt stalls the replica |
| `--tensor-parallel-size` | Shard each layer across $N$ GPUs | Needs fast interconnect; $N$ must divide head counts |
| `--quantization awq\|gptq\|fp8` | Load a pre-quantized checkpoint | Must match the checkpoint format; see `13` |
| `--enable-lora` / `--max-loras` | Multi-adapter serving on one backbone | Extra GPU memory per active adapter |
| `--speculative-model` + `--num-speculative-tokens` | Draft/target speculative decoding | See §9; if acceptance is low you *lose* speed |
| `--disable-log-stats` | Less log spam | You still want Prometheus metrics in prod |

## **7.3 Prefix caching — the highest-leverage app change**

Automatic prefix caching only hits if the **token prefix is byte-for-byte identical**.

**Do this:**
1. System prompt
2. Tool schemas
3. Static RAG corpus / policy text
4. **Then** the user-specific retrieved chunks and the question

**Don't do this:** put a timestamp, request ID, or shuffled chunk list at the top. You just flushed the cache for every call.

This is the same advice as `52` (put stable content first) and `16` (context engineering). Serving makes it *quantitatively* real: a 2k-token system prompt that hits cache is **free prefill** after the first request in that block.

## **7.4 LoRA on a shared backbone**

One 8B + many small LoRAs (per tenant, per task) is how you avoid a GPU per fine-tune. vLLM loads adapters dynamically. Limits: `--max-loras`, `--max-lora-rank`, extra KV is still per *request* not per adapter. Spoken line: *"I serve one backbone and hot-swap LoRAs; I do not dedicate a replica per customer model unless isolation or latency requires it."*

## **7.5 What I would measure on a vLLM replica (first day)**

```python
# Conceptual — engine exposes Prometheus at /metrics on the HTTP server
# Names drift slightly across versions; know the *ideas*:
#   vllm:num_requests_running
#   vllm:num_requests_waiting
#   vllm:gpu_cache_usage_perc
#   vllm:time_to_first_token_seconds
#   vllm:e2e_request_latency_seconds
#   vllm:prefix_cache_hits / misses
```

Alert on **waiting > 0 for more than N seconds** and **KV cache > 80–85%** — those two predict a TTFT cliff better than GPU SM util (which can look "healthy" while you are KV-bound).

---

# **8. Engine Comparison**

---

You do not need to have run all five. You need a **decision table**.

| Engine | Best at | Weak / costly part | When I pick it |
|---|---|---|---|
| **vLLM** | General OpenAI-API serving; PagedAttention; prefix cache; LoRA; speculative; huge community | Not always #1 on a single NVIDIA graph vs TRT-LLM | **Default self-host** for an AI engineer interview |
| **TGI** (Hugging Face) | HF ecosystem, similar continuous batching | Fewer exotic scheduler knobs than vLLM/SGLang in some versions | Team already lives in HF Inference Endpoints |
| **TensorRT-LLM** | Peak NVIDIA latency/throughput after **compile** | Engine rebuild on model/shape change; ops-heavy | Stable model, NVIDIA-only, latency SLO is money |
| **SGLang** | Radix prefix tree, structured / constrained decoding, agent loops | Slightly less "default" mindshare than vLLM | High prefix reuse, JSON-grammar, multi-turn agents |
| **Ollama** | Local DX, llama.cpp, one-command laptop/server | Not a multi-tenant SLO engine; weaker continuous-batch story | **Bilbo / AI Cargo / demo** — honest current stack |
| **llama.cpp** | CPU, Apple Silicon, edge | Not your 1k-QPS GPU farm | Edge, air-gapped CPU boxes |
| **Triton + TRT-LLM** | Multi-model ensemble, NVIDIA ops standard | YAML/complexity | Platform team already standardized on Triton |
| **Ray Serve + vLLM** | Python autoscaling, multi-replica in one app | Another control plane | You already run Ray for training/batch |

**Ollama vs vLLM (you will be asked, because your projects use Ollama):**

| | Ollama | vLLM |
|---|---|---|
| Job | Developer / single-box / demo | Concurrent users, SLOs, prefix reuse |
| Batching | Modest | Continuous + paged KV |
| API | Simple local | OpenAI-compatible at scale |
| Quant | GGUF 4-bit (Bilbo) | AWQ/GPTQ/FP8/GGUF (varies by version) |
| Honest line | *"I shipped Bilbo on Ollama because one box, one model, sub-5s e2e was the constraint."* | *"If I had to put the same 7B behind 50 concurrent clinicians, I would move the replica to vLLM and keep the app's OpenAI client."* |

---

# **9. Speculative Decoding**

---

This is the other resume keyword. Treat it like you treat distillation on Jet2: **you can explain the algorithm and the catch**.

## **9.1 The idea**

Decode is memory-bound: the **target** (the model you actually want) spends most of its time loading weights to produce **one** token. A cheap **draft** model (or draft heads) proposes $\gamma$ tokens. The target then **verifies those $\gamma$ tokens in one forward pass** (parallel over the draft tokens). You accept the longest prefix where draft and target agree, then **sample the next token from the target** at the first mismatch.

```
 draft (small / heads):   t1  t2  t3  t4     proposed
 target verify (1 step):  ✓   ✓   ✗   —     accept t1,t2; reject t3
 target resample:         t3'                 from target distribution
 emitted this iteration:  t1 t2 t3'           (often 2–4 tokens, one target forward)
```

## **9.2 Why it can be lossless**

If, on rejection, you **sample from the target model's distribution** (the Leviathan / Chen correction), the output distribution is the same as running the target alone. You are not "approximating" the target — you are **skipping work** when the draft is lucky.

> Say this out loud: *"Speculative decoding is a performance optimization, not a quality trade-off, when you use rejection sampling against the target. If I greedy-accept draft tokens without the correction, that is a different, lossy algorithm — I would not call that speculative decoding in the paper sense."*

## **9.3 Acceptance rate is the only hyperparameter that matters**

Let $\alpha$ be the probability a draft token is accepted (simplified, i.i.d. for the interview). With $\gamma$ speculated tokens, expected tokens per target-forward is about

$$
\mathbb{E}[\text{tokens per iteration}] \approx \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

(plus the bonus resampled token). Speedup is that factor **times** (target-forward time) / (target-forward + draft cost). If the draft is 50% of the target's cost and $\alpha$ is 0.4, you can **lose**.

| $\alpha$ | Meaning | What you do |
|---|---|---|
| 0.8–0.9 | Draft is very aligned (same family, same prompt style, or Medusa/EAGLE trained for this target) | Speculative is almost free speed |
| 0.5–0.7 | Typical small-draft + 7B/8B target on mixed traffic | Tune $\gamma$ (3–5); measure |
| < 0.4 | Draft is the wrong size/family or the task is high-entropy (code, creative) | **Turn it off** |

**Measure `acceptance_rate` in prod.** It is as important as prefix-cache hit rate.

## **9.4 Families of drafters**

| Method | Draft source | Interview one-liner |
|---|---|---|
| **Draft model** | Smaller LM in the same family (e.g. 1B / 3B drafting 70B) | Classic; two models in HBM |
| **Medusa** | Extra decoding **heads** on the target (same backbone, predict +1,+2,+3) | No second model; heads are cheap; trained |
| **EAGLE / EAGLE-2** | Autoregressive head that uses the target's **features** (hidden states), not just tokens | Higher $\alpha$ than naive Medusa on many benches |
| **Prompt lookup / n-gram** | Copy from the prompt (code, RAG — tokens often reappear) | Great for repetition-heavy tasks; zero extra model |
| **Lookahead** | Jacobi / n-gram Jacobi decoding | Related family; know the name |

vLLM exposes draft-model speculation (`--speculative-model`, `--num-speculative-tokens`) and has been adding Medusa/EAGLE-style paths. You do not need the flag memorized — you need **lossless verify + $\alpha$**.

## **9.5 When I would *not* use it**

- Prefill-heavy RAG with 20-token answers: you already spent the time in prefill; speculation helps **decode**.
- Tiny 7B target on an L4 where a 1B draft steals KV you needed for concurrency.
- High-temperature creative generation (low $\alpha$).
- You have not measured $\alpha$ on **your** traces. Enabling the flag because it is on the resume is how you ship a regression.

## **9.6 Tie to projects without lying**

- **Jet2 70B→7B distillation** is a *training-time* way to get a cheaper target. Speculative is an *inference-time* way to keep the big target and skip steps. Complementary, not the same.
- **Bilbo sub-5s**: first levers are retrieval trim, 4-bit weights, short answers — not speculative decoding on Ollama.
- If asked "have you enabled this in vLLM?": *"I can configure draft/target and I know I would gate it on acceptance rate. I have not A/B'd it on Bilbo traffic."*

---

# **10. Quantization for Serving**

Depth lives in `13_quantization.md`. Serving-only points:

| What you quantize | Typical | Serving effect | Risk |
|---|---|---|---|
| **Weights** | AWQ / GPTQ INT4, FP8 | Fits GPU; faster decode (fewer bytes) | Quality — run the same eval gate as a model swap (`49`) |
| **Activations** | FP8 / INT8 smoothquant | More speed on tensor cores | Outliers; calibrate |
| **KV-cache** | FP8 / INT8 / 4-bit | **Concurrency** (the usual win after weights are already 4-bit) | Long-context quality |

vLLM load path (same as `13` §7.5): `--quantization awq|gptq|fp8` on a **matching** checkpoint. Do not say "I pass `quantization='awq'` on an FP16 repo and it just works" — you need an AWQ artifact (or a path that quantizes on load, which is slower and different).

**Memory back-of-envelope (8B):**

| | Weights | KV @ 8k × 8 concurrent | Fits |
|---|---|---|---|
| FP16 | ~16 GB | ~8 GB | 24 GB is tight |
| INT4 weights, FP16 KV | ~4.5 GB | ~8 GB | Comfortable on L4 24 GB |
| INT4 weights, FP8 KV | ~4.5 GB | ~4 GB | Room for longer context or more users |

---

# **11. Tensor, Pipeline, and Expert Parallelism**

---

One GPU is simpler. You add parallelism when **the model does not fit** or **decode bandwidth** needs more HBM readers.

| Parallelism | What is split | Communication | When |
|---|---|---|---|
| **TP (tensor)** | Each linear / attention is sharded across $N$ GPUs | All-reduce **every layer**, every token | Default for 70B on 2–8 GPUs; wants NVLink |
| **PP (pipeline)** | Consecutive layers on consecutive GPUs | Point-to-point activations; **pipeline bubble** | Huge models; combine with TP |
| **DP (data)** | Full replica per GPU; different requests | None on the forward (until you aggregate metrics) | Scale **QPS**, not model size |
| **EP (expert)** | MoE experts on different GPUs | All-to-all token dispatch | Mixtral-class / frontier MoE |
| **SP / context parallel** | Sequence dimension | All-to-all for attention | Very long context |

**Interview trap:** "I'll tensor-parallel a 7B across 4 GPUs to go faster." Often **slower** — communication > savings, and you just used 4 GPUs that could have been **4 DP replicas** (4× QPS). TP a 7B only if you are doing a demo or a weird memory trick. **DP for QPS, TP/PP for models that do not fit.**

vLLM: `--tensor-parallel-size 2` on a 70B. Pipeline parallel exists but TP is the common interview path.

---

# **12. Kubernetes GPU Serving**

---

Cross-ref `23_cloud_mlops_deployment.md` for general K8s. This section is **GPU + LLM-specific**.

## **12.1 One pod, one GPU (default)**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-mistral7b
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - --model
            - mistralai/Mistral-7B-Instruct-v0.3
            - --quantization
            - awq
            - --enable-prefix-caching
            - --gpu-memory-utilization
            - "0.90"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 90   # weights must load
            periodSeconds: 10
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-L4
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

**Do not HPA on CPU.** A waiting vLLM replica can have sleepy CPUs and a full KV. Scale on:

- `num_requests_waiting`
- `gpu_cache_usage_perc` (scale out **before** 85%)
- Gateway queue depth
- p95 TTFT (custom metric)

## **12.2 KEDA sketch**

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-keda
spec:
  scaleTargetRef:
    name: vllm-mistral7b
  minReplicaCount: 1          # 0 if you accept cold start
  maxReplicaCount: 8
  cooldownPeriod: 300         # weights are expensive to reload
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        query: sum(vllm_num_requests_waiting)
        threshold: "4"
```

## **12.3 Cold start is the LLM-specific tax**

| Step | Order of magnitude |
|---|---|
| Schedule GPU node (if cluster is dry) | seconds to **minutes** (and $) |
| Pull 10–20 GB image | minutes first time |
| Load 4-bit 7B into HBM | 10–40 s |
| Load 70B TP=2 | minutes |

Scale-to-zero (KServe / KEDA `minReplicaCount: 0`) is fine for **dev** and **async batch**. For a clinician-facing RAG with a sub-5s SLO (**Bilbo**), you keep a **warm min replica**. Spoken line: *"I do not scale the interactive path to zero. I scale the batch path to zero."*

## **12.4 GPU sharing: time-slicing vs MIG vs MPS**

| Mechanism | Isolation | When |
|---|---|---|
| **1 pod / 1 GPU** | Best | Production vLLM (default) |
| **MIG** (A100/H100) | Hardware slices | Several small models, hard isolation |
| **MPS / time-slicing** | Soft; noisy neighbor | Dev clusters; **not** two latency-sensitive vLLMs |
| **Multi-model one engine** (vLLM LoRA, Triton ensemble) | Logical | Many adapters, one backbone |

Two vLLM processes time-slicing one L4 will wreck both ITLs. Don't.

## **12.5 KServe vs Ray Serve vs "just a Deployment"**

| Stack | You get | Cost |
|---|---|---|
| **Deployment + Service + HPA/KEDA** | Predictable, interview-default | You write the YAML |
| **KServe `InferenceService`** | Scale-to-zero, canary, predictor/transformer split | Another CRD to debug |
| **Ray Serve** | Bind vLLM in Python, replica autoscaling | Ray cluster ops |
| **Triton** | Multi-framework, NVIDIA standard | Model-repo conventions |

For most AI-engineer interviews: **Deployment + GPU request + KEDA on waiting-requests + a warm min**. Name KServe/Ray as options, don't pretend you need them for a 7B.

---

# **13. Capacity Planning & Cost**

---

## **13.1 The sizing loop (say it as a procedure)**

1. **Pick the model + quant** that passes eval (`49` gate).
2. **Compute bytes/token** (§4). Check GQA.
3. **Pick max context** you will actually send (not the brochure 128k).
4. **KV budget** = `gpu_memory_utilization × HBM − weights − fragmentation`.
5. **Max concurrency** $\approx$ KV budget / (bytes/token × typical seqlen). That is your hard cap — `max-num-seqs` should sit near it.
6. **Measure** TTFT/ITL/tok/s at 25%, 50%, 75%, 90% of that cap on **production-shaped** prompts (RAG prefixes, tool JSON), not `Write a poem`.
7. **Replicas** = `target_QPS × E2E_seconds / concurrency_at_SLO`.
8. **Goodput check:** if p95 TTFT blows before KV is full, you are compute/queue bound — add replicas earlier.

## **13.2 Worked example (whiteboard)**

Interactive RAG, Mistral-7B AWQ on L4 24 GB, typical 3k prompt + 200 completion, p95 TTFT < 2 s.

- Weights ~4.5 GB; KV ~128 KB/token × 3.2k × $C$ concurrent.
- $C = 16$ → KV $\approx 6.5$ GB. Fits.
- Measure: at $C=12$ you still hit TTFT; at $C=20$ queueing starts. **Operate at 12.**
- If you need 36 in-flight at SLO → **3 replicas**, not one replica with `max-num-seqs 36`.

I will not invent a tok/s number I did not measure. Directionally, a 7B-4bit on an L4 is tens of tokens/s **per request** at low batch and much higher **aggregate** tok/s at batch — those are different numbers; do not mix them.

## **13.3 Cost: self-host vs API**

| | API (Groq / OpenAI) | Self-host (L4 / A10) |
|---|---|---|
| **Unit** | $ / 1M tokens (in ≠ out) | $ / GPU-hour whether you are busy or not |
| **Wins when** | Spiky, low duty cycle, you lack GPU ops | Steady high utilization, huge shared prefixes, data cannot leave VPC |
| **Hidden cost** | Retries, output verbosity, agent loops (`49` §6.6) | Idle GPUs, over-provisioned max-len, failed speculative, idle TP shards |

Break-even sketch: if a GPU is $1/hr and you sustain 2M output-ish tokens/hr at your measured tok/s, compare to the provider's output price. **Idle nights kill self-host economics** — that is when scale-to-zero or a smaller warm pool matters.

**Prefix cache is a cost lever, not just a latency lever.** A 70% prefix hit on a 4k system prompt is 70% less prefill compute on every subsequent request in that replica.

## **13.4 Disaggregated prefill / decode (know the phrase)**

Large 2025–2026 fleets split **prefill nodes** (compute-heavy, short) from **decode nodes** (memory-heavy, long). KV is transferred over the network (or a cache fabric). You do not need to have built this. You need to say: *"If prefill and decode fight on one replica, the next architecture is disaggregation — I would not start there for a 7B."*

## **13.5 Debugging playbook (symptom → cause → first move)**

Use this at a whiteboard. Always split **queue vs prefill vs decode** before you touch flags.

| Symptom | Likely cause | First move |
|---|---|---|
| p95 TTFT high, p50 fine | Queue / admission / one fat prefill | Look at `num_waiting`, chunked prefill, add a replica — not `temperature` |
| TTFT high for everyone | Prompt length or cold start | Histogram $N_{\text{in}}$; check if a deploy just recycled pods |
| ITL has 200–400 ms "holes" | Prefill/decode interference, or a too-large chunk | Enable/tune chunked prefill; isolate batch jobs |
| OOM after traffic ramps, not at start | KV pool filled; max-len too generous | Lower `max-model-len` to p99 prompt; drop `max-num-seqs`; abort cancelled jobs |
| High tok/s, angry users | You optimized throughput, not goodput | Cap concurrency; SLO on p95 TTFT |
| Prefix cache hit ≈ 0 | Unstable leading tokens | Move system prompt / tools **first**; stop putting timestamps on top |
| Speculative slower than baseline | $\alpha$ low or draft too fat | Disable; measure $\alpha$ by task; try n-gram/PLD on RAG/code |
| GPU SM util low, KV 90% | Memory-bound, not "idle GPU" | That is decode + full cache — more replicas or shorter context, not "the GPU is broken" |
| 502s during rollout | Readiness probe too eager | Probe `/health` only after weights are loaded; surge / maxUnavailable 1 |
| Quality drop after `--quantization` | You shipped a new model | Roll back image; run the `49` eval gate — serving flags are model changes |

**Order of levers (cheap → expensive):** shorten output → stable prefix → chunked prefill → quantize weights (if not already) → lower max-len → more replicas → KV quant → speculative (if $\alpha$ high) → bigger GPU / TP.

---

# **14. Reference Architecture**

---

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION LLM SERVING                            │
│                                                                          │
│  Clients (web, agent, batch)                                             │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────┐                                 │
│  │ API gateway                         │  auth, tenant quota,            │
│  │  · pin model id                     │  timeout, retry,                │
│  │  · route: nano / 7B / frontier      │  shadow %, kill switch          │
│  └──────────────┬──────────────────────┘                                 │
│                 │                                                        │
│        ┌────────┴─────────┐                                              │
│        ▼                  ▼                                              │
│  Provider APIs      Self-host pool                                       │
│  Groq/OpenAI/       ┌─────────────────┐                                  │
│  Anthropic          │ vLLM replica × N │  PagedAttention                 │
│  (AI Cargo path)    │  prefix cache    │  continuous batch               │
│                     │  spec. decode?   │  /metrics → Prom                │
│                     └────────┬────────┘                                  │
│                              │                                           │
│                     K8s: GPU Deployment + KEDA                           │
│                     scale on waiting + KV%                               │
│                     minReplicas ≥ 1 if SLO is interactive                │
│                                                                          │
│  Observability (doc 49): traces, TTFT, cost, finish_reason,              │
│  plus engine: KV%, prefix hit, speculative α                             │
│                                                                          │
│  Eval gate (doc 49): treat engine flag changes (quant, spec decode,      │
│  max-len) like a model upgrade — shadow + golden set                     │
└──────────────────────────────────────────────────────────────────────────┘
```

**How this maps to your work without overselling:**

| Piece | What you have actually done | What this doc lets you say next |
|---|---|---|
| Local 7B | Bilbo: Ollama 4-bit Mistral, sub-5s | "The next scale step is vLLM + prefix cache for the static clinical prompt." |
| Multi-provider | AI Cargo: 4 providers + zero-LLM | "Gateway routing is the same; vLLM is one more `base_url`." |
| Distillation | Jet2 70B→7B | "Training-time cheaper target; speculative is inference-time keep-the-big-model." |
| MLOps | Fibe MLflow; ATCS Docker/CI | "Same discipline: pin the image, pin the weights, gate on eval, Prometheus on the replica." |

---

# **15. Interview Questions with Strong Answers**

Practice these **out loud**. End with how you would **measure**.

---

## **Q1: "What is the difference between prefill and decode?"**

**Strong answer:**

> "Prefill runs the whole prompt in parallel — one big compute-bound pass — writes the initial KV-cache, and emits the first token. That is TTFT. Decode then walks one token at a time: each step reads the weights and the growing KV, so it is memory-bandwidth bound. Total latency is queue plus prefill plus time-per-token times remaining output tokens. A 4k-token RAG prompt hurts TTFT; a 600-token answer hurts e2e even if TTFT was fine. I would never quote a single 'latency' number without saying which phase."

---

## **Q2: "Write the KV-cache memory formula."**

**Strong answer:**

> "Bytes per token is two, for key and value, times layers, times KV heads — not query heads — times head dim times bytes per element. Total cache is that times the sum of sequence lengths in flight. For an 8B Llama-3-style GQA model in FP16 that is about 128 KB per token, so 8k context is about a gigabyte per request. GQA matters more than '7B vs 8B.' I size concurrency from leftover HBM after weights, not from CPU cores. Then I measure, because fragmentation and activations eat the theoretical leftover."

---

## **Q3: "What is PagedAttention?"**

**Strong answer:**

> "It is virtual memory for the KV-cache. Instead of reserving max-sequence-length times max-batch as one contiguous tensor, vLLM allocates fixed-size KV blocks on demand and maps them with a block table. You stop paying for 32k slots on a 2k chat, external fragmentation drops, and identical prefixes can share blocks. That is why prefix caching is cheap to implement on top. It does not change the fact that decode is memory-bound — it just stops you OOMing from dumb reservation."

---

## **Q4: "Continuous batching vs dynamic batching vs static batching."**

**Strong answer:**

> "Static: wait for a full batch, run until the longest sequence finishes. Dynamic in older servers often meant 'batch what is waiting at the start of generate().' Continuous, or iteration-level, re-packs the batch **every decode step** — finished sequences leave, new prefills enter, possibly chunked. That is what vLLM, TGI, and TRT-LLM in-flight batching do. If someone says they 'batch requests' I ask whether a 20-token reply waits on an 800-token reply. If yes, it is not continuous."

---

## **Q5: "How does vLLM compare to TGI, TensorRT-LLM, SGLang, and Ollama?"**

**Strong answer:**

> "vLLM is my default self-host: OpenAI API, PagedAttention, prefix cache, LoRA, speculative, huge community. TGI is the Hugging Face-native cousin. TensorRT-LLM wins peak NVIDIA latency after you compile an engine — I pay ops cost on every shape or model change. SGLang's RadixAttention is excellent when many requests share a prefix tree — agents, tools, multi-turn. Ollama is what I actually used on Bilbo: best local DX, llama.cpp, not a multi-tenant SLO engine. If I had to put that same 7B behind dozens of concurrent users I would keep the OpenAI client and point it at vLLM."

---

## **Q6: "Explain speculative decoding. Is it lossy?"**

**Strong answer:**

> "A cheap drafter proposes several tokens. The target verifies them in one forward. You accept the matching prefix and sample the first disagreement from the **target** distribution. With that correction it is lossless — same distribution as the target alone. Speedup is expected accepted tokens per target-forward, minus draft cost. The number I watch is acceptance rate alpha. Below about 0.4, or if the draft is expensive relative to the target, I turn it off. Medusa and EAGLE are ways to draft without a second full model. I would not enable it on a 20-token-answer RAG where prefill already dominates, and I would not claim I A/B'd it on Bilbo."

---

## **Q7: "How do you set gpu-memory-utilization?"**

**Strong answer:**

> "It is the fraction of GPU memory vLLM is allowed to take for the KV block pool after the weights are loaded — not 'use 90% of the GPU for weights.' Too high and you OOM on CUDA graphs or a slightly longer prompt. Too low and you cut concurrency for no reason. I start at 0.9, watch KV percent and OOM, and leave headroom on the first deploy. The real cap I operate to is KV around 80 percent so a burst does not cliff TTFT."

---

## **Q8: "Why did my vLLM box OOM even though nvidia-smi showed free memory at start?"**

**Strong answer:**

> "Three usual reasons. First, utilization is reserved up front for the block pool — later requests grow into it, then a long prompt needs more blocks than free. Second, fragmentation plus activations plus CUDA graphs sit outside the mental 'weights + KV' model. Third, someone set max-model-len to 128k 'because the card says so' and the pool is sized for a nightmare prompt. I lower max-model-len to the p99 prompt I actually send, enable paging — already on — and add admission control instead of letting waiting requests allocate until death."

---

## **Q9: "How would you serve 50 concurrent users on a 7B?"**

**Strong answer:**

> "I would not assume one GPU. I compute bytes per token, pick a real max context — say 4k, not 128k — estimate KV, and see concurrency at SLO, not at OOM. A 4-bit 7B on an L4 often holds on the order of ten-plus 4k chats, not fifty. Fifty in-flight at a two-second TTFT is several replicas behind a gateway, KEDA on waiting-requests, prefix cache on the shared system prompt, and chunked prefill so one fat context cannot stall the others. I would load-test with production-shaped prompts before I quote a replica count."

---

## **Q10: "Tensor parallel a 7B across four GPUs to make it faster?"**

**Strong answer:**

> "Usually no. Tensor parallel helps when the model does not fit or when you need the aggregate memory bandwidth of several GPUs on one giant model. A 7B already fits. All-reduce every layer on every token can make TP **slower**, and I just burned four GPUs that could have been four data-parallel replicas — four times the QPS. I TP a 70B. I DP a 7B."

---

## **Q11: "How do you autoscale vLLM on Kubernetes?"**

**Strong answer:**

> "Not on CPU. I export engine metrics — running, waiting, KV cache percent, TTFT — and KEDA or HPA on waiting-requests and KV percent, with a cooldown of minutes because loading weights is expensive. Interactive paths get minReplicas at least one; I do not scale-to-zero a clinician chat. Batch and eval sweep can go to zero. Readiness probes have to wait for the model to be in HBM or we will 502 during every rollout. Cold start is node plus image plus weight load — I plan for that in the SLO, I do not pretend it is a Node.js deploy."

---

## **Q12: "Prefix caching is not helping. Why?"**

**Strong answer:**

> "The cache keys on the token prefix. If I put a timestamp, a user id, or shuffled RAG chunks first, every request misses. I pin the system prompt, tool schemas, and any static policy text at the front, unchanged, and put retrieved chunks after. Then I look at the engine's prefix hit rate, not just app-level TTFT. If hit rate is high and TTFT is still bad, the miss is the long unique tail — that is a retrieval problem, not a vLLM problem."

---

## **Q13: "Throughput went up after I raised max-num-seqs. Users are angrier. What happened?"**

**Strong answer:**

> "I optimized throughput instead of goodput. More in-flight requests raise GPU utilization and tokens per second, and they also raise queueing and KV pressure, which blows p95 TTFT. I would drop concurrency until p95 is inside the SLO, then add replicas. Admission control — 429 or a shed-to-batch queue — is kinder than a 15-second first token."

---

## **Q14: "How does this relate to Bilbo's sub-5s RAG?"**

**Strong answer:**

> "Sub-5s is end-to-end: retrieve, rerank, generate, cite. Generation is one slice. On Ollama I already cut that slice with a 4-bit 7B and short structured answers. The serving levers I would apply next on a vLLM replica are prefix-cache the clinical system prompt, cap max tokens, chunk prefills so one long note does not stall the other clinicians, and measure TTFT versus retrieval time from traces before I touch speculative decoding. I would not claim vLLM is what produced the sub-5s number today."

---

## **Q15: "Walk me through speculative decoding like I'm a staff engineer who thinks it's fake."**

**Strong answer:**

> "Decode is bottlenecked on reading weights. If I can get the target to score five draft tokens in one pass, I paid one memory-bound forward for several tokens whenever the draft is right. Verification plus target-resample keeps the distribution honest. Your skepticism should be empirical: show me alpha on our traces and the draft's extra HBM. If alpha is 0.35 on support-ticket prose, I agree it's fake for us. If it's 0.85 on a Medusa head trained for our JSON schema, it's one of the few free lunches left."

---

## **Q16: "Medusa vs a small draft model."**

**Strong answer:**

> "A small draft is a second stack in memory and a second scheduler problem — simple to understand, good when a 1B sits next to a 70B. Medusa attaches extra heads to the target so you don't load another model; you do need those heads trained. EAGLE uses the target's hidden states and usually accepts more tokens. I pick draft-model when I already have a distilled little sibling — that's the Jet2 intuition — and Medusa/EAGLE when I want to keep one backbone."

---

## **Q17: "What is chunked prefill and when do you need it?"**

**Strong answer:**

> "A 20k-token prefill is a long compute job. If I run it atomically, every other request's inter-token latency stalls. Chunked prefill splits that prompt and interleaves chunks with decode steps. I need it the moment I have mixed traffic — interactive chat plus occasional huge RAG dumps — on the same replica. If I can isolate batch summarization on another pool, I need it less."

---

## **Q18: "How do you A/B a serving change — AWQ vs FP16, or speculative on vs off?"**

**Strong answer:**

> "Like a model upgrade in doc 49, because quality can move. Offline: golden set, paired, plus a long-context / citation slice if I touched KV quant. Shadow: same prompts, new engine flags, compare TTFT, ITL, tok/s, **and** judge or task metrics. Canary a percentage of traffic with a kill switch. I never A/B speculative on throughput alone. And I pin the image digest and the weight revision so I can roll back."

---

## **Q19: "KServe or Triton or Ray — which one?"**

**Strong answer:**

> "For a single 7B with an OpenAI client, a Deployment, a GPU limit, and KEDA is enough — I would say that first so I don't sound like I collect control planes. KServe if the platform already uses InferenceService and I want canary plus scale-to-zero for non-interactive. Ray Serve if the rest of the stack is already Ray. Triton if the org standardized NVIDIA ensembles. The serving math — KV, continuous batch, prefix — does not change because the YAML CRD did."

---

## **Q20: "How do you handle a multi-LoRA SaaS on one GPU?"**

**Strong answer:**

> "One backbone, vLLM's LoRA manager, cap max active adapters and rank so HBM does not fragment into death. KV is still per request. Noisy-neighbor isolation is scheduling and quotas at the gateway, not a GPU per tenant, until a tenant's SLO or data-isolation contract forces a dedicated replica. I would eval each adapter like a prompt version — a bad LoRA is a silent quality incident."

---

## **Q21: "FP8 vs INT4 for production serving in 2026."**

**Strong answer:**

> "INT4 AWQ/GPTQ is still the default way I fit a 7B/13B on a cheap GPU with acceptable quality if I eval it. FP8 is the cleaner NVIDIA path on Hopper/Blackwell — better kernels, simpler than a 4-bit dance, often the sweet spot for 70B. I would not pick from a blog table; I would pick from the eval gate plus tok/s at the concurrency I need. KV in FP8 is a separate decision from weight FP8."

---

## **Q22: "What is goodput? Why do I care?"**

**Strong answer:**

> "Tokens per second that met the latency SLO. Raw throughput counts tokens that arrived after the user already bounced. If I maximize GPU util I can look like a hero on a Grafana panel and fail the product. I set the SLO — p95 TTFT for chat, p95 e2e for batch — and treat anything outside it as wasted capacity. That is how I decide to add a replica instead of raising max-num-seqs."

---

## **Q23: "How would you debug a sudden ITL spike — tokens arriving in bursts with gaps?"**

**Strong answer:**

> "That pattern is usually prefill/decode interference or a stop-the-world: a huge prefill landed, a GC, a weight offload, or the scheduler ran a chunk that was too big. I look at traces split by phase, KV percent, whether chunked prefill is on, and whether a batch job shared the replica. Fix: isolate batch, chunk prefills, lower max-num-seqs, or add a replica. I do not start by 'tuning temperature.'"

---

## **Q24: "You listed vLLM on your resume. Have you run it in production?"**

**Strong answer:**

> "I have not owned a multi-node vLLM fleet. I have served 7B models locally with Ollama — Bilbo's 4-bit Mistral, AI Cargo's local path — and I know vLLM's scheduler, paging, prefix cache, and speculative decode well enough to size a replica, write the Deployment, and pick flags. If you hired me to stand one up, day one is: pin a quantized checkpoint, OpenAI-compatible server, Prometheus on waiting and KV, a load test with our real prefixes, and an eval gate before I enable speculation. I would rather say that than invent a production war story."

---

## **Q25: "Design a serving stack for a medical RAG with citations and a sub-5s SLO."**

**Strong answer:**

> "Gateway with auth and a per-tenant budget. Retriever and reranker in front so the prompt is short — serving cannot fix a 30k dump. One warm vLLM replica of a 4-bit 7B, prefix-cache the system prompt and citation instructions, max tokens capped, structured output if the engine supports grammar. Traces with TTFT vs retrieval vs rerank. SLO is p95 e2e under 5s and p95 TTFT under ~1.5s after retrieval. Min replicas 1, scale on waiting, no scale-to-zero. Eval gate on groundedness and citation validity, same as Bilbo. Speculative decoding only if acceptance is high on short clinical answers — I would try retrieval trim first. Fallback: a provider API if the GPU dies, same pattern as AI Cargo's failover."

---

## **Q26: "What's the difference between time-per-output-token and tokens per second?"**

**Strong answer:**

> "TPOT is per-request: milliseconds between tokens for a user. Tokens per second is often **aggregate** across the batch. A replica can show 2k tok/s while each user sees 25 ms TPOT, or worse, 80 ms TPOT because the batch is huge. I never quote tok/s without saying per-request versus replica-aggregate. Product cares about per-request ITL; finance cares about aggregate tok per GPU-hour."

---

## **Q27: "How do MoE models change serving?"**

**Strong answer:**

> "Dense 7B is one set of weights every token. MoE only fires a few experts, so FLOPs per token drop but **you still need the experts in memory** unless you offload. Serving becomes expert-parallel: all-to-all dispatch, risk of expert imbalance, and a different KV story depending on the architecture. I would not treat Mixtral like 'a 12B that is free.' I would ask which experts are resident, what the interconnect is, and whether vLLM/SGLang's EP path is stable for that checkpoint. Capacity math starts from resident weights plus KV, not from the 'active parameter' marketing number."

---

## **Q28: "Where does speculative decoding fail on tool-calling agents?"**

**Strong answer:**

> "Agents have high-entropy turns — which tool, which argument — and then low-entropy JSON. Alpha will be poor on the planning tokens and better on `{\"name\":`. If the draft was trained on prose, tool traces will look bad. I would measure alpha **by stage** — plan vs tool JSON vs final answer — or only speculate on the structured tail. Constrained decoding / grammar (xgrammar, outlines) can beat speculation for JSON validity; they solve different problems. SGLang is in the conversation here because of the prefix tree plus structured decode."

---

## **Q29: "Give me a 60-second capacity plan for a 70B."**

**Strong answer:**

> "70B FP16 is ~140 GB — that is multiple 80 GB GPUs with TP, or a quantized 70B on fewer. I pick AWQ/FP8 after eval. Bytes per token scale with layers and KV heads — 70B KV is much fatter than 8B; concurrency per replica collapses. I would rather two TP=2 replicas than one TP=4 if interconnect is the tax and QPS is the goal, but the model has to fit first. Prefix cache is mandatory — 70B prefill is expensive. Speculative with a 7B/8B draft is the classic setup **if** alpha is high. I load-test before I promise QPS. I keep a provider fallback because a 70B cold start is a bad day."

---

## **Q30: "What would you monitor that LangSmith will not show you?"**

**Strong answer:**

> "LangSmith sees spans: retrieval, the HTTP call, tokens in/out, TTFT if I log it. It does not see KV cache percent, block allocation, prefix hit rate, speculative acceptance, or how many requests were waiting inside the engine. Those are Prometheus on the vLLM replica. When TTFT is bad I need both: the app trace tells me the prompt grew; the engine tells me the GPU was already at 92% KV. I would export both to the same dashboard rather than argue across tools."

---

# **16. Key Takeaways**

---

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. PREFILL = compute, DECODE = memory. TTFT ≠ e2e.                     │
│  2. KV bytes/token = 2 · L · n_kv · d_head · bytes. GQA matters.        │
│  3. Concurrency is an HBM problem. Size from leftover KV, then measure. │
│  4. PagedAttention = virtual memory for KV; prefix cache needs a        │
│     stable token prefix at the front of the prompt.                     │
│  5. Continuous / iteration-level batching, not static batching.         │
│  6. Optimize GOODPUT (SLO-compliant tok/s), not peak GPU util.          │
│  7. vLLM is the default self-host; Ollama is what you shipped locally.  │
│  8. Speculative decode is lossless iff you verify + resample on target; │
│     the metric is acceptance rate α — turn it off when α is low.        │
│  9. DP for QPS; TP/PP when the model does not fit. Don't TP a 7B.       │
│ 10. K8s: one GPU per vLLM pod; scale on waiting + KV%; warm min for     │
│     interactive; cold start is minutes, not milliseconds.               │
│ 11. Quantize weights to fit; quantize KV to raise concurrency; eval     │
│     both like a model upgrade (doc 49 + 13).                            │
│ 12. Never claim a vLLM production fleet you did not run. Defend the     │
│     mechanics; map next steps onto Bilbo / AI Cargo honestly.           │
└─────────────────────────────────────────────────────────────────────────┘
```

### One-liners to keep in your pocket

| Topic | Line |
|---|---|
| Mental model | *"Prefill is compute-bound, decode is memory-bound, KV is the working set."* |
| PagedAttention | *"Virtual memory for KV blocks — stop reserving max-len on every request."* |
| Continuous batching | *"Repack every decode step so a short reply does not wait on a long one."* |
| Prefix cache | *"Identical leading tokens only — put the system prompt first and never shuffle it."* |
| Speculative | *"Draft proposes, target verifies; lossless with target resample; watch alpha."* |
| Goodput | *"Tokens that met the SLO. The rest is a vanity metric."* |
| Resume honesty | *"Ollama in the projects; vLLM in the toolkit; I will not invent a fleet."* |

### How to study this the night before

1. Compute 128 KB/token on paper once. Change $n_{\text{kv}}$ and recompute.
2. Draw prefill vs decode and mark TTFT / TPOT.
3. Say the PagedAttention OS analogy out loud.
4. Say speculative decode + lossless condition + when you disable it.
5. Say the Ollama → vLLM migration line for Bilbo.
6. Skim `13` §7.5 and `49` §6.5 so the three docs agree in your mouth.

---

**Cross-references:** `13_quantization.md` (AWQ/GPTQ/FP8, vLLM load flags) · `49_llmops_and_observability.md` (SLOs, traces, eval-gated flag changes, agent cost) · `23_cloud_mlops_deployment.md` (K8s, KEDA, GPU nodes) · `04_transformers_and_attention.md` (why generation is sequential) · `16_context_engineering.md` / `52_frontier_ai_and_2026_landscape.md` (stable prefixes) · `08_knowledge_distillation.md` (training-time cheaper model vs inference-time speculation) · P07 Bilbo (Ollama 7B, sub-5s) · P15 AI Cargo (provider failover)

*Document prepared for Rahul Sharma — AI Engineer interview preparation (vLLM, speculative decoding, model serving).*
