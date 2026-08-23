# Real-Time Anomaly Detection on LivePerson Chats — Project Interview Preparation

**Candidate:** Rahul Sharma | **Company:** Jet2 and Jet2 Holidays, Leeds, UK | **Role:** Data Scientist | **Duration:** June 2022 – August 2024  
**Resume Line:** *"Developed an anomaly-detection framework for LivePerson chats using topic modelling, moving averages, z-scores, and Isolation Forest, achieving 96% specificity while reducing false-positive operational alerts."*  
**Document Scope:** End-to-end project deep dive — architecture, algorithms, statistical methods, code patterns, interview questions, and red-flag handling

---

## Table of Contents

0. [Resume Bullet ↔ Proof](#0-resume-bullet--proof)
1. [Project Overview (STAR)](#1-project-overview-star)
2. [Deep Technical Walkthrough](#2-deep-technical-walkthrough)
   - 2.7 [The Four-Signal Ensemble Framework](#27-the-four-signal-ensemble-framework)
3. [Key Metrics & Results](#3-key-metrics--results)
4. [Topics You Must Know](#4-topics-you-must-know)
5. [Interview Questions (35+) with Model Answers](#5-interview-questions-35-with-model-answers)
   - [Trick & Follow-Up Questions](#trick--follow-up-questions)
6. [Red Flags & How to Handle](#6-red-flags--how-to-handle)
7. [Key Takeaways](#7-key-takeaways)

**Companion learning notes:** [`../learning/19_classical_ml_algorithms.md`](../learning/19_classical_ml_algorithms.md) · [`../learning/18_nlp_and_embeddings.md`](../learning/18_nlp_and_embeddings.md) · [`../learning/14_evaluation_metrics.md`](../learning/14_evaluation_metrics.md) · [`../learning/34_time_series_and_forecasting.md`](../learning/34_time_series_and_forecasting.md)

---

# 0. Resume Bullet ↔ Proof

The resume bullet is one sentence with four named techniques and one number. Every piece of it maps to a section below.

| Resume Fragment | Exact Claim | Where the Proof Lives | 1-Sentence Spoken Answer |
|---|---|---|---|
| "anomaly-detection framework for **LivePerson chats**" | Live customer chat transcripts, parsed from nested LivePerson JSON, aggregated into 30-minute windows | §1.2 Architecture, §2.1 Data Pipeline, §2.7.1 LivePerson Context | "LivePerson is the live-chat platform our contact centre ran on, and I built the detection layer on top of its conversation exports." |
| "using **topic modelling**" | NMF over TF-IDF (LDA as cross-check), $k \approx 15$ tuned by c_v coherence; topic weights per window become features | §2.2 Topic Modeling, §2.7.2 | "NMF on TF-IDF surfaced what customers were talking about, and the per-window topic weights became features — so the detector saw theme, not just volume." |
| "…**moving averages**…" | SMA (stable baseline) + EMA (fast responder), 12h / 24h / 7d windows | §2.3.2, §2.7.3 | "SMA defined what normal looks like for this time of day; EMA reacted fast enough to catch the ramp-up." |
| "…**z-scores**…" | Rolling z-score on a 24h window; $\|z\| > 3$ as the strong-signal threshold | §2.3.3, §2.7.4 | "Rolling z-scores normalised every keyword and topic against its own recent history, so a spike is measured in standard deviations, not raw counts." |
| "…and **Isolation Forest**" | 200 trees, `max_samples=0.75`, `contamination=0.01`, `max_features=0.75`, grid-searched | §2.4, §2.7.5 | "Isolation Forest was the combiner — it caught the multivariate interactions that no single z-score could see." |
| "achieving **96% specificity**" | 96% of non-event windows correctly identified as normal → 4% false positive rate | §2.6, §3.1, §2.7.7 | "Ninety-six percent specificity means a four percent false-alarm rate, which is roughly one alert a day for the ops team to triage." |
| "while **reducing false-positive operational alerts**" | FPR halved from the 8% of the earlier tuning to 4%; alert fatigue was the explicit design constraint | §2.7.7 Specificity Trade-Off, §3.1 | "Specificity *is* the false-alarm metric — pushing it from the low nineties to ninety-six halved the noise the ops team had to wade through." |

> **If asked for the single number:** "96% specificity — meaning a 4% false-positive rate on non-event windows. I optimised for specificity rather than recall on purpose, because in an alerting system alert fatigue kills adoption faster than a missed alert does."

---

# 1. Project Overview (STAR)

---

## 1.1 STAR Summary

| STAR Element | Details |
|:-------------|:--------|
| **Situation** | Jet2's operations team was overwhelmed by the volume of real-time customer chat data flowing through LivePerson. When disasters struck (volcanic eruptions, airline strikes, severe weather, political unrest), the team only discovered the event *after* customers started flooding support lines — a purely **reactive** posture. |
| **Task** | Build a system to **detect emerging disaster events in real time** from chat data, shifting Jet2 from reactive to proactive crisis response. |
| **Approach** | (1) Engineered a data pipeline to parse nested LivePerson JSON into structured format; (2) Applied unsupervised **topic modelling** to surface disaster-related keyword clusters; (3) Built a statistical signal-processing layer (**moving averages**, **z-scores**, % changes) to detect sudden spikes; (4) Applied **Isolation Forest** for multivariate anomaly detection beyond simple frequency spikes; (5) Combined all four signals into a single ensemble decision rather than alerting on any one of them; (6) Validated against one year of historical incidents; (7) Tuned with grid search and iterated with operational feedback. |
| **Result** | **96% specificity** in real-time detection — a 4% false-positive rate — which **materially reduced false-positive operational alerts** and transformed the operations team from reactive firefighting to proactive alerting with prescriptive playbooks. |

---

## 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        REAL-TIME DISASTER EVENT DETECTION                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────────┐ │
│  │  LivePerson   │    │  JSON Parser &   │    │   Structured Chat Database    │ │
│  │  Chat Logs    │───▶│  Flattener       │───▶│   (cleaned, timestamped)     │ │
│  │  (Raw JSON)   │    │  (nested → flat) │    │                              │ │
│  └──────────────┘    └──────────────────┘    └──────────────┬────────────────┘ │
│                                                              │                  │
│                              ┌───────────────────────────────┤                  │
│                              ▼                               ▼                  │
│                ┌──────────────────────┐     ┌──────────────────────────────┐    │
│                │   Topic Modeling     │     │   Statistical Signal Layer   │    │
│                │   (Unsupervised)     │     │   ┌────────────────────┐    │    │
│                │                      │     │   │ Moving Averages    │    │    │
│                │  • LDA / NMF         │     │   │ (SMA + EMA)        │    │    │
│                │  • Keyword clusters  │     │   ├────────────────────┤    │    │
│                │  • Topic coherence   │     │   │ Z-Scores           │    │    │
│                │  • Theme tracking    │     │   │ (rolling window)   │    │    │
│                │                      │     │   ├────────────────────┤    │    │
│                └──────────┬───────────┘     │   │ % Change           │    │    │
│                           │                 │   │ (vs baseline)      │    │    │
│                           │                 │   ├────────────────────┤    │    │
│                           │                 │   │ IQR Outliers       │    │    │
│                           │                 │   └────────────────────┘    │    │
│                           │                 └──────────────┬─────────────┘    │
│                           │                                │                   │
│                           ▼                                ▼                   │
│                ┌─────────────────────────────────────────────────┐             │
│                │          Feature Engineering Layer              │             │
│                │  • Topic distribution per time window           │             │
│                │  • Statistical features (z, %, EMA, IQR)       │             │
│                │  • Chat volume features                         │             │
│                │  • Keyword frequency features                   │             │
│                └────────────────────────┬────────────────────────┘             │
│                                         │                                      │
│                                         ▼                                      │
│                          ┌──────────────────────────┐                          │
│                          │    Isolation Forest       │                          │
│                          │    (Anomaly Detection)    │                          │
│                          │                           │                          │
│                          │  • Anomaly score per      │                          │
│                          │    time window             │                          │
│                          │  • Grid-search tuned      │                          │
│                          │  • Validated vs actual    │                          │
│                          │    historical events      │                          │
│                          └────────────┬─────────────┘                          │
│                                       │                                        │
│                                       ▼                                        │
│                      ┌──────────────────────────────┐                          │
│                      │   Alert & Playbook Engine     │                          │
│                      │   • Severity classification   │                          │
│                      │   • Prescriptive playbooks    │                          │
│                      │   • Dashboard / notifications │                          │
│                      └──────────────────────────────┘                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Tech Stack

| Layer | Technology |
|:------|:-----------|
| **Chat Platform** | LivePerson (real-time customer chat) |
| **Data Parsing** | Python (`json`, `pandas`, custom nested-JSON flattener) |
| **Topic Modeling** | `scikit-learn` (NMF, LDA via `LatentDirichletAllocation`), `gensim` |
| **Text Processing** | `spaCy`, `nltk`, TF-IDF vectorization (`TfidfVectorizer`) |
| **Statistical Analysis** | `numpy`, `scipy.stats`, `pandas` rolling windows |
| **Anomaly Detection** | `scikit-learn` (`IsolationForest`) |
| **Hyperparameter Tuning** | `scikit-learn` (`GridSearchCV`, custom cross-validation) |
| **Visualization** | `matplotlib`, `seaborn`, internal dashboards |
| **Collaboration** | Worked with data engineering team for pipeline integration |

---

## 1.4 Real-Time Aspect

The system operated on a **near-real-time batch micro-cycle**:

| Aspect | Detail |
|:-------|:-------|
| **Ingestion frequency** | Chat data pulled in short time windows (e.g., every 15–30 minutes) |
| **Processing** | Each batch: parse → extract features → run topic model inference → compute stats → score with Isolation Forest |
| **Latency** | End-to-end from chat to alert: minutes (not sub-second streaming, but fast enough for operational response) |
| **Why not pure streaming** | Volume didn't warrant Kafka/Flink-level infrastructure; batch micro-cycles provided sufficient timeliness for disaster response |

> **Interview framing:** "Real-time in the context of disaster response means we needed alerts within minutes of a spike beginning, not after an overnight batch. We achieved this with frequent micro-batch processing cycles."

---

# 2. Deep Technical Walkthrough

---

## 2.1 Data Pipeline: LivePerson JSON → Structured Format

### 2.1.1 Raw Data Structure

LivePerson exports chat data as **deeply nested JSON**. A simplified example:

```json
{
  "conversationId": "conv_abc123",
  "info": {
    "startTime": "2023-06-15T14:32:00Z",
    "endTime": "2023-06-15T14:45:00Z",
    "channel": "web",
    "campaignInfo": {
      "campaignId": "camp_456",
      "engagementId": "eng_789"
    }
  },
  "messageRecords": [
    {
      "messageId": "msg_001",
      "sentBy": "Consumer",
      "timeStamp": "2023-06-15T14:32:05Z",
      "messageData": {
        "msg": {
          "text": "My flight to Rhodes has been cancelled, is this because of the volcano?"
        }
      },
      "participantId": "part_111"
    },
    {
      "messageId": "msg_002",
      "sentBy": "Agent",
      "timeStamp": "2023-06-15T14:33:10Z",
      "messageData": {
        "msg": {
          "text": "I'm sorry to hear that. Let me check the details for you."
        }
      },
      "participantId": "part_222"
    }
  ],
  "consumerParticipants": [
    {
      "participantId": "part_111",
      "firstName": "Jane",
      "consumerName": "Jane D."
    }
  ],
  "agentParticipants": [
    {
      "participantId": "part_222",
      "agentFullName": "Support Agent 1",
      "agentLoginName": "agent1"
    }
  ],
  "summary": "Customer inquiry about flight cancellation due to volcanic activity.",
  "csat": { "score": 4, "status": "filled" }
}
```

### 2.1.2 Parsing & Flattening Strategy

```python
import json
import pandas as pd
from typing import Dict, List, Any, Optional

def safe_get(d: dict, *keys, default=None):
    """Safely navigate nested dict keys."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

def flatten_conversation(conv: Dict[str, Any]) -> List[Dict]:
    """Flatten one nested LivePerson conversation into message-level rows."""
    rows = []
    conv_id = conv.get("conversationId", "unknown")
    start_time = safe_get(conv, "info", "startTime")
    channel = safe_get(conv, "info", "channel")
    campaign_id = safe_get(conv, "info", "campaignInfo", "campaignId")
    summary = conv.get("summary", "")
    csat_score = safe_get(conv, "csat", "score")
    
    # Build participant lookup
    participants = {}
    for p in conv.get("consumerParticipants", []):
        participants[p.get("participantId")] = {
            "role": "consumer",
            "name": p.get("consumerName", "Unknown")
        }
    for a in conv.get("agentParticipants", []):
        participants[a.get("participantId")] = {
            "role": "agent",
            "name": a.get("agentFullName", "Unknown")
        }
    
    for msg in conv.get("messageRecords", []):
        text = safe_get(msg, "messageData", "msg", "text", default="")
        participant = participants.get(msg.get("participantId"), {})
        
        rows.append({
            "conversation_id": conv_id,
            "message_id": msg.get("messageId"),
            "timestamp": msg.get("timeStamp"),
            "start_time": start_time,
            "channel": channel,
            "campaign_id": campaign_id,
            "sent_by": msg.get("sentBy"),
            "role": participant.get("role", "unknown"),
            "participant_name": participant.get("name", "unknown"),
            "text": text,
            "summary": summary,
            "csat_score": csat_score
        })
    
    return rows

def parse_liveperson_export(file_path: str) -> pd.DataFrame:
    """Parse full LivePerson JSON export into a DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversations = data if isinstance(data, list) else data.get("records", [data])
    
    all_rows = []
    parse_errors = []
    
    for i, conv in enumerate(conversations):
        try:
            rows = flatten_conversation(conv)
            all_rows.extend(rows)
        except Exception as e:
            parse_errors.append({"index": i, "error": str(e)})
    
    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    
    if parse_errors:
        print(f"Warning: {len(parse_errors)} conversations failed to parse")
    
    return df
```

### 2.1.3 JSON Parsing Challenges

| Challenge | Description | Solution |
|:----------|:------------|:---------|
| **Deeply nested structures** | Message text buried 3–4 levels deep (`messageData.msg.text`) | Recursive `safe_get()` utility with default fallbacks |
| **Missing fields** | Some conversations lacked `summary`, `csat`, or `campaignInfo` | `.get()` with defaults; never assume field existence |
| **Encoding issues** | Non-ASCII characters (accented names, emojis) in chat text | Force `utf-8` encoding; `errors='replace'` for corrupt bytes |
| **Inconsistent schemas** | LivePerson API versions had slightly different JSON structures | Schema detection at parse time; version-aware flatteners |
| **Large file sizes** | Exports could be GBs of JSON | Streaming JSON parser (`ijson`) for large files; chunked processing |
| **Timestamp formats** | Mix of ISO 8601, epoch milliseconds, and occasionally local times | Normalize all timestamps to UTC `datetime64` immediately after parsing |
| **Duplicate messages** | Same message appearing in overlapping export windows | Dedup on `(conversation_id, message_id)` composite key |

---

## 2.2 Topic Modeling

### 2.2.1 Algorithm Choice: NMF (Primary) + LDA (Validation)

| Factor | NMF (Non-negative Matrix Factorization) | LDA (Latent Dirichlet Allocation) |
|:-------|:-----------------------------------------|:-----------------------------------|
| **Interpretability** | Produces crisp, non-overlapping topics | Softer, probabilistic topic assignments |
| **Speed** | Fast — deterministic optimization | Slower — iterative Gibbs sampling or variational inference |
| **Short text** | Handles short chat messages well | Can struggle with short documents |
| **Primary use** | **Primary model** — cleaner disaster-related topic clusters | **Validation** — cross-checked NMF topics |

> **Why NMF over BERTopic:** BERTopic (BERT embeddings + HDBSCAN + c-TF-IDF) is powerful but (a) computationally heavier for the volume of chat data, (b) harder to interpret for non-technical operational leaders, and (c) NMF with TF-IDF was sufficient and more transparent for this use case. BERTopic would be a good upgrade if richer semantic understanding were needed.

### 2.2.2 Text Preprocessing Pipeline

```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

CUSTOM_STOP_WORDS = {
    "jet2", "booking", "flight", "hotel", "holiday", "please",
    "thank", "thanks", "hi", "hello", "agent", "help"
}

def preprocess_text(text: str) -> str:
    """Clean and lemmatize chat message."""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
        and len(token.text) > 2
        and token.lemma_ not in CUSTOM_STOP_WORDS
    ]
    return " ".join(tokens)

# TF-IDF vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=5,             # term must appear in at least 5 docs
    max_df=0.85,          # ignore terms in >85% of docs
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True     # apply log normalization to TF
)
```

### 2.2.3 NMF Topic Modeling

```python
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score

# Fit TF-IDF
tfidf_matrix = tfidf.fit_transform(df["processed_text"])
feature_names = tfidf.get_feature_names_out()

# NMF with tuned number of topics
nmf = NMF(
    n_components=15,         # number of topics (tuned via coherence)
    init="nndsvda",          # non-negative double SVD (better initialization)
    random_state=42,
    max_iter=500,
    alpha_W=0.1,             # L1 regularization on W
    alpha_H=0.1,             # L1 regularization on H
    l1_ratio=0.5             # balance between L1 and L2
)
W = nmf.fit_transform(tfidf_matrix)   # document-topic matrix
H = nmf.components_                    # topic-term matrix

def display_topics(model, feature_names, n_top_words=10):
    """Print top words per topic."""
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")

display_topics(nmf, feature_names)
```

**Example discovered topics:**

| Topic ID | Top Keywords | Interpretation |
|:---------|:-------------|:---------------|
| 3 | volcano, eruption, ash, airspace, close, cancel | Volcanic event |
| 7 | strike, airport, staff, cancel, delay, union | Industrial action / strike |
| 11 | storm, weather, delay, divert, wind, rain | Severe weather |
| 14 | covid, test, pcr, quarantine, restriction, travel | Pandemic restrictions |

### 2.2.4 Topic Coherence (Hyperparameter Tuning for $k$)

**Coherence score** measures how semantically similar the top words in each topic are. Higher coherence = more interpretable topics.

```python
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import numpy as np

# Prepare gensim-compatible data
texts = [doc.split() for doc in df["processed_text"]]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

coherence_scores = {}
for k in range(5, 30):
    nmf_k = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=300)
    nmf_k.fit(tfidf_matrix)
    
    # Extract top words per topic
    topics = []
    for topic in nmf_k.components_:
        top_indices = topic.argsort()[:-11:-1]
        topics.append([feature_names[i] for i in top_indices])
    
    cm = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v"     # c_v coherence (most robust)
    )
    coherence_scores[k] = cm.get_coherence()

best_k = max(coherence_scores, key=coherence_scores.get)
# Result: best_k ≈ 15 (plateau region with high coherence)
```

> **Interview note:** "I tuned the number of topics by plotting coherence scores (c_v metric) across a range of $k$ values. I picked $k$ in the plateau region where coherence was high and topics remained interpretable to the operations team — around 15 topics."

---

## 2.3 Statistical Signal Processing

The statistical layer operated on **time-windowed aggregations** of topic distributions and keyword frequencies.

### 2.3.1 Time Windowing

```python
# Aggregate chat data into time windows
def create_time_windows(df, freq="30min"):
    """Aggregate chats into fixed time windows."""
    df = df.set_index("timestamp").sort_index()
    
    windows = df.resample(freq).agg({
        "conversation_id": "nunique",       # chat volume
        "text": lambda x: " ".join(x),      # concatenated text
    }).rename(columns={"conversation_id": "chat_volume"})
    
    return windows
```

### 2.3.2 Moving Averages: SMA and EMA

**Simple Moving Average (SMA):**

$$
\text{SMA}_t = \frac{1}{n} \sum_{i=0}^{n-1} x_{t-i}
$$

**Exponential Moving Average (EMA):**

$$
\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
$$

where $\alpha = \frac{2}{n+1}$ (smoothing factor for window size $n$).

| Property | SMA | EMA |
|:---------|:----|:----|
| **Weighting** | Equal weight to all points in window | More weight to recent points |
| **Responsiveness** | Slower to react to sudden changes | Faster to react |
| **Use in this project** | Baseline for "normal" behavior | Detect rapid shifts early |

```python
import numpy as np

def compute_moving_averages(series, window=24):
    """Compute SMA and EMA for a time series."""
    sma = series.rolling(window=window, min_periods=1).mean()
    ema = series.ewm(span=window, adjust=False).mean()
    return sma, ema

# Example: track "volcano" keyword frequency per window
df_windows["volcano_freq"] = df_windows["text"].apply(
    lambda x: x.lower().count("volcano")
)
df_windows["volcano_sma"], df_windows["volcano_ema"] = compute_moving_averages(
    df_windows["volcano_freq"], window=48  # 48 windows = 24 hours if 30-min windows
)
```

### 2.3.3 Z-Scores for Anomaly Detection

**Formula:**

$$
z_t = \frac{x_t - \mu_{\text{window}}}{\sigma_{\text{window}}}
$$

where $\mu_{\text{window}}$ and $\sigma_{\text{window}}$ are the mean and standard deviation of a rolling window.

**Interpretation:**
- $|z| > 2$: Unusual (≈ 5% chance under normal distribution)
- $|z| > 3$: Highly unusual (≈ 0.3% chance)
- For disaster detection, we flag $z > 3$ as a strong signal

```python
def compute_rolling_zscore(series, window=48):
    """Compute rolling z-score for a time series."""
    rolling_mean = series.rolling(window=window, min_periods=10).mean()
    rolling_std = series.rolling(window=window, min_periods=10).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    z_score = (series - rolling_mean) / rolling_std
    return z_score.fillna(0)

# Z-scores for each keyword/topic
for col in keyword_columns:
    df_windows[f"{col}_zscore"] = compute_rolling_zscore(
        df_windows[col], window=48   # 24-hour rolling window
    )
```

**Window size choices:**

| Window | Duration (30-min windows) | Purpose |
|:-------|:--------------------------|:--------|
| 24 | 12 hours | Short-term spike detection |
| 48 | 24 hours | **Primary** — captures daily cycle |
| 336 | 1 week | Long-term trend baseline |

### 2.3.4 Percentage Change from Baseline

$$
\text{\% Change}_t = \frac{x_t - \text{baseline}_t}{\text{baseline}_t} \times 100
$$

where baseline is typically the SMA or a historical average for the same time-of-day/day-of-week.

```python
def compute_pct_change(series, baseline):
    """Compute percentage change relative to baseline."""
    # Guard against zero baseline
    safe_baseline = baseline.replace(0, np.nan)
    pct = ((series - safe_baseline) / safe_baseline) * 100
    return pct.fillna(0)

df_windows["volume_pct_change"] = compute_pct_change(
    df_windows["chat_volume"],
    df_windows["chat_volume_sma"]
)
# Flag: volume_pct_change > 200% → potential event
```

### 2.3.5 IQR-Based Outlier Detection

$$
\text{IQR} = Q_3 - Q_1
$$
$$
\text{Lower fence} = Q_1 - 1.5 \times \text{IQR}
$$
$$
\text{Upper fence} = Q_3 + 1.5 \times \text{IQR}
$$

```python
def iqr_outlier_flags(series, multiplier=1.5):
    """Flag values outside IQR fences."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return ((series < lower) | (series > upper)).astype(int)

df_windows["volume_iqr_outlier"] = iqr_outlier_flags(df_windows["chat_volume"])
```

---

## 2.4 Isolation Forest: Core Anomaly Detection

### 2.4.1 How Isolation Forest Works

**Core intuition:** Anomalies are **few and different**. If you randomly partition data, anomalies get isolated in fewer splits than normal points.

**Algorithm:**

1. **Build isolation trees (iTrees):**
   - Randomly select a feature
   - Randomly select a split value between the feature's min and max
   - Recursively partition until each point is isolated or max depth reached

2. **Compute path length:**
   - For each data point, record how many splits were needed to isolate it
   - Short path → easy to isolate → likely anomaly
   - Long path → hard to isolate → likely normal

3. **Anomaly score:**

$$
s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}
$$

where:
- $E[h(x)]$ = average path length of point $x$ across all trees
- $c(n)$ = average path length of unsuccessful search in a Binary Search Tree (normalization factor)
- $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$, where $H(i) \approx \ln(i) + 0.5772$ (Euler's constant)

**Score interpretation:**
- $s \approx 1$: Strong anomaly (very short path)
- $s \approx 0.5$: Normal point (average path)
- $s \approx 0$: Dense cluster point (very long path)

```
Normal point (long path):          Anomaly (short path):
    ┌──────────┐                      ┌──────────┐
    │  split 1 │                      │  split 1 │
    ├────┬─────┤                      ├────┬─────┤
    │    │     │                      │    │ ★   │ ← isolated in 1 split!
    │  split 2 │                      │    │     │
    │  ┌─┬──┐  │                      └────┴─────┘
    │  │ │  │  │
    │  split 3 │
    │  ┌┬─┐   │
    │  │●│    │  ← isolated after 4 splits
    └──┴──────┘
```

### 2.4.2 Why Isolation Forest Over Other Methods

| Method | Pros | Cons | Why Not Chosen |
|:-------|:-----|:-----|:---------------|
| **Isolation Forest** | ✅ Scales to large datasets, ✅ No distance metric needed, ✅ Handles mixed features, ✅ Few hyperparameters | Struggles with high-dim data | **Chosen** — best fit for tabular feature set |
| **Local Outlier Factor (LOF)** | Good for local density anomalies | Slow on large data ($O(n^2)$), sensitive to $k$ | Too slow for near-real-time micro-batches |
| **One-Class SVM** | Robust with kernel trick | Very slow to train, memory-heavy, needs feature scaling | Doesn't scale to our data volume |
| **DBSCAN** | Finds clusters and outliers together | Sensitive to `eps` and `min_samples`, not designed for scoring | Clustering tool, not a scoring anomaly detector |
| **Statistical only (z-score)** | Simple, interpretable | Univariate — misses multi-feature interactions | Used as features, but alone insufficient |
| **Autoencoders** | Captures complex nonlinear patterns | Needs lots of data, harder to interpret/explain | Overkill; IF was sufficient and more transparent |

> **Interview answer:** "I chose Isolation Forest because it handles our mixed feature set (numeric statistics + topic proportions) without requiring distance computation, scales linearly with data size, and provides a clear anomaly score. Statistical methods like z-scores were used as *features into* the Isolation Forest, not as standalone detectors — the IF captures multivariate interactions that individual z-scores miss."

### 2.4.3 Features Used as Input

```python
feature_columns = [
    # Volume features
    "chat_volume",
    "chat_volume_zscore",
    "chat_volume_pct_change",
    "chat_volume_iqr_outlier",
    
    # Topic distribution features (from NMF)
    "topic_3_weight",          # volcanic topic
    "topic_7_weight",          # strike topic
    "topic_11_weight",         # weather topic
    "topic_14_weight",         # pandemic topic
    "max_topic_weight",        # max weight across disaster topics
    "disaster_topic_entropy",  # entropy across disaster topics
    
    # Keyword z-scores
    "cancel_zscore",
    "delay_zscore",
    "emergency_zscore",
    "volcano_zscore",
    "strike_zscore",
    "storm_zscore",
    
    # Keyword EMAs
    "cancel_ema",
    "delay_ema",
    "emergency_ema",
    
    # Time features
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    
    # Rate-of-change features
    "volume_acceleration",     # 2nd derivative of volume
    "topic_shift_rate",        # how fast topic distribution is changing
]
```

### 2.4.4 Isolation Forest Implementation & Grid Search

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import numpy as np

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df_windows[feature_columns].fillna(0))

# Grid search (custom, since IF is unsupervised — no labels for CV)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_samples": [0.5, 0.75, 1.0],          # fraction of samples per tree
    "contamination": [0.005, 0.01, 0.02, 0.05], # expected anomaly proportion
    "max_features": [0.5, 0.75, 1.0],           # fraction of features per tree
    "random_state": [42]
}

best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    iso = IsolationForest(**params)
    iso.fit(X)
    predictions = iso.predict(X)   # -1 = anomaly, 1 = normal
    
    # Evaluate against historical labeled events
    flagged_windows = df_windows.index[predictions == -1]
    
    # Compute specificity against labeled events
    true_normals = labeled_events[labeled_events["is_event"] == 0]
    correctly_identified_normals = true_normals[
        ~true_normals.index.isin(flagged_windows)
    ]
    specificity = len(correctly_identified_normals) / len(true_normals)
    
    # Also track recall (sensitivity)
    true_events = labeled_events[labeled_events["is_event"] == 1]
    detected_events = true_events[true_events.index.isin(flagged_windows)]
    recall = len(detected_events) / len(true_events) if len(true_events) > 0 else 0
    
    # Optimize for high specificity while maintaining acceptable recall
    combined_score = 0.6 * specificity + 0.4 * recall
    
    if combined_score > best_score:
        best_score = combined_score
        best_params = params

# Best params found:
# n_estimators=200, max_samples=0.75, contamination=0.01,
# max_features=0.75, random_state=42
```

**Hyperparameter explanations:**

| Parameter | Best Value | Meaning |
|:----------|:-----------|:--------|
| `n_estimators` | 200 | Number of isolation trees in the forest |
| `max_samples` | 0.75 | 75% of data used to build each tree (more diversity than 1.0) |
| `contamination` | 0.01 | Expected 1% of time windows are anomalous — disasters are rare |
| `max_features` | 0.75 | 75% of features per tree (adds diversity, prevents overfitting) |

### 2.4.5 Final Model & Scoring

```python
# Final model
final_iso = IsolationForest(
    n_estimators=200,
    max_samples=0.75,
    contamination=0.01,
    max_features=0.75,
    random_state=42,
    n_jobs=-1
)
final_iso.fit(X)

# Anomaly scores (lower = more anomalous)
anomaly_scores = final_iso.decision_function(X)
predictions = final_iso.predict(X)   # -1 = anomaly, 1 = normal

# Convert to 0-1 anomaly probability for operational use
from sklearn.preprocessing import MinMaxScaler
score_scaler = MinMaxScaler()
anomaly_proba = 1 - score_scaler.fit_transform(
    anomaly_scores.reshape(-1, 1)
).flatten()  # invert so higher = more anomalous

df_windows["anomaly_score"] = anomaly_proba
df_windows["is_anomaly"] = (predictions == -1).astype(int)
```

---

## 2.5 Validation Against Historical Events

```python
# Load labeled historical events (past year)
historical_events = pd.DataFrame({
    "event_date": ["2022-09-15", "2022-12-20", "2023-03-10", ...],
    "event_type": ["volcanic_eruption", "winter_storm", "airport_strike", ...],
    "duration_hours": [48, 24, 72, ...],
    "description": ["Etna eruption affecting Sicily flights", ...]
})

# Match detected anomalies to actual events (within ±6 hour window)
def validate_detections(anomaly_windows, events, tolerance_hours=6):
    true_positives = 0
    false_negatives = 0
    
    for _, event in events.iterrows():
        event_start = pd.Timestamp(event["event_date"])
        event_end = event_start + pd.Timedelta(hours=event["duration_hours"])
        
        # Check if any anomaly falls within event window ± tolerance
        matched = anomaly_windows[
            (anomaly_windows.index >= event_start - pd.Timedelta(hours=tolerance_hours)) &
            (anomaly_windows.index <= event_end + pd.Timedelta(hours=tolerance_hours))
        ]
        
        if len(matched) > 0:
            true_positives += 1
        else:
            false_negatives += 1
    
    return true_positives, false_negatives

tp, fn = validate_detections(
    df_windows[df_windows["is_anomaly"] == 1],
    historical_events
)
```

---

## 2.6 Specificity as the Primary Metric

### 2.6.1 Why Specificity?

**Confusion Matrix for Event Detection:**

```
                          Predicted
                    Event        No Event
Actual  Event       TP            FN
        No Event    FP            TN
```

**Formulas:**

$$
\text{Specificity} = \frac{TN}{TN + FP} = \frac{\text{True Normals Correctly Identified}}{\text{All Actual Normals}}
$$

$$
\text{Sensitivity (Recall)} = \frac{TP}{TP + FN} = \frac{\text{Events Detected}}{\text{All Actual Events}}
$$

$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{True Events Among Alerts}}{\text{All Alerts}}
$$

### 2.6.2 Why Specificity Was Prioritized

| Reason | Explanation |
|:-------|:------------|
| **Severe class imbalance** | Disaster events are very rare (maybe 1% of time windows). With 99% negatives, even small FP rates create alert fatigue. |
| **Alert fatigue is deadly** | If the system raises too many false alarms, the ops team will ignore it. High specificity = low false alarm rate. |
| **Operational trust** | The ops team needed to trust that when the system said "no event," it was right. Specificity measures exactly that. |
| **Complementary to recall** | We also tracked recall — we needed to *catch* events. But we optimized the trade-off toward specificity because a missed alert (caught by other means) was less damaging than constant false alarms destroying trust. |

**96% specificity means:** Of all normal (non-event) time windows, 96% were correctly identified as normal. The false positive rate was **4%** — roughly one alert a day for an operations team to triage.

**Why this is the "reducing false-positive operational alerts" claim.** Specificity and false-positive rate are the same quantity viewed from opposite ends: $\text{FPR} = 1 - \text{Specificity}$. Moving specificity from the low nineties to 96% is not a cosmetic three-point gain — it **halves** the alert noise. In operational terms:

| Specificity | FPR | False alerts/day (≈48 windows/day) | Ops team reaction |
|---|---|---|---|
| 90% | 10% | ~5 | "This thing cries wolf." Alerts get muted. |
| 92% | 8% | ~4 | Tolerable, but trust is fragile. |
| **96%** | **4%** | **~2** | **Each alert is worth opening.** Trust holds. |
| 99% | 1% | ~0.5 | Lovely — but recall has almost certainly collapsed. |

> **Trade-off framing:** "With 96% specificity we halved the false-alarm load compared to where the earlier tuning sat, and that's the number the ops team actually felt — they went from several false alerts a day to roughly one. We could have pushed specificity higher still, but every additional point came out of recall, and at some point you have a system that never cries wolf because it never says anything at all."

---

## 2.7 The Four-Signal Ensemble Framework

The resume bullet names four techniques in one breath — **topic modelling, moving averages, z-scores, and Isolation Forest**. An interviewer will want to know why four, what each contributes that the others don't, and how they were combined. This section is the answer.

**The one-line version:** three of them are *signal generators* and one is the *combiner*. Topic modelling, moving averages and z-scores each turn raw chat into a normalised numeric feature; Isolation Forest consumes all of them at once and produces the single score that drives the alert.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                  FOUR-SIGNAL ENSEMBLE — HOW THEY COMPOSE                  │
│                                                                           │
│  LivePerson chat (30-min window)                                          │
│         │                                                                 │
│    ┌────┴──────────────┬────────────────────┬──────────────────┐          │
│    ▼                   ▼                    ▼                  │          │
│ ┌──────────┐    ┌──────────────┐    ┌──────────────┐           │          │
│ │ SIGNAL 1 │    │  SIGNAL 2    │    │  SIGNAL 3    │           │          │
│ │ TOPIC    │    │  MOVING      │    │  Z-SCORES    │           │          │
│ │ MODELLING│    │  AVERAGES    │    │              │           │          │
│ │          │    │              │    │              │           │          │
│ │ "WHAT    │    │ "WHAT IS     │    │ "HOW UNUSUAL │           │          │
│ │  are they│    │  NORMAL for  │    │  is this, in │           │          │
│ │  talking │    │  this hour?" │    │  σ units?"   │           │          │
│ │  about?" │    │              │    │              │           │          │
│ └────┬─────┘    └──────┬───────┘    └──────┬───────┘           │          │
│      │                 │                   │                   │          │
│      │  topic weights  │  SMA baseline     │  z per keyword    │          │
│      │  topic entropy  │  EMA fast-track   │  z per topic      │          │
│      │                 │  % change vs SMA  │  z per volume     │          │
│      └─────────────────┴───────────────────┴───────────────────┘          │
│                              │                                            │
│                              ▼                                            │
│              ┌──────────────────────────────────┐                         │
│              │  SIGNAL 4 — ISOLATION FOREST     │                         │
│              │  THE COMBINER                    │                         │
│              │                                  │                         │
│              │  ~25-30 numeric features/window  │                         │
│              │  → single anomaly score          │                         │
│              │                                  │                         │
│              │  "Is this COMBINATION of         │                         │
│              │   signals unusual, even if no    │                         │
│              │   single one is extreme?"        │                         │
│              └──────────────┬───────────────────┘                         │
│                             ▼                                             │
│              ┌──────────────────────────────────┐                         │
│              │  CONFIRMATION GATE               │                         │
│              │  IF + persistence + topic-coherent│                        │
│              │  → ALERT (+ playbook)            │                         │
│              └──────────────────────────────────┘                         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

### 2.7.1 LivePerson Context: What We Were Actually Watching

Before the algorithms, the substrate. Getting this wrong in an interview undermines everything after it.

**What LivePerson is:** the enterprise live-chat platform the contact centre ran on. Customers open a chat from the website or app; an agent (or a bot, then an agent) responds in real time. Every conversation is stored as a nested JSON document containing the full message-by-message transcript, participant metadata, timestamps, campaign attribution and a post-chat CSAT score.

**Why chat and not calls:** chat is *faster-moving* than voice. A customer who sees a "flight cancelled" notification will open a chat within seconds — they don't wait on hold. Chat volume therefore leads call volume by a meaningful margin during a disruption, which makes it the better early-warning channel. It's also already text, so there's no ASR error layer between the customer and the model.

**What an "operational anomaly" actually looks like in the data.** This is the concrete detail that proves you saw the data:

| Signature | What It Looks Like in the Window | Example Cause |
|---|---|---|
| **Volume surge** | Chat count jumps well above the SMA baseline for that hour-of-week | Mass cancellation event |
| **Topic convergence** | Topic entropy collapses — one topic's weight goes from ~2% to 40%+ | Volcanic ash closing airspace |
| **Keyword ignition** | A normally-rare term ("strike", "volcano", "evacuate") spikes several σ above its rolling baseline | Industrial action announced |
| **Sentiment cliff** | Average window sentiment drops sharply alongside volume | Service failure |
| **Handling-time blowout** | Average conversation duration rises as agents hit questions they can't answer | Policy change with no guidance issued |
| **Queue/abandon shift** | Abandonment rate climbs as wait times grow | Under-staffing relative to a demand shock |

Critically, a **real event usually shows several of these at once**, whereas a **false positive usually shows exactly one**. That observation is the entire justification for the ensemble — it's not architectural elegance, it's the empirical structure of the failure modes.

---

### 2.7.2 Signal 1 — Topic Modelling (What Are They Talking About?)

**Algorithm chosen: NMF over TF-IDF.** LDA was fitted in parallel as a cross-check, and BERTopic was evaluated and rejected.

**Why NMF specifically:**

| Reason | Detail |
|---|---|
| **Short-text robustness** | Chat messages are short. LDA's Dirichlet priors assume documents long enough to contain a mixture; on a two-sentence message, LDA's topic assignments are noisy. NMF's matrix factorisation has no such assumption. |
| **Determinism** | Given a fixed initialisation (`nndsvda`), NMF converges to the same factorisation. For a production system retrained monthly, reproducible topics matter — the ops team's playbook mapping is keyed to topic IDs. |
| **Speed** | Deterministic optimisation, not iterative sampling. Fast enough to refit monthly on months of history without a compute budget conversation. |
| **Interpretability** | Crisp, near-orthogonal topics. Operational leaders could read "volcano, eruption, ash, airspace" and immediately know what it meant. |

**Why not BERTopic**, despite it being the stronger method on paper: it's substantially heavier (transformer embeddings → UMAP → HDBSCAN → c-TF-IDF), the cluster count is not directly controllable which breaks the fixed topic-ID-to-playbook mapping, and the pipeline is much harder to explain to a non-technical stakeholder who has to trust the alert. It is the obvious upgrade if semantic nuance ever became the bottleneck — it would catch "my holiday is ruined" as related to "vacation disaster" without shared keywords, which TF-IDF cannot.

**How topics became features.** This is the step people skip, and it's the one that matters:

```python
# Per time window, the topic model contributes a fixed-length feature vector.
def topic_features(window_messages, tfidf, nmf, disaster_topic_ids):
    X = tfidf.transform(window_messages)          # reuse FITTED vectorizer
    W = nmf.transform(X)                          # doc-topic weights

    # Average topic distribution across all messages in the window
    dist = W.mean(axis=0)
    dist = dist / (dist.sum() + 1e-9)             # normalise to proportions

    feats = {f"topic_{i}_weight": dist[i] for i in range(len(dist))}

    # Concentration: a collapsing entropy means the room is converging
    # on ONE subject — the single most reliable event signature.
    feats["topic_entropy"] = float(-(dist * np.log(dist + 1e-9)).sum())
    feats["max_topic_weight"] = float(dist.max())

    # Domain-guided aggregate over known disaster themes
    feats["disaster_topic_mass"] = float(dist[disaster_topic_ids].sum())
    return feats
```

**Topic entropy is the underrated feature.** Volume tells you *how many* people are chatting; entropy tells you whether they're chatting about *the same thing*. A marketing campaign produces high volume with normal entropy (people ask about deals, baggage, seats, everything). A disaster produces high volume with **collapsing entropy** — everyone suddenly talking about one subject. That single feature does a lot of the work separating the two.

---

### 2.7.3 Signal 2 — Moving Averages (What Is Normal Right Now?)

Raw counts are meaningless without a baseline. 200 chats in a window is a crisis at 3am and a quiet Tuesday at 11am. Moving averages supply the "compared to what?"

**SMA — the stable baseline:**

$$
\text{SMA}_t = \frac{1}{n}\sum_{i=0}^{n-1} x_{t-i}
$$

**EMA — the fast responder:**

$$
\text{EMA}_t = \alpha x_t + (1-\alpha)\,\text{EMA}_{t-1}, \qquad \alpha = \frac{2}{n+1}
$$

**Why both, rather than picking one.** They fail in opposite directions, which is exactly why running them together is informative:

| | SMA | EMA |
|---|---|---|
| **Weighting** | Uniform across the window | Exponentially decaying toward the present |
| **Lag** | High | Low |
| **Robustness to a single spike** | High — one outlier is diluted by $n$ | Low — a single spike moves it immediately |
| **Role here** | Defines "normal". The denominator for % change, the $\mu$ for z-scores. | Detects that something is changing *now*. |

**The divergence is itself a feature.** When EMA rises sharply while SMA is still flat, that gap is the signature of a ramp-up in progress — the earliest detectable moment of an event, before the level has moved enough to be extreme.

```python
df["kw_sma"] = df["kw_freq"].rolling(48, min_periods=10).mean()
df["kw_ema"] = df["kw_freq"].ewm(span=12, adjust=False).mean()
# The early-warning feature: fast tracker pulling away from slow baseline
df["ema_sma_divergence"] = (df["kw_ema"] - df["kw_sma"]) / (df["kw_sma"] + 1e-9)
```

**Window choice, and the reasoning behind it:**

| Window | Duration (30-min bins) | Why This Length |
|---|---|---|
| 24 bins | 12 hours | Short enough to react within half a day; used for rapid-onset detection |
| **48 bins** | **24 hours** | **Primary.** Spans exactly one daily cycle, so the baseline includes the full diurnal pattern. A shorter window would treat every evening lull as a deviation. |
| 336 bins | 7 days | Captures weekday/weekend structure. Prevents "Monday is anomalous" — which is what you get if your baseline is shorter than your dominant seasonality. |

> **The principle worth stating:** the baseline window must be at least one full period of the dominant seasonality, or the seasonality itself gets detected as anomaly. Our dominant cycles were daily and weekly, so we needed 24h and 7d baselines.

---

### 2.7.4 Signal 3 — Z-Scores (How Unusual, In Standard Deviations?)

Moving averages give you a baseline; z-scores make deviations from it **comparable across features**.

$$
z_t = \frac{x_t - \mu_{\text{rolling}}}{\sigma_{\text{rolling}}}
$$

**Why rolling and not global.** A global $\mu$ and $\sigma$ computed over the whole history bakes in every past event and every seasonal regime. Rolling statistics ask the sharper question: "is this unusual *given the last 24 hours*?" That's the right question for detecting a change, and it self-adjusts as traffic patterns evolve.

**Why z-scores are the thing that makes the ensemble possible.** Chat volume is in the hundreds. Keyword counts are in the single digits. Topic weights are in $[0,1]$. Feeding those raw into one model means volume dominates by sheer magnitude. Z-scoring maps all of them onto a common "how many standard deviations from your own normal" scale — every feature gets to speak at the same volume.

**Threshold choice:**

| Threshold | Two-tailed probability under normality | Our use |
|---|---|---|
| $\|z\| > 2$ | ~5% | Too loose. At ~48 windows/day across dozens of features you'd trip this constantly. |
| $\|z\| > 3$ | ~0.3% | **Chosen** as the strong-signal marker. Rare enough to be meaningful, frequent enough to catch real ramps. |
| $\|z\| > 4$ | ~0.006% | Only for the most extreme confirmation logic. Too conservative alone. |

**The honest caveat you should volunteer:** those probabilities assume normality, and count data in short windows is **not** normal — it's closer to Poisson or negative-binomial, and it's right-skewed. Three sigma on skewed data does not mean 0.3%. Three mitigations, all of which are real practice:

1. **Treat $3\sigma$ as a tuned threshold, not a probability claim.** The value was validated empirically against the labelled event history; the normal-theory number is a starting point, not a justification.
2. **Variance-stabilise.** A $\log(1+x)$ or Anscombe transform on counts before z-scoring pulls the distribution closer to symmetric.
3. **Guard the denominator.** In quiet windows $\sigma \to 0$ and the z-score explodes on trivial fluctuations. We floored $\sigma$ and required a minimum count before trusting a z-score at all.

```python
def robust_rolling_z(series, window=48, min_periods=10, sigma_floor=1e-3):
    """Rolling z-score with variance stabilisation and a denominator guard."""
    x = np.log1p(series)                                   # stabilise count variance
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std()
    sd = sd.clip(lower=sigma_floor)                        # never divide by ~0
    return ((x - mu) / sd).fillna(0.0)
```

**A robust alternative worth naming:** the **modified z-score**, which swaps mean and standard deviation for median and median absolute deviation:

$$
z_{\text{mod}} = \frac{0.6745\,(x - \tilde{x})}{\text{MAD}}
$$

This matters because the classical z-score is contaminated by the very outliers it's meant to detect — a large spike inflates $\sigma$ in the same rolling window, suppressing its own z-score. The MAD version doesn't have that problem. Knowing this distinction is a good signal in an interview.

---

### 2.7.5 Signal 4 — Isolation Forest (The Combiner)

The first three signals are univariate. Each answers "is *this one thing* unusual?" Isolation Forest answers the question none of them can: **"is this *combination* unusual, even when no individual component is extreme?"**

**How it works.** Anomalies are *few and different*, so they're easy to isolate by random partitioning. Build many trees; at each node pick a random feature and a random split value between that feature's min and max; recurse until each point is alone. Points isolated in few splits are anomalous.

$$
s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}, \qquad c(n) = 2H(n-1) - \frac{2(n-1)}{n}
$$

where $E[h(x)]$ is the average path length to isolate $x$ and $c(n)$ normalises by the expected path length in a random binary tree. Score near 1 is a strong anomaly; near 0.5 is normal.

**Why Isolation Forest suits high-dimensional chat features specifically:**

| Property | Why It Matters Here |
|---|---|
| **No distance metric** | Our ~25–30 features are heterogeneous — topic proportions, z-scores, counts, binary flags, cyclical time encodings. Any Euclidean distance over that mix is close to meaningless. IF never computes one; it only ever compares a single feature to a single threshold. |
| **Linear time complexity** | $O(t \cdot n \log n)$. LOF and One-Class SVM are $O(n^2)$-to-$O(n^3)$ — non-starters for a job re-scoring every 30 minutes. |
| **Subsampling is a feature, not a limitation** | `max_samples < 1.0` actually *improves* anomaly detection by reducing swamping and masking, unlike most models where less data is strictly worse. |
| **Naturally handles the imbalance** | It doesn't need positive examples. Disaster events are ~1% of windows; a supervised classifier on ~15 labelled events per year would overfit instantly. |
| **Few hyperparameters** | Four that matter. With almost no labels to tune against, a low-dimensional hyperparameter space is a genuine advantage. |
| **Captures feature interactions** | The core value-add: "moderate volume rise AND moderate entropy drop AND moderate keyword lift" is anomalous as a *conjunction* while no single term crosses $3\sigma$. That's the case the univariate signals structurally cannot see. |

**The `contamination` parameter — the most important knob, and the one interviewers probe.** It is the assumed proportion of anomalies, and it directly sets the score threshold that converts a continuous anomaly score into a binary label.

```python
final_iso = IsolationForest(
    n_estimators=200,      # enough trees for score stability; returns flatten after ~200
    max_samples=0.75,      # subsample per tree — reduces masking, adds diversity
    contamination=0.01,    # ~1% of windows expected anomalous (domain prior, grid-refined)
    max_features=0.75,     # feature subsampling — decorrelates trees
    random_state=42,
    n_jobs=-1,
)
```

**Contamination is effectively a specificity dial**, and saying so out loud shows you understand the mechanism rather than the API:

$$
\text{contamination} \approx \text{fraction flagged} \;\Longrightarrow\; \text{specificity} \approx 1 - \text{contamination (on mostly-normal data)}
$$

Setting `contamination=0.01` on data that is ~99% normal caps the false positive rate near 1% *before* any confirmation logic. Loosen it to 0.05 and specificity cannot exceed ~95% no matter how good the features are. This is the single most direct lever on the 96%.

**A caveat to volunteer:** Isolation Forest uses **axis-aligned splits**. An anomaly that is only unusual along a diagonal in feature space — normal volume, normal entropy, but an unusual *ratio* between them — is hard for it to isolate. Two mitigations: engineer the interaction explicitly as a feature (which we did, via divergence and rate-of-change terms), or use **Extended Isolation Forest**, which splits on random hyperplanes rather than single axes.

---

### 2.7.6 How the Four Signals Were Combined

Not a vote among four equals. A **two-stage architecture**: the Isolation Forest fuses the signals into one score, and then a confirmation gate converts score into alert.

```
STAGE 1 — FEATURE FUSION (inside the Isolation Forest)
──────────────────────────────────────────────────────
  Topic weights + entropy      ─┐
  SMA / EMA / divergence       ─┼──► ~25-30 features ──► IF ──► anomaly score
  Rolling z-scores             ─┤                              (continuous, 0-1)
  Volume, temporal, rate terms ─┘

STAGE 2 — CONFIRMATION GATE (rule layer on top of the score)
────────────────────────────────────────────────────────────
  ALERT  IF   anomaly_score  > τ_high
         AND  persistence    ≥ 2 consecutive windows
         AND  topic_coherent (a disaster-family topic is
                              among the top contributors)

  WATCH  IF   anomaly_score  > τ_low   (dashboard only, no notification)

  Everything else → silent.
```

**Why a rule layer on top of a model, rather than just trusting the score?** Because the three gate conditions each kill a *specific, observed* false-positive mode, and each one is cheap:

| Gate | The False Positive It Kills | Cost |
|---|---|---|
| **Score threshold $\tau_{\text{high}}$** | Marginal windows that barely cross the contamination boundary | Loses the weakest true signals |
| **Persistence ($\geq 2$ windows)** | Single-window blips — a burst of chats from one frustrated customer, a brief platform glitch, a bot loop. Real events *sustain*; noise doesn't. | Adds ~30 minutes of detection latency |
| **Topic coherence** | High-volume events with the wrong content — the flash-sale problem. A marketing campaign spikes volume and concentrates topics, but concentrates them on *deal/price/booking*, not *cancel/delay/disruption*. | Would miss a genuinely novel event type whose vocabulary sits outside every known topic |

**Persistence is the highest-leverage rule.** It is nearly free in latency terms (one extra 30-minute window against an event that unfolds over hours) and it removes a large share of false positives, because the defining property of noise is that it doesn't repeat.

> **Say this:** "It wasn't a vote — that would waste information by binarising each signal before combining. The Isolation Forest does the fusion on continuous features, so it can see that three signals are each *mildly* off simultaneously, which is a stronger indicator than any one being extreme. Then a thin rule layer sits on top: score threshold, plus persistence across two consecutive windows, plus a check that a disruption-family topic is actually driving the score. Persistence alone removed a large chunk of the false alarms, because real events sustain and noise doesn't. That two-stage design is where most of the specificity came from — not from the model, from the confirmation logic."

**Explainability came free.** Because the gate names *which* features drove the score, every alert shipped with its reasoning: "volume z = 4.2, topic 7 (strike) weight 0.38 vs 0.02 baseline, entropy down 61%, sustained 3 windows." The ops team could sanity-check an alert in five seconds. That is a large part of why they trusted it.

---

### 2.7.7 Why Specificity (96%) Was Optimised Rather Than Recall

This is the judgement call the resume bullet is really about, and it needs a crisp answer.

**The confusion matrix, and which cell hurts:**

$$
\text{Specificity} = \frac{TN}{TN + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad \text{Precision} = \frac{TP}{TP + FP}
$$

**Argument 1 — Alert fatigue is the failure mode that destroys the system permanently.** A missed event is a *bad day*. A stream of false alarms is a *dead product*. Once an ops team learns that the alert channel is mostly noise, they mute it, and then the system's recall on paper is irrelevant because nobody is reading the output. Missed alerts degrade gracefully; lost trust does not come back. Alert fatigue is well documented in clinical monitoring and in security operations for exactly this reason.

**Argument 2 — The base rate makes false positives numerically overwhelming.** Events are roughly 1% of windows. With ~48 windows a day:

```
At 90% specificity:  ~47.5 normal windows × 10%  ≈ 5 false alerts/day
At 96% specificity:  ~47.5 normal windows ×  4%  ≈ 2 false alerts/day

Meanwhile true events:  ~10-15 per YEAR.
```

So even at excellent specificity, **the overwhelming majority of alerts are false positives** — that's what a 1% base rate does to you, and it's the same arithmetic as the classic medical-screening paradox. Precision is structurally low here no matter how good the model is, which is precisely why the false positive rate has to be driven down hard.

**Argument 3 — Recall had a safety net; specificity did not.** The ops team had other routes to event awareness: news feeds, airline notifications, government travel advisories, airport operations contacts. Our system's job was to be *earlier*, not to be the only detector. A missed automated alert usually meant finding out through another channel thirty minutes later. A false alert had no compensating mechanism — it just consumed attention and eroded credibility.

**Argument 4 — The asymmetry was validated with the users, not assumed.** The 60/40 weighting in the grid search objective (`0.6 × specificity + 0.4 × recall`) came from conversations with operational leaders about what they'd tolerate, not from a default.

**The honest counter-argument, which you should raise yourself.** This trade is only correct because of the safety net. If this system were the *sole* detector — a safety-critical monitor with no alternative channel — the calculus inverts completely and you optimise recall, accept the noise, and invest in triage tooling instead. The right answer depends entirely on what happens when you miss, and I'd want to re-derive it for any new context rather than carrying the 60/40 over.

> **Say this:** "I optimised specificity because the dominant failure mode of an alerting system isn't a missed event, it's being ignored. With events at roughly one percent of windows, even at 96% specificity most alerts are still false positives — that's just the base rate arithmetic. If I'd tuned for recall the team would have muted the channel inside a fortnight and the recall would have become irrelevant. The reason I'm comfortable with that trade specifically is that the ops team had other ways of finding out about a volcanic eruption; they had no other way of getting their attention back once we'd spent it. If this had been the only detector, I'd have made the opposite call."

---

# 3. Key Metrics & Results

---

## 3.1 Primary Metrics

| Metric | Value | Meaning |
|:-------|:------|:--------|
| **Specificity** | **96%** | 96% of non-event periods correctly identified as normal — **this is the resume number** |
| **False Positive Rate** | **4%** | $1 - \text{specificity}$. Roughly one false alert per day — the "reduced false-positive operational alerts" claim |
| **Recall (sensitivity)** | Deliberately traded down — see §2.7.7 and Trick Q1 | We optimised the specificity/recall trade-off toward specificity; I discuss the trade rather than quoting a fabricated recall figure |
| **Event Detection Coverage** | High on significant events | Validated against 1 year of historical events — caught the ones that mattered operationally |
| **Detection Latency** | Minutes | Events flagged within minutes of chat spikes beginning |
| **Ensemble signals combined** | 4 | Topic modelling + moving averages + z-scores + Isolation Forest (§2.7) |

## 3.2 Business Impact

| Impact Area | Before (Reactive) | After (Proactive) |
|:------------|:-------------------|:-------------------|
| **Event awareness** | Discovered events after customer flood | Alerted within minutes of emerging pattern |
| **Response time** | Hours to mobilize | Pre-positioned with playbooks |
| **Customer experience** | Long wait times during events | Faster, informed responses |
| **Operational posture** | Reactive firefighting | Proactive, data-driven response |
| **Decision support** | Gut feel + delayed reports | Real-time dashboards + prescriptive playbooks |

## 3.3 Prescriptive Playbooks

The detected event type (from topic modeling) informed **which playbook** to activate:

| Detected Theme | Playbook Triggered |
|:---------------|:-------------------|
| Volcanic activity | Rerouting protocols, alternative airports, customer comms template |
| Industrial action / strike | Strike impact assessment, rebooking automation, FAQs |
| Severe weather | Weather monitoring escalation, delay comms, hotel provisions |
| Pandemic restrictions | Country-specific guidance, refund/change policy activation |

> **Interview framing:** "The system didn't just detect anomalies — the topic modeling component told us *what kind* of event was emerging, which allowed us to trigger the right prescriptive playbook automatically."

---

# 4. Topics You Must Know

---

## 4.1 Anomaly Detection Algorithms

### 4.1.1 Isolation Forest (Deep Dive)

**Algorithm steps:**
1. Sample a subset of data ($\text{max\_samples}$)
2. Randomly select a feature
3. Randomly select a split value between feature min and max
4. Recursively split until point is isolated or max depth ($\lceil \log_2 n \rceil$)
5. Record path length $h(x)$
6. Repeat for $\text{n\_estimators}$ trees
7. Average path lengths → anomaly score

**Key properties:**
- **Time complexity:** $O(t \cdot n \cdot \log n)$ where $t$ = number of trees, $n$ = sample size
- **Space complexity:** $O(t \cdot n)$
- **No distance metric required** — works on raw feature space
- **Linear time** — much faster than LOF or OCSVM

**Strengths:**
- Scales to large datasets
- Handles high-dimensional data better than distance-based methods
- Naturally handles mixed feature types (after scaling)
- Minimal hyperparameters
- Can detect global anomalies effectively

**Weaknesses:**
- Axis-aligned splits — struggles with anomalies that are only detectable in rotated feature spaces (Extended Isolation Forest addresses this)
- Not ideal for local anomalies in varying-density data
- `contamination` parameter needs to be set (or estimated)
- Doesn't capture sequential/temporal dependencies directly

### 4.1.2 Local Outlier Factor (LOF)

**How it works:**
- Computes local density for each point based on $k$-nearest neighbors
- Compares each point's density to its neighbors' density
- Points with significantly lower density than neighbors → anomalies

$$
\text{LOF}(x) = \frac{\sum_{o \in N_k(x)} \frac{\text{lrd}(o)}{\text{lrd}(x)}}{|N_k(x)|}
$$

where $\text{lrd}(x)$ = local reachability density.

- LOF > 1: less dense than neighbors (potential outlier)
- LOF ≈ 1: similar density to neighbors (normal)
- LOF < 1: denser than neighbors (inlier)

**vs Isolation Forest:** LOF is better at detecting **local anomalies** (unusual relative to their neighborhood) but is $O(n^2)$ — too slow for our use case.

### 4.1.3 One-Class SVM

**How it works:**
- Learns a boundary (hyperplane in kernel space) that encloses the "normal" data
- Points outside the boundary → anomalies
- Kernel trick allows nonlinear boundaries

**Key hyperparameters:** `nu` (upper bound on fraction of outliers), `kernel` (rbf, poly), `gamma`

**vs Isolation Forest:** Powerful with kernel trick but $O(n^2)$ to $O(n^3)$ training time. Not practical for our data size.

### 4.1.4 DBSCAN for Outlier Detection

**How it works:**
- Density-based clustering: groups points in dense regions
- Points not belonging to any cluster → outliers (label = -1)
- Two parameters: `eps` (neighborhood radius), `min_samples`

**vs Isolation Forest:** DBSCAN is primarily a clustering algorithm; outlier detection is a byproduct. It doesn't produce anomaly *scores*, just binary cluster/outlier labels. Less suitable for our graduated alerting needs.

---

## 4.2 Topic Modeling Algorithms

### 4.2.1 LDA (Latent Dirichlet Allocation)

**Generative model:**
1. For each document, draw a topic distribution $\theta_d \sim \text{Dir}(\alpha)$
2. For each word in the document:
   - Draw a topic $z \sim \text{Multinomial}(\theta_d)$
   - Draw a word $w \sim \text{Multinomial}(\beta_z)$

**Key concepts:**
- $\alpha$: Dirichlet prior on per-document topic distribution (low → documents have few topics)
- $\beta$: Dirichlet prior on per-topic word distribution (low → topics have few dominant words)
- Inference: Variational Bayes or Gibbs sampling

**Strengths:** Principled probabilistic model, well-understood.  
**Weaknesses:** Slow inference, bag-of-words assumption, struggles with short texts.

### 4.2.2 NMF (Non-negative Matrix Factorization)

**Matrix decomposition:**

$$
V \approx W \cdot H
$$

where:
- $V$: document-term matrix ($n \times m$) — TF-IDF weighted
- $W$: document-topic matrix ($n \times k$)
- $H$: topic-term matrix ($k \times m$)
- All entries non-negative (natural for word frequencies)

**Optimization:** Minimize $\|V - WH\|_F^2$ subject to $W, H \geq 0$

**Strengths:** Fast, deterministic (given initialization), produces crisp topics.  
**Weaknesses:** Not probabilistic, sensitive to initialization, topics can be less coherent than LDA for long documents.

### 4.2.3 BERTopic

**Pipeline:**
1. Generate document embeddings (BERT, Sentence-BERT)
2. Reduce dimensionality (UMAP)
3. Cluster (HDBSCAN)
4. Extract topic representations (c-TF-IDF per cluster)

**Strengths:** Captures semantic meaning, handles synonyms, state-of-the-art topic quality.  
**Weaknesses:** Computationally expensive, less interpretable pipeline, harder to explain to non-technical stakeholders.

---

## 4.3 Statistical Process Control

### 4.3.1 Z-Scores and Control Charts

**Z-score** standardizes a value relative to its distribution:

$$
z = \frac{x - \mu}{\sigma}
$$

**Control charts** (Shewhart charts) plot a metric over time with:
- Center line: $\mu$
- Upper control limit (UCL): $\mu + 3\sigma$
- Lower control limit (LCL): $\mu - 3\sigma$

Points outside UCL/LCL → process is "out of control" → alert.

### 4.3.2 EWMA (Exponentially Weighted Moving Average) Control Chart

More sensitive to small persistent shifts than Shewhart charts:

$$
Z_t = \lambda x_t + (1 - \lambda) Z_{t-1}
$$

where $\lambda \in (0, 1]$ is the weighting factor (typically 0.05–0.25).

**Control limits:**

$$
\text{UCL/LCL} = \mu \pm L \cdot \sigma \sqrt{\frac{\lambda}{2 - \lambda}\left[1 - (1-\lambda)^{2t}\right]}
$$

where $L$ is a width parameter (typically 2.7–3.0).

**Relevance to project:** EWMA is essentially the statistical foundation of our EMA-based baseline tracking.

---

## 4.4 Time Series Anomaly Detection

| Method | Type | How It Works |
|:-------|:-----|:-------------|
| **Z-score (rolling)** | Statistical | Flag points beyond $n\sigma$ from rolling mean |
| **EWMA control chart** | Statistical | Detect mean shifts with exponential smoothing |
| **Isolation Forest** | ML | Random partitioning, path length scoring |
| **Prophet anomaly** | Decomposition | Facebook Prophet decomposes trend + seasonality, flags residuals |
| **LSTM Autoencoder** | Deep Learning | Train on normal sequences, flag high reconstruction error |
| **Seasonal Hybrid ESD** | Statistical | Twitter's method — seasonal decomposition + generalized ESD test |

---

## 4.5 Precision vs Recall vs Specificity vs Sensitivity

| Metric | Formula | Question It Answers |
|:-------|:--------|:-------------------|
| **Precision** | $\frac{TP}{TP+FP}$ | Of all things I flagged, how many were real? |
| **Recall (Sensitivity)** | $\frac{TP}{TP+FN}$ | Of all real events, how many did I catch? |
| **Specificity** | $\frac{TN}{TN+FP}$ | Of all non-events, how many did I correctly ignore? |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean of precision and recall |
| **FPR** | $\frac{FP}{FP+TN} = 1 - \text{Specificity}$ | False alarm rate |

**Full confusion matrix:**

```
                       Predicted Positive    Predicted Negative
Actual Positive             TP                     FN
Actual Negative             FP                     TN

Accuracy    = (TP + TN) / (TP + TN + FP + FN)
Precision   = TP / (TP + FP)
Recall      = TP / (TP + FN)        ← also called Sensitivity, TPR
Specificity = TN / (TN + FP)        ← also called TNR
FPR         = FP / (FP + TN)        = 1 - Specificity
```

---

## 4.6 Grid Search vs Random Search vs Bayesian Optimization

| Method | Strategy | Pros | Cons |
|:-------|:---------|:-----|:-----|
| **Grid Search** | Exhaustive search over all parameter combinations | Guaranteed to find best in grid; reproducible | Exponential cost with dimensions; wastes budget on unimportant params |
| **Random Search** | Random sampling from parameter distributions | More efficient than grid; covers space better; finds good configs faster | No guarantee of optimality; less systematic |
| **Bayesian Optimization** | Builds surrogate model (GP), selects next point to maximize expected improvement | Most sample-efficient; models parameter interactions | Complex to implement; overhead per iteration; can get stuck |

> **In this project:** Grid search was feasible because Isolation Forest has few hyperparameters (4 main ones) and training is fast. For higher-dimensional hyperparameter spaces, I'd switch to Bayesian optimization (e.g., `optuna`).

---

## 4.7 Moving Averages: SMA vs EMA

| Property | SMA | EMA |
|:---------|:----|:----|
| **Formula** | $\frac{1}{n}\sum_{i=0}^{n-1}x_{t-i}$ | $\alpha x_t + (1-\alpha)\text{EMA}_{t-1}$ |
| **Weight distribution** | Equal weight to all $n$ points | Exponentially decaying weights |
| **Lag** | Higher lag (all points equal) | Lower lag (more recent = more weight) |
| **Smoothness** | Smoother | Less smooth but more responsive |
| **Use case** | Stable baseline estimation | Quick detection of trend changes |
| **Sensitivity to outliers** | Less sensitive (averaged over window) | More sensitive to recent outliers |

---

## 4.8 Feature Engineering for Text-Based Anomaly Detection

| Feature Category | Examples | Rationale |
|:-----------------|:---------|:----------|
| **Volume features** | Chat count per window, unique conversation count | Raw demand signal |
| **Keyword frequencies** | Count of "cancel", "delay", "volcano" per window | Direct disaster signal |
| **Topic proportions** | NMF topic weights per window | Thematic signal — what people are talking about |
| **Statistical derivatives** | Z-scores, EMA, % change of the above | Normalized signals that capture *change from normal* |
| **Temporal features** | Hour, day-of-week, is_weekend, is_holiday | Control for expected patterns (e.g., higher volume on Mondays) |
| **Rate features** | Volume acceleration (2nd derivative), topic shift rate | How *fast* things are changing, not just current level |
| **Entropy features** | Entropy of topic distribution | Sudden focus on one topic = low entropy = potential event |
| **Sentiment features** | Average sentiment score per window | Negative sentiment spike may accompany disaster |

---

## 4.9 Real-Time Data Processing Architectures

| Architecture | Description | Latency | Complexity | Project Fit |
|:-------------|:------------|:--------|:-----------|:------------|
| **Batch** | Process accumulated data at intervals (hourly, daily) | High | Low | ❌ Too slow for disaster detection |
| **Micro-batch** | Process small batches frequently (every 5–30 min) | Medium | Medium | ✅ **Used in this project** |
| **Stream processing** | Process each event as it arrives (Kafka, Flink, Spark Streaming) | Low (sub-second) | High | ⚡ Ideal but over-engineered for this volume |
| **Lambda architecture** | Batch + stream processing in parallel | Low | Very High | ❌ Unnecessary complexity |
| **Kappa architecture** | Stream-only, replay from log | Low | High | ❌ Requires full streaming infra |

> **Why micro-batch:** "Disaster events unfold over minutes to hours, not seconds. A 15–30 minute micro-batch cycle gave us detection within the operationally useful time window without the complexity of full streaming infrastructure."

---

# 5. Interview Questions (35+) with Model Answers

---

### Q1: "Walk me through this project end to end."

> **Answer:** "At Jet2, the operations team was reactive to disaster events — they'd only realize a volcanic eruption or airline strike was impacting customers after support lines were already overwhelmed. My task was to build a system that detected these events in near-real-time from customer chat data.
>
> I started with the data pipeline: LivePerson exports deeply nested JSON, so I built a parser that flattened these into structured, timestamped message records. Working with the data engineering team, we set this up to run in frequent micro-batch cycles.
>
> Next, I applied NMF topic modeling on TF-IDF-vectorized chat text to discover recurring themes — the model surfaced clusters like volcanic activity, strikes, severe weather, and pandemic restrictions. These topics became a feature layer.
>
> On top of that, I built a statistical signal-processing layer: rolling z-scores, exponential moving averages, percentage changes from baseline, and IQR-based outlier flags — all computed per keyword and per topic, per time window.
>
> These statistical features, combined with topic proportions and volume metrics, fed into an Isolation Forest model. I chose IF because it handles mixed features well, scales linearly, and requires minimal tuning. I validated against a year of historical events and used grid search to optimize hyperparameters, prioritizing specificity to avoid alert fatigue.
>
> The result was 96% specificity — a four percent false-alarm rate, roughly one alert a day. That was the whole point: the ops team got a small number of high-quality alerts instead of a stream of noise, and the topic modelling told them *what kind* of event was happening, triggering the right prescriptive playbook automatically. This turned a reactive firefighting process into a proactive, data-driven one."

---

### Q2: "How does Isolation Forest work?"

> **Answer:** "Isolation Forest is based on the insight that anomalies are few and different, so they're easier to isolate through random partitioning.
>
> The algorithm builds an ensemble of isolation trees. Each tree randomly selects a feature and a random split value, recursively partitioning the data until each point is isolated. Anomalies, being rare and distinct, get isolated in fewer splits — they have shorter path lengths. Normal points, being in dense clusters, require many more splits.
>
> The anomaly score is computed as $s(x, n) = 2^{-E[h(x)]/c(n)}$, where $E[h(x)]$ is the average path length across all trees and $c(n)$ is a normalization factor. A score near 1 indicates a strong anomaly; near 0.5 is normal.
>
> The key hyperparameters are: `n_estimators` (number of trees), `max_samples` (data fraction per tree), `contamination` (expected anomaly proportion), and `max_features` (feature fraction per tree). In our case, we used 200 trees with 0.75 sample fraction and 1% contamination rate, tuned via grid search."

---

### Q3: "Why Isolation Forest over other anomaly detection methods?"

> **Answer:** "Three main reasons:
>
> First, **scalability**: we were processing frequent micro-batches of chat data. LOF is $O(n^2)$ and One-Class SVM is $O(n^2)$ to $O(n^3)$ — both too slow. Isolation Forest is $O(n \log n)$ per tree, making it practical.
>
> Second, **feature flexibility**: our feature set was mixed — topic proportions, z-scores, counts, binary flags. IF doesn't require a distance metric and works directly on the feature space, unlike LOF which is distance-dependent.
>
> Third, **simplicity and interpretability**: IF has only a few hyperparameters and produces a clear anomaly score. For a system that needed operational trust, being able to explain 'this data point was isolated quickly because its combination of features was unusual' was valuable.
>
> We also considered autoencoders, but they added complexity without clear benefit given the relatively structured nature of our engineered features. IF was the right tool for the data and the constraints."

---

### Q4: "What features did you engineer?"

> **Answer:** "I engineered features across four categories:
>
> **Volume features:** Raw chat count per time window, unique conversation count, and their statistical derivatives (z-scores, EMA, percentage change from baseline, IQR outlier flags).
>
> **Topic features:** NMF topic model weights per time window — especially for disaster-related topics. I also computed topic entropy (a sudden drop in entropy means the conversation is converging on one topic, which is a disaster signal).
>
> **Keyword features:** Frequency, z-scores, and EMAs for specific disaster-related keywords like 'cancel', 'delay', 'volcano', 'strike', 'storm'. These were identified partly from the topic model's top words and partly from domain knowledge.
>
> **Temporal features:** Hour of day, day of week, weekend flag, holiday flag — critical for controlling against expected patterns. Monday mornings naturally have higher chat volume; without temporal features, every Monday would be flagged."

---

### Q5: "How did you define 'disaster event'?"

> **Answer:** "This was a nuanced definitional challenge. We defined a disaster event as any external occurrence that causes a statistically significant departure from normal customer contact patterns — both in volume and in topic distribution.
>
> Practically, we validated this definition by working backwards from known historical events: volcanic eruptions (like Etna), airline strikes (baggage handler strikes, ATC actions), severe weather (storms causing mass cancellations), and pandemic restriction changes. These were events where the operations team needed to shift into crisis mode.
>
> The key insight was that we didn't need to define every possible disaster in advance — the unsupervised approach (topic modeling + anomaly detection) would surface *any* significant departure from normal, regardless of the specific cause. The topic model then helped *classify* what kind of event it was."

---

### Q6: "Why specificity and not recall?"

> **Answer:** "In an alerting system, the greatest operational risk is alert fatigue. If the system produces too many false positives, the operations team will start ignoring alerts — and then miss real events.
>
> Specificity measures how well we identify *non-events* as non-events. High specificity means low false positive rate, which means the alerts we *do* raise are trustworthy.
>
> We tracked recall too — we needed to actually catch events. But we optimized toward specificity because: (1) the ops team had other channels for event awareness (news, airline alerts, government advisories), so a missed automated alert wasn't catastrophic; (2) a constant stream of false alarms *was* catastrophic because it would erode trust and make the whole system useless.
>
> 96% specificity — a 4% false positive rate — was the level the ops team could comfortably triage without fatigue. That's about one false alert a day, which is a number a human will actually keep opening."

---

### Q7: "How did you handle false positives?"

> **Answer:** "Three strategies:
>
> First, **multi-signal confirmation**: a single high z-score on one keyword wasn't enough. The Isolation Forest required multiple features to be simultaneously unusual — volume spike *plus* topic shift *plus* keyword surge — reducing single-metric false positives.
>
> Second, **temporal controls**: by including hour-of-day and day-of-week features, the model learned that Monday morning spikes are normal, not anomalous. This eliminated a huge source of false positives.
>
> Third, **feedback loop**: I worked regularly with operational leaders. When they reported false alarms, I analyzed those time windows, identified which features drove the score, and adjusted — sometimes by adding features (like holiday flags) or adjusting the contamination parameter. This continuous improvement was what took specificity from the initial ~80% range up to 96%, and most of that gain came from the last two levers rather than from anything clever in the model."

---

### Q8: "How did topic modeling feed into anomaly detection?"

> **Answer:** "Topic modeling served two purposes:
>
> **Feature generation:** For each time window, I computed the distribution of chat messages across the NMF topics. These topic proportions became features for the Isolation Forest. A normal day might have 30% booking inquiries, 25% refund questions, etc. During a disaster, one or two topics would dominate — say volcanic activity jumping from 2% to 40%. The IF detects this distributional shift.
>
> **Event classification:** Once the IF flagged an anomaly, the dominant topic told us *what* the event was. This was crucial for triggering the right playbook. A volcanic eruption needs different response actions than an airline strike.
>
> So topic modeling was both an input to the anomaly detector and a downstream classifier for the detected anomalies."

---

### Q9: "What was the latency of the real-time system?"

> **Answer:** "End-to-end from chat messages being sent to an alert being raised was in the range of 15–45 minutes. Here's the breakdown:
>
> - Data ingestion: micro-batch pull every 15–30 minutes
> - JSON parsing and feature extraction: ~1–2 minutes
> - Topic model inference and statistical computation: ~1 minute
> - Isolation Forest scoring: seconds
> - Alert generation and routing: near-instant
>
> For disaster detection, this was more than adequate. Disaster events unfold over hours, and the ops team needed lead time measured in 'before the call center is overwhelmed,' not 'sub-second.' A 15–30 minute detection window gave them meaningful time to prepare.
>
> If we needed lower latency in the future, we could move to stream processing with Kafka and Flink, but the operational benefit didn't justify the infrastructure investment at that stage."

---

### Q10: "How did you validate against actual events?"

> **Answer:** "I conducted a retrospective validation over one year of historical data:
>
> 1. **Collected ground truth:** Compiled a list of known disaster events from that period — volcanic eruptions, strikes, weather events, COVID restriction changes — with their dates and durations. This came from news archives, company incident reports, and ops team logs.
>
> 2. **Ran the model:** Processed the full year of chat data through the pipeline and recorded all flagged time windows.
>
> 3. **Matched detections to events:** For each known event, I checked whether the model flagged any time window within ±6 hours of the event (tolerance for early detection or delayed chat response).
>
> 4. **Computed metrics:** Calculated specificity (TN/(TN+FP)), recall (TP/(TP+FN)), and precision (TP/(TP+FP)) against these labels.
>
> 5. **Iterated:** Used grid search to tune hyperparameters, optimizing a weighted combination of specificity (60%) and recall (40%). This yielded the final 96% specificity."

---

### Q11: "What would you improve if you did this again?"

> **Answer:** "Several things:
>
> **BERTopic or sentence embeddings:** NMF on TF-IDF captures keyword-level patterns, but misses semantic nuance. Sentence-BERT embeddings would capture meaning better — e.g., understanding that 'my holiday is ruined' and 'vacation disaster' are related even without shared keywords.
>
> **Temporal modeling:** The Isolation Forest treats each time window independently. An LSTM or temporal convolutional network could model sequences of windows and detect the *ramp-up pattern* of an event, not just the spike.
>
> **Active learning:** Instead of static grid search against historical labels, an active learning loop where the ops team labels alerts as true/false positive in real time would continuously improve the model.
>
> **Streaming architecture:** Moving from micro-batch to true streaming (Kafka + Flink) would reduce latency to seconds, enabling even earlier detection.
>
> **Extended Isolation Forest:** The standard IF uses axis-aligned splits. Extended IF uses hyperplane splits, which would better capture anomalies that only appear in feature interactions."

---

### Q12: "Explain the difference between SMA and EMA and why you used both."

> **Answer:** "SMA gives equal weight to all points in the window — it's a stable, smooth baseline. EMA gives exponentially more weight to recent points — it reacts faster to changes.
>
> I used SMA as the **baseline** for 'normal' behavior: what's the typical chat volume for this time of day? The percentage change and z-score features were computed against this SMA baseline.
>
> I used EMA as a **trend tracker**: when the EMA of a keyword frequency starts rising sharply while the SMA is still flat, that divergence is an early signal of an emerging event.
>
> Together, they gave two complementary views: SMA told us 'what normal looks like,' and EMA told us 'something is changing right now.'"

---

### Q13: "How did you choose the rolling window sizes?"

> **Answer:** "I used three window sizes, each serving a different purpose:
>
> - **12-hour window** (24 periods of 30 minutes): Short-term spike detection. Catches rapid onset events like a sudden volcanic eruption announcement.
>
> - **24-hour window** (48 periods): The primary window. Captures the full daily cycle so z-scores are normalized against a full day of variation. This was the main statistical feature window.
>
> - **7-day window** (336 periods): Long-term trend baseline. Ensures we're comparing against a representative baseline that includes weekday/weekend patterns.
>
> These were informed by the nature of the events: disasters cause chat patterns that deviate over hours to days, so windows needed to be long enough to establish a baseline but short enough to be responsive."

---

### Q14: "What was the contamination parameter and how did you set it?"

> **Answer:** "Contamination in Isolation Forest is the expected proportion of anomalies in the data. It sets the threshold on the anomaly score — a higher contamination means more points are classified as anomalies.
>
> We set it to 0.01 (1%) based on domain knowledge: disaster events are rare. Over a year, maybe 10–15 significant events each lasting a few days, out of thousands of time windows. 1% was a reasonable prior.
>
> This was then refined through grid search against the labeled validation set. We tested 0.005, 0.01, 0.02, and 0.05. 0.01 gave the best balance of specificity and recall. 0.005 was too conservative (missed some events), and 0.05 flagged too many normal periods."

---

### Q15: "How do you explain this system to a non-technical stakeholder?"

> **Answer:** "I'd say: 'We built a system that reads customer chat messages in near-real-time and detects when something unusual is happening. It works in three layers:
>
> First, it understands *what people are talking about* — it groups messages into themes like travel disruptions, weather, strikes, etc.
>
> Second, it compares *how much* people are talking about each theme to what's normal for that time of day and day of week.
>
> Third, it combines all these signals and flags time periods where the pattern is significantly different from normal.
>
> When it detects an anomaly, it tells you what kind of event it thinks it is, and triggers the right response playbook. It's like having a tireless analyst watching every customer conversation and raising a hand when they see something brewing — before the phones start ringing off the hook.'"

---

### Q16: "How did you handle the class imbalance problem?"

> **Answer:** "Class imbalance was fundamental — disaster events are maybe 1% of time windows. I addressed it at multiple levels:
>
> **Algorithm choice:** Isolation Forest is inherently suited for imbalanced data because it's designed to detect the minority (anomalies). It doesn't need balanced classes like a classifier would.
>
> **Contamination parameter:** Set to 0.01, directly reflecting the expected class proportion.
>
> **Metric choice:** Specificity rather than accuracy. With 99% negatives, a model that always predicts 'normal' would have 99% accuracy but be useless. Specificity measures performance specifically on the majority class, and we paired it with recall on the minority class.
>
> **Validation approach:** We validated against known events, not random train/test splits, because random splits would rarely include events in the test set."

---

### Q17: "What's the difference between Isolation Forest's `decision_function` and `predict`?"

> **Answer:** "`decision_function` returns the raw anomaly score — a continuous value where more negative = more anomalous. This is useful for ranking and for setting custom thresholds.
>
> `predict` returns a binary label: -1 for anomaly, 1 for normal. It applies the threshold determined by the `contamination` parameter to the decision function scores.
>
> In practice, I used `decision_function` for monitoring dashboards (showing a continuous anomaly score over time) and `predict` for binary alerting. I also converted the raw scores to a 0–1 scale using MinMaxScaler (inverted) so the ops team could see '85% anomaly probability' rather than raw negative scores."

---

### Q18: "What is topic coherence and why does it matter?"

> **Answer:** "Topic coherence measures how semantically related the top words in a topic are. High coherence means the topic is interpretable — the words make sense together to a human.
>
> I used the c_v coherence metric, which measures word co-occurrence patterns in a sliding window over the corpus. A topic with words like 'volcano, eruption, ash, airspace' has high coherence. A topic with 'volcano, booking, morning, please' has low coherence.
>
> It matters for two reasons: first, as a hyperparameter tuning criterion — I plotted coherence across different numbers of topics to find the optimal $k$. Second, for operational utility — the topics needed to be interpretable by non-technical ops leaders who would use them to understand what kind of event was happening."

---

### Q19: "What if a completely new type of disaster occurred that your model had never seen?"

> **Answer:** "This is exactly why we used an unsupervised approach rather than training a classifier on known disaster types.
>
> The system detects *statistical anomalies* — any significant deviation from normal patterns. It doesn't need to know what a 'volcanic eruption' looks like in advance. If a new type of event (say, a cyberattack on the booking system) caused unusual chat patterns, the volume spike, the keyword convergence, and the topic distribution shift would still trigger the Isolation Forest.
>
> The topic model might not have a neat label for this new event type, but it would show unusual concentration in certain keywords, and the ops team would see those keywords in the alert and understand what's happening.
>
> This was a deliberate design decision — we wanted a system that was robust to novel events, not just a lookup table of known disaster signatures."

---

### Q20: "How did you collaborate with the data engineering team?"

> **Answer:** "The data engineering team owned the LivePerson data pipeline — the scheduled extraction of chat data from the LivePerson API and its storage. My collaboration with them focused on three areas:
>
> **Data access:** They set up the infrastructure so I could pull recent chat data in near-real-time micro-batches rather than waiting for nightly batch exports.
>
> **Schema stability:** LivePerson's JSON schema occasionally changed with API updates. The data engineers helped maintain schema versioning so my parser could adapt.
>
> **Pipeline integration:** Once the model was validated, they helped integrate it into the production pipeline — scheduling the micro-batch pull, running the Python scoring script, and routing alerts to the ops dashboard.
>
> This collaboration was essential. I focused on the ML and analytics; they focused on reliable, timely data delivery."

---

### Q21: "How did you decide on the 30-minute micro-batch window?"

> **Answer:** "It was driven by balancing three factors:
>
> **Detection timeliness:** Shorter windows = faster detection. We needed events flagged within an operationally useful timeframe (< 1 hour).
>
> **Statistical stability:** Each window needed enough chat messages to compute meaningful statistics. A 5-minute window might have too few messages for reliable topic modeling. 30 minutes provided enough volume for stable feature computation.
>
> **Infrastructure cost:** More frequent processing = more compute. 30 minutes was a sweet spot — fast enough for the ops team, stable enough statistically, and manageable computationally.
>
> We experimented with 15-minute and 60-minute windows. 15 minutes was noisier (more false positives due to small sample sizes). 60 minutes was too slow for early detection."

---

### Q22: "What's the role of TF-IDF in this pipeline?"

> **Answer:** "TF-IDF transforms raw chat text into numerical features for topic modeling. Specifically:
>
> **TF (Term Frequency):** How often a word appears in a document. With `sublinear_tf=True`, I applied log scaling so that a word appearing 10 times isn't valued 10x more than one appearance.
>
> **IDF (Inverse Document Frequency):** Downweights words that appear in many documents (common words like 'the', 'flight'). The rarer a word is across the corpus, the more discriminative it is.
>
> TF-IDF was the input to NMF — the document-term matrix that NMF decomposes into topics. I also used the TF-IDF vectorizer's feature names to interpret the topics (which words define each topic).
>
> I set `max_df=0.85` to exclude terms in >85% of documents (too common), `min_df=5` to exclude very rare terms, and `ngram_range=(1,2)` to capture bigrams like 'flight cancel' or 'volcanic ash' which are more informative than individual words."

---

### Q23: "How would you deploy this model in a production environment?"

> **Answer:** "The production deployment would look like:
>
> **Scheduling:** A cron job or Airflow DAG triggers every 30 minutes, pulling the latest batch of chat data from LivePerson's API.
>
> **Processing container:** A Docker container running the Python pipeline — parse JSON, compute features, run topic model inference (pre-fitted NMF), compute statistics, score with pre-fitted Isolation Forest.
>
> **Model serving:** The NMF and Isolation Forest models are serialized (pickle/joblib) and loaded at container startup. They're re-trained periodically (weekly or monthly) on recent data to account for evolving language patterns.
>
> **Alerting:** If `is_anomaly == True`, push alert to ops dashboard and Slack/Teams channel with anomaly score, dominant topic, and top contributing features.
>
> **Monitoring:** Track model drift — if the distribution of anomaly scores shifts over time, or if the ops team reports increasing false positives/negatives, trigger a model retrain.
>
> **Retraining pipeline:** Monthly scheduled retrain: re-fit TF-IDF + NMF on recent 3 months of data, re-fit Isolation Forest on recent data, validate against recent known events, promote if metrics are stable."

---

### Q24: "Can you explain the math behind Isolation Forest's anomaly score?"

> **Answer:** "The anomaly score formula is:
>
> $$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$
>
> Where $E[h(x)]$ is the average path length for point $x$ across all trees. Path length is the number of edges traversed from the root to the terminal node where $x$ is isolated.
>
> $c(n)$ is the average path length of an unsuccessful search in a Binary Search Tree with $n$ nodes, serving as a normalization factor:
>
> $$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$
>
> where $H(i) = \ln(i) + \gamma$ ($\gamma \approx 0.5772$ is Euler-Mascheroni constant).
>
> Intuitively: if $E[h(x)] \ll c(n)$, the exponent is very negative, so $s \to 1$ — strong anomaly. If $E[h(x)] \approx c(n)$, $s \approx 0.5$ — normal. The score elegantly captures how much easier a point is to isolate compared to what you'd expect by chance."

---

### Q25: "What's the difference between NMF and LDA for topic modeling?"

> **Answer:** "The fundamental difference is that LDA is a probabilistic generative model while NMF is a linear algebra decomposition.
>
> LDA posits that documents are mixtures of topics, and topics are mixtures of words, with Dirichlet priors. You learn the posterior distributions via variational inference or Gibbs sampling. It gives you a principled probability distribution over topics for each document.
>
> NMF factorizes the TF-IDF matrix into two non-negative matrices: document-topic and topic-word. It's solving an optimization problem (minimize reconstruction error) without probabilistic semantics.
>
> For this project, NMF was better because: (1) it's faster — important for frequent retraining; (2) it produces crisper, more interpretable topics for short chat messages; (3) the non-probabilistic nature was fine since we didn't need posterior uncertainty estimates; (4) with the right initialization (`nndsvda`), NMF topics were more coherent than LDA on our short-text data."

---

### Q26: "Tell me about a time the model got it wrong."

> **Answer:** "Early on, the model would flag major marketing promotions as disaster events. A flash sale or email campaign would cause a sudden spike in chat volume with concentrated topics around 'deal', 'price', 'booking.' The volume spike and topic concentration patterns looked similar to a disaster event.
>
> This was a false positive driven by volume-only features. I addressed it in three ways:
>
> First, I added temporal features — marketing campaigns typically happened during business hours on specific days, creating a learnable pattern.
>
> Second, I added a marketing calendar feature — a binary flag for known campaign dates. This was a simple but effective domain knowledge injection.
>
> Third, and most importantly, I analyzed the *topic content* of the spikes. Marketing spikes concentrated on positive booking topics, while disaster spikes concentrated on cancellation and disruption topics. The NMF topic features naturally captured this distinction, but it took a few iterations of training data (with these false positives labeled) to push the IF to weight these topic features appropriately.
>
> This experience reinforced the importance of the continuous feedback loop with the ops team."

---

### Q27: "How did you ensure the model stayed accurate over time?"

> **Answer:** "Three mechanisms:
>
> **Concept drift monitoring:** Customer language evolves — new destinations, new policies, new slang. I tracked the NMF reconstruction error over time. If it started rising, it meant the topic model was struggling to represent new language patterns, signaling a need to retrain.
>
> **Periodic retraining:** The TF-IDF vocabulary, NMF topics, and Isolation Forest were retrained on a regular schedule (initially monthly, eventually quarterly as the model stabilized) using recent data as the training set.
>
> **Feedback-driven iteration:** Regular meetings with operational leaders where they reviewed recent alerts (true and false), and I used their feedback to adjust features, window sizes, and thresholds. This wasn't formal online learning, but it was an effective human-in-the-loop improvement cycle."

---

### Q28: "You list four techniques. Why four? Wouldn't one good model do?"

> **Answer:** "Because they answer four different questions, and the failure modes are complementary rather than redundant.
>
> Topic modelling answers *what* people are talking about. Moving averages answer *what is normal for this hour of this day*. Z-scores answer *how unusual this is, in comparable units across features of wildly different scales*. And Isolation Forest answers the question none of the others can — *is this combination unusual even when no single component is extreme?*
>
> The empirical justification is more convincing than the architectural one. When I looked at real events versus false positives in the historical data, real events almost always lit up several signals simultaneously: volume up, entropy collapsing, keywords igniting, sentiment dropping. False positives typically lit up exactly one. So requiring agreement across signals wasn't a design preference, it was the structure of the data.
>
> And to be precise about the composition: it isn't four models voting. Three of them are feature generators and the fourth, the Isolation Forest, is the combiner that consumes all of them."

---

### Q29: "How exactly did you combine the four signals? Voting? Weighted average?"

> **Answer:** "Neither — it's two stages, and I deliberately avoided voting.
>
> Stage one is feature fusion inside the Isolation Forest. Topic weights, topic entropy, SMA and EMA levels and their divergence, rolling z-scores per keyword and per topic, volume, temporal encodings, rate-of-change terms — around twenty-five to thirty numeric features per thirty-minute window, all fed in together, out comes one continuous anomaly score.
>
> I avoided a voting scheme specifically because voting requires you to binarise each signal first, and that throws away information. Three signals each sitting at two sigma is a genuinely stronger indicator than one signal at four sigma, but a three-sigma vote rule scores the first case as zero votes and the second as one. The forest sees the continuous values and can learn that conjunction.
>
> Stage two is a thin rule layer on top of the score: alert if the score clears the threshold, *and* it persists for at least two consecutive windows, *and* a disruption-family topic is among the top contributors. Each of those conditions kills a specific false-positive mode I'd actually observed. Persistence was the highest-leverage one — it costs about thirty minutes of latency on an event that unfolds over hours, and it removes most single-window noise, because real events sustain and blips don't."

---

### Q30: "Why is topic entropy a useful feature? That's not an obvious one."

> **Answer:** "Because it separates the two things that look identical if you only watch volume.
>
> A marketing flash sale and a volcanic eruption both produce a chat volume spike. On volume alone they're indistinguishable, and early on the system flagged promotions as disasters for exactly that reason.
>
> But the *shape* of the conversation differs. During a promotion, volume rises and people ask about everything — deals, baggage, seat selection, payment. The topic distribution stays broad, so entropy stays normal. During a disruption, volume rises and everyone converges on one subject. Topic weight for one topic goes from around two percent to forty, and entropy collapses.
>
> Formally it's just the Shannon entropy of the normalised per-window topic distribution, $H = -\sum_i p_i \log p_i$. A sharp drop means the room has focused. It turned out to be one of the more discriminative single features we had, and it's a good example of why the topic model needed to be *in* the feature set rather than just used for post-hoc labelling."

---

### Q31: "Rolling z-scores on count data — isn't the normality assumption violated?"

> **Answer:** "Yes, and I'd rather raise that myself than have it raised at me. Chat counts in short windows are closer to Poisson or negative-binomial than Gaussian — right-skewed, non-negative, variance tied to the mean. So the clean 'three sigma equals 0.3 percent' interpretation doesn't hold.
>
> Three things follow. First, I treated three sigma as an *empirically tuned threshold* validated against the labelled event history, not as a probability claim. The normal-theory number is where you start, not where you justify.
>
> Second, variance stabilisation — a log-one-plus transform on counts before z-scoring pulls the distribution much closer to symmetric, which makes the sigma units meaningful again.
>
> Third, a denominator guard. In quiet overnight windows the rolling standard deviation approaches zero and the z-score explodes on trivial fluctuation — a keyword going from zero to two occurrences reads as a massive anomaly. I floored the standard deviation and required a minimum observation count before trusting a z-score at all.
>
> The more robust alternative I'd reach for is the modified z-score using median and median absolute deviation instead of mean and standard deviation. That one's genuinely better here, because the classical z-score is contaminated by the outlier it's trying to detect — a big spike inflates the rolling sigma within its own window and partially suppresses its own score. MAD doesn't have that problem."

---

### Q32: "Why is Isolation Forest a good fit for high-dimensional chat features?"

> **Answer:** "Four properties line up with what this feature set looks like.
>
> **No distance metric.** Our features are genuinely heterogeneous — topic proportions bounded in zero to one, z-scores centred at zero, raw counts in the hundreds, binary flags, cyclical time encodings. Any Euclidean distance across that mix is close to meaningless, which rules out LOF and k-NN-based methods on principle rather than on speed. Isolation Forest never computes a distance; it only ever compares one feature to one threshold.
>
> **Linear time.** It's $O(t \cdot n \log n)$. LOF is quadratic and One-Class SVM is worse. We re-scored every thirty minutes, so anything quadratic was out.
>
> **Subsampling helps rather than hurts.** Setting `max_samples` below one actually improves anomaly detection by reducing swamping and masking effects. That's unusual — for most models less data is strictly worse.
>
> **It doesn't need positive examples.** We had maybe ten to fifteen labelled events in a year against tens of thousands of windows. A supervised classifier would memorise those fifteen events. Isolation Forest learns the shape of *normal* and flags departures, which is the right formulation when labels are that scarce.
>
> The weakness I'd name unprompted: axis-aligned splits. An anomaly that's only unusual along a diagonal — normal volume, normal entropy, but an unusual *ratio* between them — is hard for it to isolate. I mitigated that by engineering the interactions explicitly as features, and Extended Isolation Forest with hyperplane splits would be the principled fix."

---

### Q33: "How did you set `contamination`, and how does it relate to your 96%?"

> **Answer:** "Directly — contamination is effectively a specificity dial, and I think that's the most useful way to describe it.
>
> Contamination is the assumed proportion of anomalies in the data, and mechanically it sets the percentile cut-off on the continuous anomaly score that converts it into a binary label. So on data that's roughly ninety-nine percent normal, setting contamination to 0.01 caps the false positive rate near one percent *before* any confirmation logic runs. Set it to 0.05 and your specificity cannot exceed about ninety-five percent no matter how good your features are — you've capped yourself.
>
> I set the initial value from domain knowledge: ten to fifteen significant events a year, each spanning a few days of windows, against tens of thousands of windows total. One percent was a defensible prior. Then grid search over 0.005, 0.01, 0.02 and 0.05 refined it against the labelled event history. 0.005 was too conservative and started missing events; 0.05 flagged far too much.
>
> The subtlety worth adding: contamination doesn't change the *ranking* of windows by anomaly score at all — the forest produces the same scores either way. It only changes where you draw the line. So in production I actually stored the continuous score and applied the threshold downstream, which meant we could retune the operating point without refitting anything."

---

### Q34: "How would you know if your topic model had drifted?"

> **Answer:** "Three signals, in increasing order of how much they should worry me.
>
> First, **reconstruction error**. NMF factorises the TF-IDF matrix as $V \approx WH$; the Frobenius norm of the residual tells you how well the learned topics still explain incoming text. A rising trend means new vocabulary and new themes that the existing topics can't represent — new destinations, new policies, new slang.
>
> Second, **out-of-vocabulary rate on the TF-IDF vectorizer**. If an increasing share of tokens in incoming chat aren't in the fitted vocabulary, the topic model is literally not seeing part of the conversation.
>
> Third, and most concerning, **the OTHER-ish signature**: rising topic entropy overall combined with falling maximum topic weight. That means messages are being spread thinly across topics because none of them fit well. It's the topic-model equivalent of the model shrugging.
>
> The mitigation was scheduled refitting — the TF-IDF vocabulary and NMF topics were refit on recent months of data, monthly at first and quarterly once things stabilised. The operational catch with refitting a topic model is that **topic IDs are not stable across fits**, and our playbook mapping is keyed to topic IDs. So part of the refresh procedure was re-aligning new topics to old ones by comparing their top-term overlap, and flagging any topic that couldn't be matched for human review before promotion."

---

### Q35: "Your ensemble has a lot of moving parts. How do you debug it when an alert is wrong?"

> **Answer:** "By construction, every alert carries its own explanation, and that was a deliberate design requirement rather than an afterthought.
>
> When the gate fires, it records the top contributing features and their values against baseline — so an alert reads something like 'volume z equals 4.2, topic 7 which is the strike topic at weight 0.38 against a 0.02 baseline, entropy down sixty-one percent, sustained across three windows.' The ops team could sanity-check that in about five seconds, and so could I.
>
> For a false positive, that trace immediately localises the problem to a layer. If the volume z-score is extreme but no topic is coherent, it's a volume artefact — usually a platform issue or a campaign, and the fix is a feature like a marketing-calendar flag. If a topic is spiking but volume is flat, it's often a topic-model artefact where a topic has drifted to absorb unrelated vocabulary, and the fix is refitting. If everything is mildly elevated and nothing is decisive, the score threshold is too loose and that's a contamination or threshold question.
>
> The general principle I'd state: with a stacked pipeline, invest in per-layer observability up front, because otherwise a wrong answer is just a wrong answer and you're guessing. The cost of logging the feature attributions was near zero and it's the reason iteration was fast."

---

# Trick & Follow-Up Questions

Adversarial probes. These are designed to find out whether you ran this system or read about it.

---

### Trick Q1: "96% specificity — what's your false negative rate, and isn't missing a real anomaly worse?"

> **Answer:** "Let me take those separately, because the second one is the real question.
>
> On the false negative rate: I optimised the trade-off toward specificity, so recall is meaningfully lower than 96% and I'm not going to invent a number for it. What I can tell you is the shape of what we missed. It wasn't random — misses concentrated on *slow-onset* events, where chat volume ramps gradually over a day or two rather than spiking. A gentle ramp never crosses a three-sigma threshold because the rolling baseline drifts up with it. Fast-onset events, which are the ones that actually overwhelm a contact centre, we caught reliably.
>
> On whether missing one is worse: in general yes, and in *this specific context* no — and the reason is that recall had a safety net and specificity didn't. The ops team had news feeds, airline notifications and government travel advisories. Our system's job was to be *earlier*, not to be the only detector. A miss usually meant finding out thirty minutes later through another channel. A false alarm had no compensating mechanism; it just spent the team's attention.
>
> And I'd flag that this reasoning is contingent, not universal. If this were a safety-critical monitor with no alternative channel, I'd invert the whole thing — optimise recall, accept the noise, and spend the effort on triage tooling instead. The 60/40 weighting in my objective function came from conversations with the ops leads about what they'd tolerate. I wouldn't carry it into a different context without re-deriving it."

---

### Trick Q2: "Specificity is a suspicious metric on imbalanced data. A model that never fires scores 100%. How do I know you didn't just build that?"

> **Answer:** "That's the correct challenge and it's exactly why specificity was never optimised alone.
>
> The grid search objective was a weighted combination — sixty percent specificity, forty percent recall — evaluated against the labelled historical events. If I'd optimised specificity by itself, the search would have driven contamination toward zero and returned a model that flags nothing and scores a perfect 100%. The recall term is what makes the objective non-degenerate.
>
> There's also a structural guard: `contamination` in Isolation Forest forces a fixed proportion of windows to be flagged. At 0.01 it *will* flag about one percent of windows — it cannot silently collapse to flagging nothing. So the never-fires degenerate solution isn't reachable through that parameter.
>
> And the practical evidence is that the system was used. It caught real events — the ops team activated playbooks off the back of it. A detector that never fires produces no playbook activations, and that's a much harder thing to fake than a metric.
>
> If I were presenting this more rigorously I'd show the full specificity-recall curve as contamination varies rather than a single operating point, because that's the honest way to display a tuned trade-off. A single number invites exactly the suspicion you're raising."

---

### Trick Q3: "You had maybe 15 labelled events in a year. How is any of your validation statistically meaningful?"

> **Answer:** "It largely isn't, for recall — and I'd rather say that plainly than dress it up.
>
> With ten to fifteen positive events, the confidence interval on recall is enormous. Catching twelve of fifteen versus ten of fifteen is well inside noise. Any recall figure from that sample should be read as directional, not precise.
>
> Specificity is on much firmer ground, and that's a second, less-discussed reason it was the headline metric. The negative class has tens of thousands of windows, so the false positive rate is estimated from a large sample and its confidence interval is genuinely tight. The metric I quote is the one my data can actually support.
>
> What I did to squeeze more out of the positive side: evaluated at *window* granularity rather than event granularity, which gives more evaluation points since an event spans many windows; used a plus-or-minus six hour matching tolerance so near-misses in timing weren't scored as total failures; and leaned on qualitative review of the misses rather than pretending the aggregate was precise.
>
> The honest summary is that this was a weak-label problem and the validation was closer to careful retrospective sanity-checking than to a powered statistical evaluation. What made me comfortable deploying it anyway is that the alternative was no detection at all, and the ongoing operator feedback loop gave a continuous stream of real-world labels that the one-year retrospective couldn't."

---

### Trick Q4: "Isolation Forest is famously weak in high dimensions. You have 25-30 features. Isn't that a problem?"

> **Answer:** "It's a real limitation and it's worth being precise about *which* limitation, because 'weak in high dimensions' gets used loosely.
>
> The actual mechanism: Isolation Forest picks a random feature at each split. If most of your features are irrelevant to what makes a point anomalous, most splits are wasted, and the anomaly signal gets diluted — the useful dimensions get drowned out. That's the failure mode, and it bites hardest in the hundreds-to-thousands of dimensions.
>
> At twenty-five to thirty features we're not really in that regime, and more importantly the features weren't arbitrary. Every one was purpose-built to carry event signal — z-scores of disaster keywords, disaster-topic weights, entropy, volume derivatives. There isn't a long tail of noise dimensions for the random splits to waste themselves on. That's a meaningful difference from throwing a raw high-dimensional embedding at it.
>
> Two things I did that also help: `max_features=0.75`, so each tree sees a different three-quarters of the feature space, which decorrelates the trees and gives the useful features more chances to be selected. And I kept the feature set deliberately curated rather than letting it sprawl — I dropped candidate features that didn't improve the validation objective, partly for this reason.
>
> If the feature count had grown past fifty or so, I'd have put a dimensionality reduction step in front — PCA, or better, a supervised feature selection pass against the labelled events — or switched to Extended Isolation Forest, which handles correlated feature structure more gracefully."

---

### Trick Q5: "Persistence over two windows adds 30 minutes of latency. For a disaster, isn't that unacceptable?"

> **Answer:** "It'd be a fair objection if the alternative were a working alert thirty minutes earlier. It isn't — the alternative is roughly twice as many false alarms.
>
> The relevant comparison is against the operational response time. When an event is confirmed, the ops team activates a playbook: reassign agents, publish customer comms, brief the front line. That process itself takes well over thirty minutes. So a thirty-minute detection delay doesn't sit on the critical path in the way it would for, say, fraud blocking or an intrusion response.
>
> And the baseline was *hours*. Before this system the team found out when the phones started ringing — often two or three hours into an event. Detecting in forty-five minutes instead of fifteen is a rounding error against that.
>
> I did soften the trade rather than just accepting it. The system had two tiers: a WATCH state that appeared on the dashboard on the *first* window crossing the threshold, with no notification, and an ALERT that fired on confirmation. So an ops lead who happened to be looking at the dashboard saw the first signal immediately; the persistence gate only governed whether we actively interrupted someone. That gets you most of the latency back for the people actively monitoring, without spending the interrupt budget on unconfirmed blips."

---

### Trick Q6: "You used NMF, which is a bag-of-words method. How do you detect an event whose vocabulary you've never seen?"

> **Answer:** "Partially, and I'd be upfront that this is the sharpest genuine weakness in the design.
>
> What still works: the non-topic signals are vocabulary-agnostic. Chat volume spikes regardless of what words are used. Sentiment drops. Handling time blows out. Abandonment rises. So a novel event still trips the volume and behavioural features, and the Isolation Forest would flag the window.
>
> What breaks: the topic-coherence gate in the confirmation layer specifically checks whether a *known disruption-family topic* is driving the score. A genuinely novel event — say a cyberattack on the booking system, using vocabulary that's in none of our fifteen topics — could clear the score threshold and the persistence check and then be filtered out by the coherence gate. That's a real hole and I'd own it.
>
> Two mitigations. First, the coherence gate was not a hard requirement in all configurations — a sufficiently extreme score could bypass it, on the reasoning that something *that* unusual deserves a human look regardless of whether we can name it. Second, an unexpectedly high residual — text that the topic model reconstructs poorly, so it doesn't load onto any topic — is itself a feature and itself an anomaly signal. 'None of my topics explain this conversation' is informative.
>
> The proper fix is semantic embeddings. Sentence-BERT or BERTopic would represent 'the booking system is locked me out' meaningfully even with no keyword overlap with anything in the training corpus. That's the top item on my list of what I'd change, and the reason I didn't do it at the time was compute cost and stakeholder interpretability, not because I thought TF-IDF was better."

---

### Trick Q7: "Your model flagged marketing campaigns as disasters. You fixed it by adding a marketing-calendar flag. Isn't that just hard-coding the answer?"

> **Answer:** "Partly, yes — and the calendar flag was the least interesting of the three fixes, so let me be clear about the ordering.
>
> The calendar flag *is* a hard-coded domain feature. I'd defend it as legitimate — knowing when your own company is running a promotion is real information the model should have access to, in the same way holiday flags are — but I'd concede it doesn't generalise. It only covers campaigns we know about in advance, and it does nothing for a competitor's promotion or an organic viral moment.
>
> The fix that actually generalised was the topic-content distinction. Marketing spikes and disaster spikes both raise volume, but they concentrate on completely different topics — deal, price, availability versus cancel, delay, stranded. Once the topic weights were properly weighted in the feature set, the model separated them on *content* rather than on calendar. That's a learned distinction and it handles campaigns nobody told me about.
>
> The third fix was temporal features. Campaigns cluster in business hours on particular weekdays; the hour-of-day and day-of-week encodings let the model learn that pattern without anyone hard-coding it.
>
> The honest lesson I took from it: my first instinct was the hard-coded flag because it was fast, and it did work for known campaigns. But it was papering over a missing signal. The durable fix was making sure the model could see *what* people were talking about, not just how many of them there were. If I'd stopped at the calendar flag I'd have shipped something brittle that looked fixed."

---

### Trick Q8: "How do you know your 96% wasn't just overfitting to the one year of historical events you tuned against?"

> **Answer:** "I can't fully rule it out, and there's a specific reason to be suspicious: I used the same labelled year both to tune hyperparameters via grid search and to report the metric. That's a genuine methodological weakness and I'd flag it before an interviewer does.
>
> What limits the damage. The tuned surface is small — four hyperparameters over a coarse grid, so there's very little capacity to overfit compared to, say, tuning a neural network. The core model is unsupervised: the Isolation Forest learns the structure of *normal* from tens of thousands of unlabelled windows, and the labels only influence where the operating threshold sits. And specificity is estimated from the large negative class, which is much harder to overfit than recall on fifteen positives.
>
> What I should have done, and would do now: hold out a temporal block. Tune on the first nine months, report on the last three, untouched. That's the walk-forward discipline I applied on the forecasting work and didn't apply rigorously enough here. The excuse is that with ten to fifteen events a year, splitting them leaves too few in either half to tune against — but 'the honest evaluation was inconvenient' isn't really a defence.
>
> The strongest evidence that it wasn't pure overfitting is out-of-sample and operational rather than statistical: the system continued to perform after deployment on months it had never seen, and the false-alarm volume the ops team actually experienced stayed in line with the four percent we'd predicted. Production is the real test set, and it agreed."

---

# 6. Red Flags & How to Handle

---

## 6.1 "Was topic modeling actually used in the model features?"

**Red flag:** Interviewer might suspect topic modeling was separate from the IF model — just used for exploration, not in the actual anomaly detection pipeline.

**How to handle:**

> "Yes, explicitly. The NMF topic model produced a document-topic matrix. For each time window, I computed the average topic weights across all messages in that window. These topic proportion features were direct inputs to the Isolation Forest alongside the statistical features. The topic model wasn't just used for exploratory analysis — it was a core feature engineering step that gave the IF information about *what* people were talking about, not just *how much* they were talking."

---

## 6.2 "Isolation Forest for text data — that's an unusual choice."

**Red flag:** IF is typically associated with tabular numeric data. Interviewer may question using it on text.

**How to handle:**

> "Good observation. The Isolation Forest didn't operate on raw text — it operated on a carefully engineered **numeric feature matrix** derived from text. The pipeline was: text → TF-IDF → NMF (topic proportions) and keyword extraction → statistical features (z-scores, EMAs, % changes) → numeric feature vector per time window → Isolation Forest.
>
> By the time data reached the IF, it was a standard tabular anomaly detection problem with ~25–30 numeric features per time window. The 'text' aspect was fully captured in the feature engineering layers. This is actually a common and effective pattern: use NLP techniques to extract numeric features, then use traditional ML for the downstream task."

---

## 6.3 "Real-time" — what was the actual latency?

**Red flag:** Interviewer may challenge the "real-time" claim if latency was minutes, not seconds.

**How to handle:**

> "Fair question. The system was near-real-time — micro-batch processing with 15–30 minute cycles, giving end-to-end detection within about 15–45 minutes. I wouldn't call it real-time in the streaming sense (sub-second).
>
> But real-time needs to be defined relative to the use case. For disaster event detection, where events unfold over hours and the operational response (mobilizing teams, activating playbooks) itself takes 30+ minutes, detecting an event within 15–45 minutes of its first signal in chat data was operationally real-time. It gave the ops team enough lead time to shift from reactive to proactive.
>
> I'm transparent about this distinction. If we needed sub-second detection (e.g., fraud), we'd need a proper streaming architecture with Kafka and Flink."

---

## 6.4 "96% specificity doesn't sound that high."

**Red flag:** Interviewer may think 96% isn't impressive, or may suspect specificity was picked because it's the flattering metric in an imbalanced problem.

**How to handle:**

> "Two things. First, on the magnitude: 96% specificity means a 4% false positive rate. With 30-minute windows that's about 48 windows a day, so roughly two false alerts daily — a number a human will actually keep opening. At 92% it's four a day, and at 90% it's five, which is where people start muting the channel. The three or four points between those numbers are the difference between a tool that gets used and one that gets ignored, so the improvement is bigger operationally than it looks arithmetically.
>
> Second, on the metric choice — and this is the fairer version of the challenge. Specificity *is* flattering in an imbalanced problem: a model that never fires scores 100%. That's exactly why I never reported it alone. It was always paired with recall against the labelled historical events, and the grid search optimised a weighted combination — 60% specificity, 40% recall — rather than specificity by itself. If I'd optimised specificity alone the search would have driven contamination to zero and produced a system that detects nothing.
>
> And the baseline matters: before this, there was *zero* automated detection. The ops team found out about a volcanic eruption when the phones started ringing."

---

## 6.5 "How much data did you have? Was it enough for Isolation Forest?"

**How to handle:**

> "We had two years of chat history with millions of messages. After aggregating into 30-minute time windows, this gave us tens of thousands of data points, each with ~25–30 engineered features. Isolation Forest actually works well with modest data sizes because it's ensemble-based with subsampling — each tree only uses a fraction of the data. If anything, too much data can slow it down, which is why the `max_samples` parameter exists."

---

## 6.6 "Did you consider supervised approaches?"

**How to handle:**

> "I considered it, but the fundamental challenge was **label scarcity**. Disaster events are rare — maybe 10–15 in a year. That's not enough positive examples for a supervised classifier to learn from reliably. We'd overfit immediately.
>
> The unsupervised approach (topic modeling + Isolation Forest) was essential because it didn't need labeled disasters to learn from. It learned what 'normal' looks like and flagged departures. The limited labels we had were used for validation and hyperparameter tuning, not for training the core model.
>
> If we had accumulated enough labeled events over several years, a semi-supervised approach could be interesting — using the IF as the anomaly detector but training a classifier to categorize detected anomalies."

---

# 7. Key Takeaways

---

## 7.1 Technical Takeaways

| # | Takeaway |
|:--|:---------|
| 1 | **Feature engineering > model complexity.** The statistical features (z-scores, EMAs, % changes) were more important than the choice of anomaly detection algorithm. |
| 2 | **Unsupervised methods shine when labels are scarce.** With only ~15 disaster events per year, supervised learning wasn't viable. Isolation Forest + topic modeling worked without labels. |
| 3 | **The four-signal ensemble beats any single method.** Topic modelling (what), moving averages (what's normal), z-scores (how unusual), Isolation Forest (is the *combination* unusual). Real events light up several signals; false positives usually light up one. |
| 4 | **The confirmation gate did more for specificity than the model did.** Score threshold + persistence over two windows + topic coherence — three cheap rules that each kill an observed false-positive mode. |
| 5 | **Domain-appropriate "real-time" matters more than absolute latency.** 30-minute micro-batches were "real-time" for disaster response. |
| 6 | **Specificity (96%) over recall in alerting systems.** Alert fatigue kills adoption faster than missed alerts — but only because recall had a safety net and attention didn't. |
| 7 | **`contamination` is a specificity dial.** It sets the score percentile that becomes the alert threshold, so it caps your achievable false-positive rate before any feature work matters. |

## 7.2 Process Takeaways

| # | Takeaway |
|:--|:---------|
| 1 | **Continuous feedback loops with end-users** made the difference between a proof-of-concept and a trusted operational tool. |
| 2 | **Retrospective validation** against known historical events was the closest thing to ground truth in an unsupervised setting. |
| 3 | **Collaboration with data engineering** was essential — ML doesn't work without reliable, timely data. |
| 4 | **Translating ML outputs into prescriptive actions** (playbooks) is what made this impactful, not just the detection itself. |

## 7.3 One-Liner Summaries (For Quick Reference)

- **Project in 1 sentence:** "Built an unsupervised anomaly-detection framework over LivePerson chat data that detects emerging operational events at 96% specificity, cutting false-positive alerts and enabling proactive response."
- **Technical in 1 sentence:** "A four-signal ensemble — NMF topic modelling on TF-IDF, SMA/EMA moving averages, rolling z-scores, and an Isolation Forest that combines them all — validated retrospectively against a year of known events."
- **Impact in 1 sentence:** "Transformed Jet2's operations team from reactive firefighting to proactive crisis management, with a false-alarm rate low enough (4%) that the alerts were actually trusted and acted on."

---

*Document prepared for Rahul Sharma — Data Scientist, Jet2 and Jet2 Holidays, Leeds, UK (June 2022 – August 2024)*
*Last updated: August 2026 — aligned to the current resume (96% specificity; four-signal ensemble; reduced false-positive operational alerts).*
