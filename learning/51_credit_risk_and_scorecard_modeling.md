# Credit Risk & Scorecard Modeling — WOE/IV, Binning, KS/Gini, PSI, Validation

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, University of Maryland
**Anchor project:** Credit-risk scorecard at Fibe (EarlySalary) — CIBIL/Experian/transaction/behavioural data, 1,000+ variables screened via WOE/IV + monotonic binning, **0.94 ROC-AUC** out-of-time. (See `projects/P01_fibe_behaviour_scorecard.md`.)
**Document scope:** The complete scorecard discipline — problem framing, target definition, binning, WOE/IV, model fitting, points scaling, the validation battery, reject inference, governance, production monitoring, code, and 30 interview answers.

> **Why this document exists.** Credit risk is the one domain where the *modelling* is the easy part. Anyone can fit a logistic regression. The hard parts are: defining "bad" correctly, proving your features were knowable at decision time, proving the model still works on a future population, and being able to explain a single decline to a regulator two years later. This file is organised around those hard parts.

---

## Table of Contents

1. [The Lending Problem — What a Scorecard Actually Decides](#1-the-lending-problem)
2. [Target Definition — The Highest-Leverage Decision](#2-target-definition)
3. [Data Sources & Point-in-Time Correctness](#3-data-sources--point-in-time-correctness)
4. [Binning & Transformation](#4-binning--transformation)
5. [WOE & Information Value](#5-woe--information-value)
6. [Model Fitting — Logistic Regression on WOE](#6-model-fitting--logistic-regression-on-woe)
7. [Scorecard Scaling — Log-Odds to Points](#7-scorecard-scaling--log-odds-to-points)
8. [The Validation Battery](#8-the-validation-battery)
9. [Reject Inference](#9-reject-inference)
10. [Governance & Regulation](#10-governance--regulation)
11. [Production Concerns](#11-production-concerns)
12. [Python Implementations](#12-python-implementations)
13. [Interview Questions with Strong Answers (30)](#13-interview-questions-with-strong-answers)
14. [Key Takeaways & Cheat Sheet](#14-key-takeaways--cheat-sheet)

---

# 1. The Lending Problem

---

## 1.1 What a scorecard is for

A scorecard is a **ranking function turned into a policy instrument**. It takes an applicant or an existing customer, produces an integer score, and that score is wired into automated decisions: approve, decline, what limit, what price, what collections treatment. The model's job is not to be clever — it is to rank-order risk *stably enough that a policy team can set a cutoff and predict what will happen*.

That framing explains almost every design choice downstream. Stability beats peak accuracy. Monotonicity beats flexibility. Interpretability is a hard requirement, not a nice-to-have, because the score is the legal justification for an adverse action.

## 1.2 The three scorecard families

| Family | Scored population | Data available | Typical target | Typical discrimination |
|--------|-------------------|----------------|----------------|------------------------|
| **Application scorecard** | New applicants ("through-the-door") | Application form + bureau pull + any alternative data. **No internal repayment history.** | 90+ DPD in 12 months from booking | AUC 0.70–0.82, KS 25–40 |
| **Behavioural scorecard** | Existing customers | Everything above **plus** your own repayment, utilisation, transaction and app-behaviour history | 30/60/90+ DPD in the next 3–6 months | AUC 0.85–0.95, KS 45–70 |
| **Collections scorecard** | Already-delinquent accounts | Delinquency history, contact/promise-to-pay outcomes, payment attempts | Cure (roll back to current) vs charge-off | AUC 0.65–0.80 |

> **The single most useful fact in this table:** behavioural scorecards score much higher than application scorecards, and it is not because the modeller is better. It is because knowing how someone has repaid *you* for the last 12 months is enormously more predictive than anything on an application form. When someone challenges a high AUC, the first thing you establish is which family you were in. My Fibe model was a **behavioural** scorecard on rich bureau plus internal repayment data — 0.94 is inside the normal band for that family and would be a screaming red flag for an application scorecard.

There is also a **pre-approval / prescreen** variant (bureau-only, no application) and **early-warning** models on the portfolio side. The methodology in this document is identical across all of them; only the observation point and available data change.

## 1.3 PD, LGD, EAD and Expected Loss

Credit loss decomposes into three separately-modelled quantities:

$$EL = PD \times LGD \times EAD$$

| Component | Meaning | Typical model | Range |
|-----------|---------|---------------|-------|
| **PD** — Probability of Default | Chance the account goes bad in the horizon | Logistic scorecard (this document) | 0–1 |
| **LGD** — Loss Given Default | Fraction of exposure you fail to recover after default | Beta regression / two-stage (cure model + severity), or empirical recovery curves | 0–1 (unsecured personal loans often 0.7–0.9) |
| **EAD** — Exposure at Default | Balance outstanding at the moment of default | For term loans: amortisation schedule. For revolving: current balance + CCF × undrawn limit | Currency amount |

For an unsecured personal loan of ₹50,000 with $PD = 0.08$, $LGD = 0.85$, $EAD = 45{,}000$:

$$EL = 0.08 \times 0.85 \times 45{,}000 = ₹3{,}060$$

You need the loan's revenue (interest + fees over expected life) to exceed $EL$ plus cost of funds plus servicing cost. That inequality is the entire lending business.

**Why PD gets 90% of the modelling attention:** it varies most across applicants and it is the only component you can materially change through the approve/decline decision. LGD and EAD are more product- and process-driven (collections effectiveness, collateral, limit policy) than applicant-driven.

**Where this shows up in interviews:** if someone asks "your model has 0.94 AUC, what's the business impact?", the answer must route through $EL$. A better ranking lets you shift the cutoff, which changes the approve rate and the portfolio $PD$, which changes $EL$ — that is the money.

## 1.4 The decision surface

A score is not just approve/decline. In a mature lender the score feeds:

| Decision | How score is used |
|----------|-------------------|
| **Approve / decline** | Hard cutoff, plus policy rules that can decline independently of score (fraud flags, KYC failures, DPD in the last 3 months) |
| **Manual review band** | A middle band routed to underwriters — the expensive path, sized to underwriting capacity |
| **Credit limit assignment** | Limit is a function of score band × income × existing exposure |
| **Risk-based pricing** | Interest rate tiers by band; must satisfy risk-based-pricing disclosure rules |
| **Tenure / product eligibility** | Longer tenures and larger tickets restricted to top bands |
| **Line management** | Behavioural score drives limit increases and proactive decreases |
| **Collections treatment** | Collections score drives contact intensity and settlement authority |

## 1.5 The cost asymmetry — and why it is not as simple as "false negatives are worse"

The lazy version of this argument is "approving a defaulter costs more than declining a good customer, so optimise recall." That is wrong often enough to be dangerous.

| Error | Immediate cost | Hidden cost |
|-------|----------------|-------------|
| **False negative** (approve a bad) | $LGD \times EAD$ — a large, immediate, visible loss | Collections cost, regulatory attention if systemic |
| **False positive** (decline a good) | Lost lifetime margin on that customer | **Invisible** — you never observe the counterfactual, so it never appears in a P&L line. Also fair-lending exposure, brand damage, and it starves the next model of training data in that region |

The asymmetry ratio depends entirely on the product. For a ₹50,000 loan at 24% APR over 12 months, gross margin might be ~₹6,000 and $EL$ on a bad ~₹38,000 — roughly a 6:1 asymmetry, so the break-even PD is around $1/(1+6) \approx 14\%$. For a low-margin, high-ticket secured product the ratio can be 30:1. For a high-margin card with a small initial limit it can approach 2:1.

The right way to set the cutoff is therefore **not** "maximise KS" — it is to write down the profit function and maximise it:

$$\text{Expected profit per applicant at cutoff } c = \sum_{i: s_i \ge c} \left[ (1-PD_i) \cdot m_i - PD_i \cdot LGD_i \cdot EAD_i \right]$$

where $m_i$ is the expected margin. Maximising KS gives the cutoff that best separates goods from bads; maximising profit gives the cutoff the business actually wants. They coincide only by accident.

> **Say this in an interview:** *"The cost asymmetry is real but it is product-specific, and the false-positive cost is invisible in the P&L, which biases every organisation toward being too conservative. I don't pick the cutoff from KS — I pick it from an expected-profit curve with the margin and LGD assumptions written down explicitly, then show the business the approve-rate/bad-rate trade-off table so they choose with their eyes open."*

---

# 2. Target Definition

---

**This is the highest-leverage decision in the entire project.** Every downstream metric — AUC, KS, IV, the coefficients — is defined relative to the target. Change the bad definition and you change the model, the cutoff, the approval rate, and the loss forecast. Yet it is routinely delegated to a one-line assumption. Get this right and a mediocre model is useful; get it wrong and a brilliant model is worse than useless because it is confidently optimising the wrong thing.

## 2.1 Defining "bad"

The canonical definition is **90+ days past due (DPD) within the performance window**, which aligns with the Basel definition of default (90 days past due *or* unlikeliness to pay). But the right choice varies:

| Bad definition | When to use | Trade-off |
|----------------|-------------|-----------|
| **30+ DPD** | Short-tenure products, behavioural scorecards, when you need a fast-maturing target | More events (easier to model, better statistical power), but many 30-DPD accounts self-cure — you are partly predicting *administrative lateness*, not credit distress |
| **60+ DPD** | Middle ground | Reasonable compromise; commonly used in fintech |
| **90+ DPD** | Regulatory alignment (Basel, IFRS 9 Stage 3), term loans | Fewer events, longer window needed, but it is genuinely "credit bad" |
| **Charge-off / write-off (180+)** | Loss forecasting, LGD work | Very few events, very long window — usually too slow for an origination scorecard |
| **Ever-90 vs current-90** | "Ever" = hit 90+ at any point in the window; "current" = 90+ at window end | "Ever" is standard — it captures accounts that went bad and then cured, which is still a bad customer economically |

Additional definitions that must be folded into "bad" regardless of DPD: bankruptcy, fraud confirmed, settlement/restructure, and death-with-outstanding-balance. Some lenders exclude confirmed first-party fraud from the credit model entirely and route it to a separate fraud model — a defensible choice, because fraud has different drivers, but you must state it.

> **Spoken version:** *"At Fibe the target was 30+ DPD within a 6-month performance window, because it was a behavioural scorecard on short-tenure products and we needed a target that matured fast enough to retrain on. I'd defend that choice, but I'd also say clearly that a 30-DPD target inflates apparent performance relative to a 90-DPD target on the same portfolio, because you're partly predicting people who are just disorganised rather than distressed. If I were building an application scorecard for a 36-month loan I'd use ever-90 over 12 months."*

## 2.2 Observation window, performance window, and the sample frame

```
   observation window            observation point         performance window
 ┌──────────────────────┐              │              ┌────────────────────────┐
 │  features are built  │              │              │  target is measured    │
 │  from data in HERE   │              ▼              │  from behaviour HERE   │
 └──────────────────────┘         ═════╪═════         └────────────────────────┘
        (e.g. 12 months               t=0                  (e.g. 12 months
         of history)                                        after booking)
        NOTHING after t=0 may enter a feature      NOTHING before t=0 may enter the target
```

| Term | Definition | How to choose it |
|------|------------|------------------|
| **Observation point** | The instant the decision would be made. Everything is defined relative to it. | Application timestamp (application scorecard) or a monthly snapshot date (behavioural) |
| **Observation window** | The historical lookback used to build features | Long enough for aggregates to be meaningful (12–24 months typical); constrained by data retention |
| **Performance window** | The forward period over which "bad" is measured | Chosen from **vintage analysis** (§2.3) — long enough that bad rates have stabilised |
| **Sample window** | The calendar range of observation points included in the development sample | Wide enough to cover seasonality and at least one policy regime; excludes anomalous periods (COVID moratoria) or flags them |

The **cardinal rule**: a feature may only use information timestamped strictly before the observation point, and the target may only use information timestamped after it. Any violation is leakage, and leakage in credit is uniquely seductive because the leaked variables (current DPD, current balance, "account status") look like perfectly reasonable features.

## 2.3 Vintage analysis — how you actually pick the performance window

A **vintage** is a cohort of accounts booked in the same month. Vintage analysis plots cumulative bad rate against **months on book (MOB)**, one curve per vintage. Two things fall out:

1. **When does the bad rate flatten?** That is your minimum performance window. Cutting the window shorter censors bads and understates risk.
2. **Are recent vintages worse than older ones at the same MOB?** That is early deterioration — a policy, mix, or macro signal that shows up long before the full window closes.

| MOB | 2024-Q1 vintage | 2024-Q3 vintage | 2025-Q1 vintage |
|-----|-----------------|-----------------|-----------------|
| 3 | 0.9% | 1.1% | 1.6% |
| 6 | 2.4% | 2.9% | 4.0% |
| 9 | 3.6% | 4.2% | — |
| 12 | 4.2% | 4.9% | — |
| 15 | 4.5% | — | — |
| 18 | 4.6% | — | — |

Reading this table: bad rates flatten around **MOB 12–15**, so a 12-month performance window captures roughly 90% of eventual bads — acceptable. And the 2025-Q1 vintage is tracking materially worse at MOB 3 and 6 than either predecessor. That is actionable *now*, nine months before its 12-month number exists. Vintage curves are the earliest honest read on portfolio quality you have.

> **This is also the answer to "how do you know something is wrong before the performance window closes?"** You do not wait. You compare early-MOB delinquency of new vintages against the same MOB of the development vintages.

## 2.4 Roll-rate analysis — how you justify the DPD threshold

A roll-rate matrix shows the probability of moving between delinquency buckets in one month.

| From \ To | Current | 1–29 DPD | 30–59 | 60–89 | 90+ |
|-----------|---------|----------|-------|-------|-----|
| **Current** | 97.8% | 2.0% | 0.2% | — | — |
| **1–29 DPD** | 58% | 24% | 18% | — | — |
| **30–59 DPD** | 21% | 12% | 19% | 48% | — |
| **60–89 DPD** | 6% | 3% | 6% | 12% | **73%** |
| **90+ DPD** | 2% | — | 1% | 2% | **95%** |

The point of interest is where the roll-forward rate becomes near-irreversible. Here, once an account reaches 60–89 DPD it has a 73% chance of rolling to 90+, and 90+ is 95% sticky. Cure rates collapse between 30–59 (21% cure) and 60–89 (6% cure). That is empirical justification for treating 60+ or 90+ as the "point of no return" — and it is a much stronger answer than "90 is the Basel standard."

> **Spoken version:** *"I don't pick the DPD threshold by convention, I pick it from the roll-rate matrix. I look for the bucket where the cure rate collapses and the roll-forward rate approaches one — that's the point where delinquency stops being noise and becomes default. If 30-DPD accounts cure 60% of the time, a 30-DPD target is measuring something quite different from credit loss."*

## 2.5 Indeterminates and exclusions

Not every account is a good or a bad.

| Category | Definition | Treatment |
|----------|------------|-----------|
| **Good** | Never exceeded the bad threshold in the window, and matured through the window | Target = 0 |
| **Bad** | Hit the bad threshold at any point in the window | Target = 1 |
| **Indeterminate** | In the grey zone — e.g. hit 30–59 DPD but never 90+, when the target is 90+ | **Excluded from training**, but scored and monitored. Including them as "good" blurs the contrast and depresses discrimination; including them as "bad" inflates the bad rate |
| **Insufficient performance** | Booked too recently to have matured through the window | Excluded (they are censored, not good) |
| **Non-credit exits** | Voluntary closure, death, confirmed fraud, refunded/cancelled | Excluded, and each exclusion documented with a count |
| **Policy declines** | Never approved, so no outcome | The reject-inference population (§9) |

**Rule of thumb:** if exclusions exceed ~10–15% of the through-the-door population you have a definitional problem, not a data problem. And every exclusion must appear in the model documentation with a count and a reason — an auditor will reconcile the sample waterfall.

## 2.6 Why target definition is the highest-leverage decision

Three reasons, in order of how often they bite:

1. **It sets the ceiling on achievable performance.** A 30-DPD target on a behavioural scorecard is *substantially* easier to predict than a 90-DPD target on an application scorecard. Most of the difference between a "0.94 AUC model" and a "0.75 AUC model" is target and population, not modelling skill.
2. **It determines what the cutoff means economically.** A cutoff calibrated to a 30-DPD PD cannot be plugged into an $EL$ calculation that assumes 90-DPD default without a bridge factor (the 30→90 roll rate). Teams get this wrong and mis-price entire portfolios.
3. **It is nearly impossible to change later.** Once the cutoff, the limit matrix, the pricing tiers and the loss forecast are all calibrated to a target definition, redefining "bad" means re-deriving all of them. In practice organisations live with a bad definition for years.

---

# 3. Data Sources & Point-in-Time Correctness

---

## 3.1 The four sources

| Source | Examples | Predictive strength | Availability / friction |
|--------|----------|--------------------|-------------------------|
| **Credit bureau** | CIBIL, Experian, Equifax, CRIF High Mark (India); Experian/Equifax/TransUnion + FICO (US) | **Strongest single source.** DPD history, trade lines, utilisation, enquiries, account vintages | Costs money per pull; requires consent; monthly refresh lag; ~15–30% thin/no-hit in emerging markets |
| **Application data** | Declared income, employment type, tenure, address, product requested, loan amount | Moderate. Self-reported → gameable and noisy | Free, always available, but verify what you can (income via bank statement/payslip parsing) |
| **Transaction data** | Bank statement inflows/outflows, salary credits, bounce/NSF events, balance volatility, EMI outflows | **Very strong**, especially for thin-file. Bounces and end-of-month balance troughs are excellent stress signals | Requires consent + aggregator (account aggregator framework in India, Plaid/Yodlee in the US); parsing quality is the bottleneck |
| **Behavioural / device** | App session frequency, time-of-day usage, form-fill hesitation, device model, SIM tenure, reminder-response rate | Weak-to-moderate individually, useful in aggregate and for thin-file | Cheap; **highest fair-lending and privacy risk** — device price is a proxy for income and possibly for protected class |

**In my Fibe scorecard** all four were present: CIBIL and Experian bureau records, internal transaction histories, and app-behaviour signals, merged into a customer-level table before the 1,000+ variable screen.

## 3.2 Thin-file and no-hit borrowers

A **thin file** has too few trade lines for the bureau score to be meaningful; a **no-hit** has no bureau record at all. In India this can be 25–40% of a young fintech's applicant base. Three consequences:

1. **You cannot impute a bureau score for them.** Imputing the population median tells the model "this person is average," which is a fabrication about the population you understand least. Give "no-hit" its **own bin** with its own empirically-measured WOE (§4.4). Frequently that bin's bad rate lands between the mid and low bureau bands — informative, and nothing like the median.
2. **You often need a separate segment model.** If the thin-file population is large and the bureau features carry most of the model's weight, a single model effectively has two sub-populations with different feature availability. Two options: (a) build a dedicated thin-file scorecard on transaction + behavioural features, or (b) build one model with explicit missing-indicator bins and validate performance *within* the thin-file segment separately. I prefer (b) first, and split only when segment-level validation shows the pooled model under-performs there.
3. **This is a fair-lending pressure point.** Thin-file correlates with young, immigrant, and lower-income populations. If your model performs materially worse for them, you have both a business problem and a compliance problem.

## 3.3 Alternative data — the honest version

Utility/telecom payments, rent history, e-commerce behaviour, psychometrics, social graph. The pitch is financial inclusion; the reality is mixed:

- **Genuinely useful:** rent and utility payment history (it is literally repayment behaviour), verified cash-flow data from bank statements, and salary-credit regularity. In the US, FCRA-regulated rent/utility furnishing is increasingly mainstream.
- **Legally fraught:** device price, social graph, browsing behaviour, education institution. These are strong proxies for socioeconomic status and, through it, for protected class. In the US, using them invites disparate-impact liability and complicates FCRA adverse-action reasons ("your friends have low scores" is not a lawful principal reason). Several markets have banned some of these outright.
- **The test I apply:** *can I state the causal story for why this predicts repayment, in one sentence, to a regulator, without embarrassment?* "You have missed three utility payments" passes. "You use an Android phone worth less than ₹10,000" does not.

## 3.4 Point-in-time correctness — the leakage section

This is where credit models die. The failure mode is not exotic; it is that **your data warehouse stores current state, not historical state**.

| Leak | How it happens | Fix |
|------|----------------|-----|
| **Current-status fields** | `account_status`, `current_dpd`, `days_since_last_payment` are overwritten in place. Reading them today gives you post-outcome state | Reconstruct from the transaction/event log as-of the observation date, or use a temporal table with valid-from/valid-to |
| **Bureau refresh timing** | You use the January bureau pull for a December application because it is the one in the table | Join bureau on `pull_date <= observation_date`, take the latest such pull, and record the staleness in days |
| **Post-decision fields** | `approved_limit`, `disbursed_amount`, `first_emi_date` only exist for approved accounts — using them makes approval status a feature | Feature audit: for each variable ask "does this exist for a declined applicant at decision time?" If not, drop it |
| **Target-derived aggregates** | `total_amount_written_off`, `max_dpd_ever` computed over all time including the performance window | Recompute all aggregates with a hard filter to the observation window |
| **WOE fitted on the full dataset** | Binning and WOE mapping computed on train+test together | Fit bins and WOE on **train only**; apply the frozen mapping to validation/OOT |
| **Row-level split on a customer-level entity** | The same customer appears in train and test through different monthly snapshots | Split by **customer**, then by **time** — never randomly by row |

**Detection heuristics** (none is conclusive alone, all are worth running):
- Any single feature with $IV > 0.5$ → investigate before celebrating (§5.4).
- A feature whose univariate AUC exceeds the full model's AUC.
- Train AUC ≈ OOT AUC ≈ 0.99. Real models degrade a little out-of-time; a model that does not is usually reading the answer.
- A feature name containing `status`, `current`, `latest`, `final`, `closed`, `written_off`, `recovery`.

> **Spoken version:** *"Leakage in credit almost always comes from the warehouse storing current state instead of as-of state. My standard defence is three things: reconstruct every feature from event logs as of the observation timestamp, run a feature-by-feature audit asking whether that field exists for a declined applicant at decision time, and treat any IV above 0.5 as guilty until proven innocent."*

---

# 4. Binning & Transformation

---

## 4.1 Fine classing and coarse classing

**Fine classing** is the first pass: cut a continuous variable into many small bins (typically 20–50 equal-frequency bins) and compute the bad rate in each. This is diagnostic — you are looking at the *shape* of the risk relationship without imposing anything.

**Coarse classing** is the decision: merge those fine bins into a final set of 4–8 bins that are (a) monotonic in bad rate, (b) each large enough to be stable, and (c) meaningful to a human.

Practical constraints on the final bins:

| Constraint | Typical value | Why |
|-----------|---------------|-----|
| Minimum bin population | ≥ 5% of the sample | Small bins produce unstable WOE that will not replicate out-of-time |
| Minimum bads per bin | ≥ 30 (some shops use 50) | The WOE is a ratio of proportions; with 3 bads it is noise |
| Maximum number of bins | 5–8 | More bins = more parameters implicitly, and a scorecard row per bin that a human has to read |
| Monotonic WOE | Required (see below) | Interpretability, defensibility, robustness |
| Adjacent bins materially different | WOE gap ≥ ~0.1 | If two adjacent bins have near-identical WOE, merge them — the split is not carrying information |

## 4.2 Why monotonic binning is non-negotiable in credit

Monotonic binning means: as the raw variable increases, the bin's WOE moves in one consistent direction. Four independent arguments, and you should be able to give all four:

**1. Interpretability.** The scorecard becomes a sentence a human can finish: "higher utilisation → fewer points → higher risk." A non-monotonic variable produces a scorecard row that says 50–80% utilisation scores worse than 80–100%, which no credit officer will accept and no policy team can use.

**2. Regulatory defensibility.** An adverse-action reason has to survive the question "would this customer have been approved with a *worse* value of this variable?" Under monotonicity the answer is always no. Under non-monotonicity it can be yes, and you have just handed a regulator or a plaintiff's lawyer an arbitrariness argument.

**3. Business sense as a specification test.** Domain knowledge says utilisation, enquiry count and DPD are monotonically bad; account age and income are monotonically good. If your empirical binning disagrees, the most likely explanations are, in order: a small-sample artefact in one bin, a data quality problem (sentinel values mixed into the numeric range), or a genuine mixture of sub-populations. All three are worth finding. **Non-monotonicity is a diagnostic, not just an inconvenience.**

**4. Robustness and stability.** A non-monotonic bin pattern is usually fitting noise in a thin region of the distribution. It is precisely the pattern most likely to invert out-of-time — and an inverted bin does not just lose accuracy, it *reverses* the direction of the points for a slice of the population.

**The honest counter-argument, which you should volunteer:** some relationships are genuinely non-monotonic. Age is the classic — very young and very old borrowers can both be riskier than the middle. Utilisation can be U-shaped when zero-utilisation signals a dormant, disengaged customer rather than a prudent one. The correct handling is not to force monotonicity through the anomaly; it is either to **split the variable** (e.g. a separate zero-utilisation indicator plus monotonic binning of positive utilisation) or to accept a documented non-monotonic bin with an explicit business justification. What you never do is silently force a U-shaped variable into a monotonic straitjacket and lose the signal.

## 4.3 Binning algorithms

| Algorithm | How it works | Pros | Cons |
|-----------|--------------|------|------|
| **Equal-frequency / equal-width** | Quantile or fixed-width cuts | Trivial, good for fine classing | Ignores the target entirely; rarely monotonic |
| **Decision-tree binning** | Fit a shallow `DecisionTreeClassifier` on the single variable against the target; use its thresholds as cut points. Set `monotonic_cst` (sklearn ≥ 1.4) or check-and-merge afterwards | Target-aware, finds high-information cut points, easy to constrain with `min_samples_leaf` | Greedy; needs an explicit monotonicity pass unless constrained |
| **ChiMerge** | Bottom-up: start with many bins, repeatedly merge the adjacent pair with the smallest $\chi^2$ (i.e. the most statistically similar bad rates) until a threshold or bin count is reached | Statistically principled — merges only bins that are not significantly different | Does not directly optimise IV; needs a monotonicity pass |
| **Monotone-adjacent-merge (PAVA-style)** | Start from fine bins; while any adjacent pair violates monotonicity, merge the offending pair. This is pool-adjacent-violators applied to bad rates | Simple, guaranteed monotonic, deterministic | Can over-merge and lose resolution if the fine bins are noisy |
| **Spearman-search** | Start with $n$ equal-frequency bins; if $|\rho_{\text{Spearman}}(\text{bin index}, \text{bad rate})| < 1$, decrement $n$ and retry | Very simple to implement and explain | Coarse; can end at 2–3 bins |
| **Optimal binning (MIP)** | Formulate as a mixed-integer program: maximise IV subject to monotonicity, min-bin-size, max-bins and (optionally) fairness constraints. `optbinning` implements this | Genuinely optimal for the stated objective; supports constraints directly | Slower; opaque unless you understand the formulation; maximising IV can overfit |

**What I use in practice:** decision-tree binning with `min_samples_leaf = 5%` to get candidate cuts, then a monotonic merge pass, then a manual review of any variable that ended with fewer than 3 bins or a strange cut point. `optbinning` when I want the constraint machinery (monotonic + min size + max bins in one call) without hand-rolling it.

$\chi^2$ for the ChiMerge merge criterion on two adjacent bins:

$$\chi^2 = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(A_{ij} - E_{ij})^2}{E_{ij}}, \qquad E_{ij} = \frac{R_i \cdot C_j}{N}$$

where $A_{ij}$ is the observed count of class $j$ in bin $i$, $R_i$ the bin total, $C_j$ the class total, $N$ the total over the two bins. Small $\chi^2$ ⇒ the two bins have statistically indistinguishable bad rates ⇒ merge.

## 4.4 Missing and special values get their own bins

Bureau data is full of sentinel codes: `-1` = no record, `-2` = not applicable, `999` = suppressed, `9999999` = value unavailable. If these flow into numeric binning they will be treated as extreme numeric values and will corrupt the cut points and the monotonicity.

**The rule:** every special value is either its own bin, or explicitly assigned to a bin after inspection of its empirical bad rate. Never impute, and never let a sentinel into the numeric range.

| Value class | Treatment |
|-------------|-----------|
| **Missing (NaN)** | Own bin with its own WOE. The bad rate of "missing" is empirical information |
| **Structural zero** (e.g. utilisation with no revolving line) | Own bin — semantically different from "0% used on an open line" |
| **Bureau sentinel codes** | Map to named categories first, then bin. Own bin each if population allows, otherwise pool the ones with similar bad rates |
| **Small-population bin** (< 5%) | Merge with the neighbouring bin whose WOE is closest — not automatically the adjacent one by value |
| **Unseen category at scoring time** | Must map to a defined default bin. Decide this at build time and freeze it; a `KeyError` in production scoring is an outage |

> **Why "missing gets its own bin" is a fairness point, not just a technical one:** imputing a bureau score for a no-hit applicant fabricates a credit history for the population that has the least of one — disproportionately young, immigrant and underbanked applicants. Giving missing its own empirically-measured bin is both more accurate and more defensible.

---

# 5. WOE & Information Value

---

## 5.1 Definitions

For bin $i$, let $g_i$ = number of goods, $b_i$ = number of bads, $G = \sum_i g_i$, $B = \sum_i b_i$. Define the within-class shares:

$$\%\text{Good}_i = \frac{g_i}{G}, \qquad \%\text{Bad}_i = \frac{b_i}{B}$$

**Weight of Evidence:**

$$WOE_i = \ln\left(\frac{\%\text{Good}_i}{\%\text{Bad}_i}\right)$$

**Information Value:**

$$IV = \sum_i \left(\%\text{Good}_i - \%\text{Bad}_i\right) \times WOE_i$$

Interpretation of the sign convention used here ($\%Good/\%Bad$): **positive WOE = the bin is over-represented among goods = lower risk.** The opposite convention ($\%Bad/\%Good$) is equally common in industry; both are correct, and the only thing that matters is that you state which one you used, because it flips the expected sign of every coefficient. State it unprompted in an interview — it signals you have actually shipped one of these.

## 5.2 Why WOE linearises the logit — the derivation that matters

This is the question that separates people who have used WOE from people who have read about it.

Start from the empirical log-odds of *good* within bin $i$:

$$\ln\left(\frac{g_i}{b_i}\right) = \ln\left(\frac{\%\text{Good}_i \cdot G}{\%\text{Bad}_i \cdot B}\right) = \ln\left(\frac{\%\text{Good}_i}{\%\text{Bad}_i}\right) + \ln\left(\frac{G}{B}\right) = WOE_i + \ln\left(\frac{G}{B}\right)$$

**The bin's empirical log-odds equal its WOE plus a constant** — the population log-odds. So when you replace a raw variable by its WOE, you are replacing it with a quantity that is *exactly, by construction, on the log-odds scale with unit slope*. Logistic regression assumes the log-odds are linear in the inputs; WOE hands it inputs for which that assumption is exactly satisfied univariately.

Three consequences you can state:

1. **A univariate logistic regression of bad on a single WOE variable recovers $\beta = -1$ and $\beta_0 = \ln(B/G)$ exactly.** (Negative because higher WOE = more good = lower probability of bad.) This is a genuine unit test — run it on one variable and confirm.
2. **In a multivariate model, coefficients shrink toward zero** as correlated variables share the signal. Healthy scorecards land with most $\beta_j \in (-1, 0)$.
3. **Any $\beta_j > 0$ is a red flag**, not a finding. It means the multivariate fit reversed the univariate direction — almost always multicollinearity, occasionally a genuine suppressor effect, occasionally a data error. Investigate; do not ship it (§6.2).

**The other three things WOE buys you**, beyond linearisation:
- **Common scale.** Every variable, continuous or categorical, high- or low-cardinality, becomes a single numeric column on the same log-odds scale. No dummy explosion, no scaling step.
- **Outlier immunity.** Binning collapses the tails; a data-entry error of income = 99,999,999 lands in the top bin and receives that bin's WOE. Nothing blows up.
- **Missing handled natively.** A missing bin gets a WOE like any other bin — no imputation, no indicator-column proliferation.

**The cost, which you must acknowledge:** WOE is a **supervised** transformation. It uses the target. That means (a) it must be fit on training data only and applied frozen thereafter, (b) with thin bins it can overfit — this is exactly why the minimum-bads-per-bin rule exists, and (c) the binning itself is a modelling choice that should be re-validated out-of-time, not just the coefficients.

## 5.3 IV is the Jeffreys divergence

Expand the IV sum:

$$IV = \sum_i \%\text{Good}_i \ln\frac{\%\text{Good}_i}{\%\text{Bad}_i} + \sum_i \%\text{Bad}_i \ln\frac{\%\text{Bad}_i}{\%\text{Good}_i} = D_{KL}(\text{Good} \,\|\, \text{Bad}) + D_{KL}(\text{Bad} \,\|\, \text{Good})$$

So **IV is the symmetrised KL divergence (Jeffreys divergence) between the distribution of goods and the distribution of bads across the bins.** Two immediate consequences: $IV \ge 0$ always, and $IV = 0$ if and only if goods and bads are distributed identically across the bins — i.e. the variable carries no information about the target. That is a satisfying thing to be able to say out loud.

## 5.4 IV thresholds — and the one that matters

| IV | Interpretation | Action |
|----|----------------|--------|
| **< 0.02** | Not predictive | Drop |
| **0.02 – 0.1** | Weak | Keep only if it adds incrementally in the multivariate model or covers a segment nothing else covers |
| **0.1 – 0.3** | Medium | Solid candidate |
| **0.3 – 0.5** | Strong | Include |
| **> 0.5** | **Suspicious — check for leakage** | Investigate before celebrating |

The last row is the one interviewers probe. An IV above 0.5 on a single variable means that variable alone nearly separates goods from bads. Genuine causes exist — a `max_dpd_last_6m` variable on a behavioural scorecard legitimately hits 0.6–0.9 because recent delinquency is close to a leading indicator of the target. But the base rate of "IV > 0.5 means leakage" is high enough that the default posture is suspicion.

**The checklist when you see IV > 0.5:**
1. Is the variable a **restatement of the target**? (`max_dpd_ever`, `account_status`, `write_off_flag`)
2. Is it **timestamped after** the observation point? (the warehouse-current-state problem)
3. Is it a **downstream consequence** of the decision rather than an input to it? (`collections_contact_count`)
4. Is the IV **driven by one thin bin**? Look at the per-bin IV contributions — if 90% of the IV comes from a bin holding 1.5% of the population, it is small-sample noise, not signal.
5. Does the IV **hold out-of-time**? A leaked variable usually keeps its IV (the leak is structural); a noisy one collapses. This test discriminates between the two causes.

Two caveats on IV thresholds that show sophistication:
- **IV is univariate.** It says nothing about incremental contribution. A variable with IV 0.45 that correlates 0.95 with your best variable adds nothing. Selection must be IV **plus** correlation/VIF **plus** multivariate significance.
- **IV depends on the binning.** More bins ⇒ mechanically higher IV. Comparing IVs across variables binned with different granularity is comparing nothing. Fix the binning policy first, then compare.

## 5.5 Zero-count bins and Laplace smoothing

If a bin has $b_i = 0$, then $WOE_i = \ln(\text{something}/0) = +\infty$; if $g_i = 0$, $WOE_i = -\infty$. Either destroys the model.

**Preferred fix — prevent it.** Enforce the minimum-bads-per-bin constraint during binning (≥ 30 is a good default). A bin with zero bads should have been merged.

**When you cannot prevent it — smooth.** Additive (Laplace/Jeffreys) smoothing with $k$ bins and pseudocount $\alpha$:

$$WOE_i^{\text{smoothed}} = \ln\left(\frac{(g_i + \alpha)/(G + \alpha k)}{(b_i + \alpha)/(B + \alpha k)}\right)$$

$\alpha = 0.5$ (the Jeffreys / Haldane–Anscombe correction) is the standard choice and is less aggressive than $\alpha = 1$. An alternative used in some shops is to shrink each bin's WOE toward zero in proportion to its sample size (empirical-Bayes shrinkage), which is more principled but harder to explain to a validator.

**What you must not do:** clip the WOE to an arbitrary $\pm 3$ and move on without documenting it. That is a silent modelling decision that changes the points allocation for a real slice of applicants.

## 5.6 Worked example — credit utilisation

15,000 goods, 2,000 bads (11.8% bad rate).

| Bin | Range | Goods | Bads | %Good | %Bad | WOE | IV contribution |
|-----|-------|-------|------|-------|------|-----|-----------------|
| 1 | 0–20% | 5,000 | 200 | 0.3333 | 0.100 | $\ln(0.3333/0.100) = 1.204$ | $(0.3333-0.100)\times1.204 = 0.281$ |
| 2 | 20–50% | 4,500 | 400 | 0.3000 | 0.200 | $\ln(0.300/0.200) = 0.405$ | $(0.300-0.200)\times0.405 = 0.041$ |
| 3 | 50–80% | 3,500 | 600 | 0.2333 | 0.300 | $\ln(0.2333/0.300) = -0.252$ | $(0.2333-0.300)\times(-0.252) = 0.017$ |
| 4 | 80–100% | 2,000 | 800 | 0.1333 | 0.400 | $\ln(0.1333/0.400) = -1.099$ | $(0.1333-0.400)\times(-1.099) = 0.293$ |
| **Total** | | **15,000** | **2,000** | **1.000** | **1.000** | | **IV = 0.632** |

**Reading it:** WOE decreases monotonically (1.204 → 0.405 → −0.252 → −1.099) — exactly the pattern credit domain knowledge predicts and regulators expect. IV = 0.632 is above the 0.5 suspicion line, so it gets the §5.4 checklist: utilisation is a legitimate, causally-sensible driver that is knowable at decision time and not a restatement of the target, the IV is spread across bins 1 and 4 rather than concentrated in one thin bin, and it holds out-of-time. It passes. **But the fact that it passes is a conclusion, not an assumption.**

Note also that the IV is dominated by the extreme bins. That is normal and is why the middle bins can often be merged with little IV loss and a real gain in stability.

## 5.7 WOE pitfalls checklist

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Fitting WOE on train+test | OOT AUC unexpectedly close to train AUC | Fit on train, freeze the mapping, apply everywhere else |
| Too many bins | High IV, unstable OOT, jagged WOE | Enforce min 5% population and ≥30 bads per bin |
| Ignoring bin population when merging | Merged two bins with very different bad rates because they were adjacent by value | Merge by WOE proximity, not value adjacency |
| No plan for unseen categories | `KeyError` at scoring time | Define a default bin at build time |
| Comparing IVs across different binning policies | Nonsense variable ranking | Fix binning policy globally before ranking |
| Silently clipping infinite WOE | Undocumented points shift | Prevent via min-bads; if smoothing, document $\alpha$ |

---

# 6. Model Fitting — Logistic Regression on WOE

---

## 6.1 Why logistic regression is the industry standard

The model is:

$$\ln\left(\frac{P(\text{bad} \mid x)}{1 - P(\text{bad} \mid x)}\right) = \beta_0 + \sum_{j=1}^{k} \beta_j \cdot WOE_j(x_j)$$

Six reasons this remains the standard in regulated lending, in roughly the order a model-risk validator cares about them:

| Reason | Detail |
|--------|--------|
| **Direct scorecard conversion** | Log-odds are linear in the inputs, so they convert to additive points exactly (§7). There is no natural additive-points form for a tree ensemble |
| **Monotonicity is inherited, not imposed** | If each $WOE_j$ is monotonic in the raw variable and $\beta_j$ has consistent sign, the model's response is monotonic *by construction*. In a GBM you must impose it with `monotone_constraints` and then prove you did |
| **Every decision decomposes exactly** | The contribution of each characteristic is $\beta_j \cdot WOE_j$ — an exact decomposition, not an attribution approximation. Adverse-action reasons fall straight out (§10.4) |
| **Stability** | 30–50 parameters fit to hundreds of thousands of rows. Coefficient standard errors are tiny; re-fitting on a new sample moves the points by a point or two, not by a band |
| **Well-understood failure modes** | Validators know exactly how logistic regression fails (multicollinearity, separation, extrapolation). Novel failure modes are expensive in a governance process |
| **Regulatory precedent** | Basel IRB submissions, RBI digital-lending expectations, and every model-validation team in existence have decades of experience with it. Precedent has real economic value in a regulated process |

## 6.2 Coefficient sanity checks — the ones that catch real bugs

With the $\ln(\%Good/\%Bad)$ convention and a model of $P(\text{bad})$:

| Check | Expected | If violated |
|-------|----------|-------------|
| **Sign** | All $\beta_j < 0$ | A positive coefficient means the multivariate fit reversed the univariate direction. Cause is almost always multicollinearity between two near-duplicate variables. Drop one, or combine them |
| **Magnitude** | Most $\beta_j \in (-1, 0)$; univariate would be exactly $-1$ | $|\beta_j| \gg 1$ suggests the variable is compensating for another — check VIF |
| **Intercept** | $\beta_0 \approx \ln(B/G)$ of the training sample | A large deviation means your sample weighting or oversampling is not what you think it is |
| **Stability across folds** | Coefficient sign stable in all folds; magnitude within ~±30% | Sign flips across folds = the variable is not carrying real independent signal |
| **Standard errors** | Small relative to the coefficient | Huge SEs indicate quasi-separation — usually a leaked variable or a bin with almost no bads |

> **Spoken version:** *"With WOE inputs I have a free unit test that most people don't use: a univariate logistic regression on a single WOE variable must return a coefficient of exactly minus one and an intercept equal to the log-odds of the sample. If it doesn't, my WOE calculation is wrong. And in the multivariate model, every coefficient should be negative — a positive one is a multicollinearity alarm, not a discovery."*

## 6.3 Multicollinearity and VIF

Bureau variables are extravagantly correlated: `num_accounts`, `num_open_accounts`, `num_active_accounts` and `num_trade_lines` measure nearly the same thing.

$$VIF_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ comes from regressing $WOE_j$ on all the other WOE variables. Rules of thumb: $VIF < 5$ preferred, $< 10$ tolerable. In credit, teams tend to be stricter ($< 5$) because coefficient *stability* matters more than fit — the points assigned to an attribute must not swing when the model is refreshed.

**Removal order that works:** correlation filter first (drop the lower-IV member of any pair with $|r| > 0.7$), then iterative VIF removal, then check that surviving coefficients are all correctly signed. Doing VIF first on 200 variables is slow and removes the wrong ones.

## 6.4 Variable selection — stepwise vs L1 vs the practical hybrid

| Method | Mechanics | Verdict for credit |
|--------|-----------|--------------------|
| **IV screen** | Keep $IV > 0.02$ | Necessary first filter, never sufficient — it is univariate |
| **Correlation + VIF** | Remove redundancy | Necessary |
| **Forward/backward stepwise** | Add/remove by p-value or AIC | Widely used in credit and widely criticised in statistics: p-values after selection are invalid, and it is unstable under resampling. **Defensible only because credit teams then subject the result to business review and OOT validation** — the stepwise output is a proposal, not a conclusion |
| **L1 / LASSO** | Penalty drives coefficients to zero | Excellent for *selection*; but L1 shrinks the surviving coefficients too, which distorts the points allocation. Standard fix: use L1 to select, then **refit unpenalised (or L2) on the selected set** |
| **L2 / Ridge** | Shrinks all coefficients | Preferred final regularisation — stabilises coefficients without zeroing them |
| **Business review** | Credit policy team reviews every surviving variable | Not optional. It catches proxies, unstable data sources, and variables the business is about to stop collecting |

**My funnel (as run at Fibe):** 1,000+ → drop near-zero variance → drop >70% missing (unless missingness itself is predictive) → WOE bin everything and drop $IV < 0.02$ → correlation filter at $|r| > 0.7$ → VIF < 5 → monotonicity validation → LASSO as a cross-check → business review → ~40–50 final characteristics.

## 6.5 The honest comparison with gradient boosting

**The empirical fact:** on the same data, a well-tuned XGBoost/LightGBM will typically beat a WOE scorecard by **1–3 AUC points** (at Fibe: 0.94 vs ~0.96). It is not nothing, and pretending otherwise in an interview reads as dogma.

**The honest accounting of what that lift costs:**

| Cost | Detail |
|------|--------|
| **Explanation becomes attribution** | The scorecard gives an *exact* decomposition; SHAP gives an *attribution*. For adverse-action reasons, "exact" is legally cleaner |
| **Monotonicity must be imposed and proven** | `monotone_constraints` works, but you now have to demonstrate to a validator that every constraint is correctly signed and actually binding |
| **Governance cost is real, recurring, and expensive** | Independent validation of a GBM takes longer, requires more documentation, and each refresh repeats it. Multiply by every model in the inventory |
| **Instability across refreshes** | Retrain a GBM on three more months of data and individual predictions move materially. Retrain a 40-variable logistic scorecard and points move by 1–2. Policy teams have set limits and prices against those points |
| **Extrapolation behaviour** | Trees are constant outside the training range; a scorecard extrapolates linearly in WOE. Neither is right, but the scorecard's behaviour is at least predictable and reviewable |
| **Overfits the selection region harder** | GBMs exploit fine structure in the accepted population, which is precisely the population that reject inference (§9) tells you is unrepresentative |

**When the GBM lift *does* justify the cost:**
- **Non-decisioning uses**: fraud detection, collections prioritisation, marketing/pre-qualification, early-warning triage. Fewer adverse-action obligations, so the governance cost collapses.
- **Very large portfolios** where 2 AUC points is tens of millions of dollars per year — the governance cost is fixed, the benefit scales.
- **Thin-file / alternative-data segments** where the relationships are genuinely non-linear and interactive and a linear-in-WOE model is leaving real signal behind.
- **As a challenger** running in shadow mode permanently (§11.2) — you get the measurement without the governance burden, and you earn the right to promote it with evidence.

**The bridge pattern I actually like:** use the GBM as a **feature discoverer**, not the production model. Fit it, read the SHAP interaction values, find the two or three interactions it is exploiting, hand-craft those as new characteristics (e.g. a utilisation × recent-enquiry composite), WOE-bin them, and put them in the scorecard. You capture a meaningful share of the lift inside an artefact that is still a scorecard. This is a genuinely strong answer because it refuses the false choice.

> **Spoken version:** *"XGBoost beat my scorecard by about two AUC points, and I'd say that plainly — the interesting question is what those two points cost. In a decisioning model they cost me exact decomposability for adverse action, they cost me a monotonicity proof instead of monotonicity by construction, and they cost me stability across refreshes, which matters because the policy team has set limits against my points. So the scorecard shipped. But I used the GBM as a feature discoverer — read its SHAP interactions, hand-built the two it was exploiting as new characteristics, and recovered part of the lift inside a model I could still put in front of a validator."*

## 6.6 SHAP as a bridge — and where it stops

SHAP gives per-prediction additive attributions $f(x) = \phi_0 + \sum_j \phi_j$, which superficially looks like a scorecard's additive points. Two reasons it is not a full substitute in regulated lending:

1. **Attribution is not decomposition.** A scorecard's $\beta_j \cdot WOE_j$ *is* the model. SHAP's $\phi_j$ is an attribution computed *from* the model, dependent on a background distribution and (for TreeSHAP) on a specific conditional-expectation convention. Two defensible SHAP configurations can rank an applicant's top decline reasons differently. A validator will find that.
2. **Correlated features split credit arbitrarily.** With two near-duplicate bureau variables, Shapley values split the attribution between them in a way that depends on the model's internal split choices. The customer-facing reason code can then flip between two variables that mean the same thing.

SHAP is still the right tool for GBM challenger models, global model understanding, and internal review. It is a bridge, not a replacement. (Deeper treatment in `learning/36_explainability_and_interpretability.md`.)

---

# 7. Scorecard Scaling — Log-Odds to Points

---

## 7.1 The scaling equations

A scorecard is a linear rescaling of the log-odds into a human-friendly integer range. Three parameters define the scale:

| Parameter | Meaning | Typical |
|-----------|---------|---------|
| **Base score** | The score at which the odds equal the base odds | 600 |
| **Base odds** | Good:bad odds at the base score | 30:1 or 50:1 |
| **PDO** | Points to Double the Odds | 20 (sometimes 50) |

$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{odds})$$

$$\text{Factor} = \frac{PDO}{\ln 2}, \qquad \text{Offset} = \text{Base Score} - \text{Factor} \times \ln(\text{Base Odds})$$

Here $\text{odds}$ means **good:bad** odds, so a higher score means lower risk — the near-universal convention.

**Worked scale:** base score 600 at base odds 30:1, PDO 20.

$$\text{Factor} = \frac{20}{0.6931} = 28.854, \qquad \text{Offset} = 600 - 28.854 \times \ln(30) = 600 - 98.15 = 501.85$$

So $\text{Score} = 501.85 + 28.854 \times \ln(\text{odds}_{\text{good:bad}})$. Sanity check: at score 620 the odds should be 60:1. $\ln(\text{odds}) = (620-501.85)/28.854 = 4.095 \Rightarrow e^{4.095} = 60.0$. ✓

## 7.2 From coefficients to per-attribute points

With the model $\ln\!\big(\text{odds}_{\text{bad:good}}\big) = \beta_0 + \sum_j \beta_j WOE_j$, we have $\ln(\text{odds}_{\text{good:bad}}) = -\big(\beta_0 + \sum_j \beta_j WOE_j\big)$. Substituting and distributing the constants evenly across the $k$ characteristics:

$$\text{Points}_{ij} = -\text{Factor} \times \beta_j \times WOE_{ij} \;+\; \frac{\text{Offset} - \text{Factor} \times \beta_0}{k}$$

where $i$ indexes the attribute (bin) and $j$ the characteristic. The second term is the **base allocation** — the constant slice of the intercept and offset assigned to each characteristic so that the points sum to the total score. Since every $\beta_j < 0$, the first term is increasing in WOE: safer bin ⇒ more points. Total score is $\sum_j \text{Points}_{i(j),j}$.

With Factor = 28.854, Offset = 501.85, $\beta_0 = -2.0$ and $k = 5$ characteristics:

$$\text{Base allocation} = \frac{501.85 - 28.854 \times (-2.0)}{5} = \frac{501.85 + 57.71}{5} = 111.91 \text{ points per characteristic}$$

## 7.3 The scorecard table

Five characteristics, coefficients from the fitted model, points rounded to integers.

| Characteristic | $\beta_j$ | Attribute (bin) | WOE | Points |
|----------------|-----------|-----------------|-----|--------|
| **Bureau score** | −0.62 | < 600 | −1.25 | 90 |
| | | 600–679 | −0.48 | 103 |
| | | 680–739 | 0.15 | 115 |
| | | 740–789 | 0.82 | 127 |
| | | 790+ | 1.46 | 138 |
| | | No hit / thin file | −0.35 | 106 |
| **Credit utilisation** | −0.48 | 0–20% | 1.19 | 128 |
| | | 20–50% | 0.41 | 118 |
| | | 50–80% | −0.27 | 108 |
| | | 80–100% | −1.12 | 96 |
| | | No revolving line | 0.05 | 113 |
| **Max DPD, last 12m** | −0.71 | 0 (never late) | 0.71 | 126 |
| | | 1–29 | −0.22 | 107 |
| | | 30–59 | −0.95 | 92 |
| | | 60+ | −1.83 | 74 |
| **Hard enquiries, last 6m** | −0.35 | 0 | 0.58 | 118 |
| | | 1–2 | 0.12 | 113 |
| | | 3–5 | −0.44 | 107 |
| | | 6+ | −1.06 | 101 |
| **Age of oldest trade line** | −0.29 | < 12 months | −0.86 | 105 |
| | | 12–35 months | −0.21 | 110 |
| | | 36–71 months | 0.34 | 115 |
| | | 72+ months | 0.79 | 119 |

**Reading a row:** `Points = 111.91 + 28.854 × |β| × WOE`. For bureau score 740–789: $111.91 + 28.854 \times 0.62 \times 0.82 = 111.91 + 14.67 = 126.6 \to 127$.

## 7.4 Three worked applicants

**Applicant A — middling.** Bureau 705 (115), utilisation 62% (108), max DPD 1–29 (107), 3 enquiries (107), oldest trade 40 months (115).

$$\text{Score} = 115 + 108 + 107 + 107 + 115 = 552$$

Invert to a probability: $\ln(\text{odds}) = (552 - 501.85)/28.854 = 1.738 \Rightarrow \text{odds} = 5.69{:}1 \Rightarrow PD = 1/(1+5.69) = 14.8\%$.

Cross-check directly from the model: $\beta_0 + \sum\beta_j WOE_j = -2.0 + [(-0.62)(0.15) + (-0.48)(-0.27) + (-0.71)(-0.22) + (-0.35)(-0.44) + (-0.29)(0.34)] = -2.0 + 0.248 = -1.752$, so $PD = \sigma(-1.752) = 14.8\%$ and $\text{Score} = 501.85 + 28.854 \times 1.752 = 552.4$. ✓ The points table and the model agree to rounding.

**Applicant B — prime.** 790+ (138), 0–20% utilisation (128), never late (126), 0 enquiries (118), 72+ months (119) ⇒ **629**. $\ln(\text{odds}) = 4.406 \Rightarrow$ odds 82:1 ⇒ $PD = 1.2\%$.

**Applicant C — deep subprime.** <600 (90), 80–100% (96), 60+ DPD (74), 6+ enquiries (101), <12 months (105) ⇒ **466**. $\ln(\text{odds}) = -1.243 \Rightarrow$ odds 0.29:1 ⇒ $PD = 77.6\%$.

**Volunteer this caveat about Applicant C:** a 78% PD is an extrapolation. That combination of attributes is rare in an accepted-population training sample precisely *because* the previous policy declined those people, so the model has very few observations there and is extending a linear-in-WOE relationship into a region it has barely seen. The score is directionally right — decline — but I would not use that PD in a loss forecast. This is the reject-inference problem (§9) surfacing in the scorecard.

**PDO check:** 552 + 20 = 572 ⇒ $\ln(\text{odds}) = 2.431 \Rightarrow$ odds 11.37:1, exactly double 5.69:1. ✓

## 7.5 Risk bands and the policy table

| Score | Band | Approx PD | Policy |
|-------|------|-----------|--------|
| 640+ | A | < 1% | Auto-approve, maximum limit, best pricing |
| 600–639 | B | 1–3% | Auto-approve, standard limit |
| 560–599 | C | 3–8% | Auto-approve, reduced limit, higher rate |
| 520–559 | D | 8–18% | Manual review |
| 480–519 | E | 18–40% | Decline unless compensating factors |
| < 480 | F | > 40% | Auto-decline |

Two things this table must always be paired with: the **approve rate** and the **expected bad rate** at each candidate cutoff, so the business is choosing on the trade-off curve rather than on a number; and the **swap-set analysis** (§8.10) if this scorecard is replacing an existing one.

---

# 8. The Validation Battery

---

Discrimination, stability and calibration are three different properties. A model can be excellent at one and broken at another, and the failure modes are entirely different. Run all three.

## 8.1 ROC-AUC

$$AUC = P(\hat{s}_{\text{bad}} > \hat{s}_{\text{good}})$$

The probability that a randomly chosen bad is ranked riskier than a randomly chosen good. Threshold-free, insensitive to the base rate, and equal to the normalised Mann–Whitney U statistic.

Benchmarks: application scorecards 0.70–0.82; behavioural scorecards 0.85–0.95; anything above 0.95 requires an explanation before it requires congratulations.

## 8.2 Gini

$$\text{Gini} = 2 \times AUC - 1$$

An exact rescaling: 0 = random, 1 = perfect. It is the reporting convention in Europe, India and most of the Basel world; AUC dominates in US tech. They are the same number. Know both, and know they are the same — being asked "what's the difference between Gini and AUC" is partly a test of whether you will invent one.

## 8.3 The KS statistic

$$KS = \max_s \left| F_{\text{bad}}(s) - F_{\text{good}}(s) \right|$$

The maximum vertical gap between the cumulative distributions of bads and goods across the score range. Reported as a percentage (54.4) or a fraction (0.544). Unlike AUC, KS is a **single-point** statistic: it tells you the best achievable separation and *where* in the score distribution it occurs.

### Worked decile table

100,000 accounts, 10,000 bads (10% bad rate), sorted riskiest decile first.

| Decile | Accounts | Bads | Goods | Bad rate | Lift | Cum % Bad | Cum % Good | **Gap** |
|--------|----------|------|-------|----------|------|-----------|------------|---------|
| 1 (riskiest) | 10,000 | 4,200 | 5,800 | 42.0% | 4.20 | 42.00% | 6.44% | 35.56 |
| 2 | 10,000 | 2,300 | 7,700 | 23.0% | 2.30 | 65.00% | 15.00% | 50.00 |
| 3 | 10,000 | 1,400 | 8,600 | 14.0% | 1.40 | 79.00% | 24.56% | **54.44** |
| 4 | 10,000 | 850 | 9,150 | 8.5% | 0.85 | 87.50% | 34.72% | 52.78 |
| 5 | 10,000 | 550 | 9,450 | 5.5% | 0.55 | 93.00% | 45.22% | 47.78 |
| 6 | 10,000 | 320 | 9,680 | 3.2% | 0.32 | 96.20% | 55.98% | 40.22 |
| 7 | 10,000 | 180 | 9,820 | 1.8% | 0.18 | 98.00% | 66.89% | 31.11 |
| 8 | 10,000 | 110 | 9,890 | 1.1% | 0.11 | 99.10% | 77.88% | 21.22 |
| 9 | 10,000 | 60 | 9,940 | 0.6% | 0.06 | 99.70% | 88.92% | 10.78 |
| 10 (safest) | 10,000 | 30 | 9,970 | 0.3% | 0.03 | 100.00% | 100.00% | 0.00 |
| **Total** | **100,000** | **10,000** | **90,000** | **10.0%** | | | | **KS = 54.4** |

**What to read off this table:**
- **KS = 54.4, occurring at decile 3.** Maximum separation is at the 30th percentile of risk.
- **Rank-ordering is clean** — bad rate falls monotonically 42.0 → 0.3% with no inversions. A single inversion between adjacent deciles is a serious finding even if AUC looks fine.
- **Top 3 deciles capture 79% of all bads** while holding 30% of the population. That is the operationally useful number: if manual review capacity is 30% of applications, reviewing the top 3 deciles catches 79% of eventual bads.
- **Decile-trapezoid AUC ≈ 0.844, Gini ≈ 0.688.** (Computed by trapezoidal integration over the ten (cum%good, cum%bad) points. Grouping into deciles slightly *understates* the true continuous AUC — worth saying if you present it.)

## 8.4 When KS, AUC and Gini disagree — and the relationship between them

**They measure different things.** AUC/Gini integrate performance over *all* thresholds; KS reports the single *best* threshold. So two models with identical AUC can have different KS if their ROC curves cross — one concentrating its separation in the middle of the distribution, the other spreading it across the tails.

**There is a provable relationship: for a concave ROC curve, $\text{Gini} \ge KS$.**

*Sketch:* Gini is twice the area between the ROC curve and the diagonal. Consider the minimal concave ROC achieving a given maximum gap: two line segments from $(0,0)$ through the max-gap point to $(1,1)$. That triangle's area above the diagonal is exactly $KS/2$, so its Gini is exactly $KS$. Any concave ROC passing through the same max-gap point encloses *more* area. Hence $\text{Gini} \ge KS$, with equality only for the two-segment case.

**The diagnostic value of the ratio $\text{Gini}/KS$:**
- **Ratio ≈ 1** — all the discriminatory power is concentrated at one cut point; the ROC is nearly two straight segments. The model is essentially a single well-placed threshold. Fine if your cutoff sits there; fragile otherwise.
- **Ratio ≈ 1.2–1.4** — healthy, separation distributed across the score range. The worked example above: $0.688 / 0.544 = 1.26$.
- This also explains the commonly-cited approximation "$KS \approx \text{Gini}$" — it is the *equality case*, so in practice Gini is an **upper bound** on KS. In the Fibe model, AUC 0.94 implies Gini 0.88 and the observed KS was ≈ 0.75, which is exactly the expected direction. If someone reports a KS *above* their Gini, one of the two numbers is computed wrong.

**Which to report:** all three, plus **the metric at your actual operating point**. If your cutoff is at the 10th percentile but KS peaks at the 30th, KS is describing a region you do not operate in. The number that matters is the bad-rate capture and approve rate at the live cutoff.

> **Spoken version:** *"Gini is just two-times-AUC-minus-one so those never disagree — they're the same statistic. KS is different: it's the single best separation point, not an integral. For a concave ROC you can show Gini is always at least KS, and the ratio between them tells you whether the model's power is concentrated at one cut point or spread across the range. Practically, I report all three but I make the decision on the bad-rate capture at the cutoff we actually use, because a KS that peaks at the 30th percentile is irrelevant if we cut at the 10th."*

## 8.5 Out-of-time vs out-of-sample — and why OOT is mandatory

| Split | Construction | What it tests |
|-------|--------------|---------------|
| **In-sample** | Training data | Nothing. A floor |
| **Out-of-sample (OOS)** | Random hold-out from the *same* period | Generalisation to new *individuals* |
| **Out-of-time (OOT)** | A *later* period entirely excluded from training | Generalisation to a new *population and a new environment* |

**Credit data is non-stationary in ways that OOS cannot detect.** Between the training period and deployment: the applicant mix changes (new marketing channels, new geographies), credit policy changes (a cutoff move rewrites who is in the population), the macro environment changes, and the bureau itself changes (new scoring versions, new furnishers, new reporting rules). An OOS split holds all of that constant and will happily report a great number for a model that is about to fail.

**How to build it:** reserve the most recent 3–6 months of observation points, entirely excluded from training, feature selection, *and* binning. The last part is where people cheat — if you fit WOE bins on the full history including the OOT window, the OOT result is contaminated.

**What to expect:** a small degradation (AUC down 0.01–0.02) is normal and healthy. **Zero degradation is suspicious** — it usually means the split was not really out-of-time or something leaked. A large drop (> 0.05) means the model is fitting period-specific structure. Regulators and independent validation will ask for OOT first and everything else second.

## 8.6 PSI and CSI

$$PSI = \sum_i \left(\%\text{Actual}_i - \%\text{Expected}_i\right) \ln\frac{\%\text{Actual}_i}{\%\text{Expected}_i}$$

Same Jeffreys-divergence form as IV, but comparing a *reference* population to a *current* one over score bands (usually the development deciles, frozen at build time).

| PSI | Interpretation | Action |
|-----|----------------|--------|
| **< 0.10** | Stable | Continue monitoring |
| **0.10 – 0.25** | Moderate shift | Investigate; consider recalibration |
| **> 0.25** | Significant shift | Root-cause; likely recalibrate or rebuild |

**CSI (Characteristic Stability Index)** is the identical formula applied to a *single characteristic's* WOE bins. PSI tells you *that* the score distribution moved; CSI tells you *which variable* moved it. Always run both — PSI alone gives you an alarm with no diagnosis.

### Worked PSI example — a real shift

| Band | Expected % (dev) | Actual % (current) | $A - E$ | $\ln(A/E)$ | Contribution |
|------|------------------|--------------------|---------|------------|--------------|
| 1 (lowest score) | 10% | 20% | 0.10 | 0.693 | 0.0693 |
| 2 | 10% | 16% | 0.06 | 0.470 | 0.0282 |
| 3 | 10% | 13% | 0.03 | 0.262 | 0.0079 |
| 4 | 10% | 12% | 0.02 | 0.182 | 0.0036 |
| 5 | 10% | 11% | 0.01 | 0.095 | 0.0010 |
| 6 | 10% | 9% | −0.01 | −0.105 | 0.0011 |
| 7 | 10% | 7% | −0.03 | −0.357 | 0.0107 |
| 8 | 10% | 5% | −0.05 | −0.693 | 0.0347 |
| 9 | 10% | 4% | −0.06 | −0.916 | 0.0550 |
| 10 (highest) | 10% | 3% | −0.07 | −1.204 | 0.0843 |
| **Total** | **100%** | **100%** | | | **PSI = 0.296** |

PSI ≈ 0.30 — well past the 0.25 line. Note the shape: the population has shifted **systematically toward lower scores**, not scattered randomly. That directionality is itself diagnostic — a mix shift or an acquisition-channel change, not a data glitch (a broken feed usually produces a lumpy, non-monotonic PSI profile concentrated in one or two bands). §13 Q10 walks through what to actually do.

## 8.7 Rank-ordering, lift, and KS by vintage

**Rank-ordering** is the most basic requirement and the first thing to break: bad rate must decrease monotonically across score bands. Check it on OOT and within each vintage. **An inversion between adjacent bands is more serious than a 0.02 AUC drop**, because it means the score is actively wrong for a slice of the population and the limit/pricing matrix built on those bands is mispriced.

**Lift** = band bad rate / overall bad rate. Decile 1 lift of 4.2× in the §8.3 table means the riskiest decile defaults at 4.2 times the portfolio average.

**KS by vintage** is the drift early-warning system. Compute KS separately for each monthly booking cohort as it matures and plot the trend:

| Booking month | Accounts | KS at MOB 6 |
|---------------|----------|-------------|
| 2025-01 | 12,400 | 52.1 |
| 2025-02 | 13,100 | 51.4 |
| 2025-03 | 12,900 | 50.8 |
| 2025-04 | 14,600 | 47.2 |
| 2025-05 | 15,200 | 44.9 |

A steady erosion like this — rather than a single bad month — is the signature of genuine model decay, and it is visible at MOB 6 long before the 12-month numbers exist. A single anomalous month is usually a data or policy event; a trend is the model.

## 8.8 Risk-segment analysis

Pooled metrics hide segment failures. Report the full battery **within each segment**, not just overall:

| Segment | N | Bad rate | AUC | KS | Rank-order clean? |
|---------|---|----------|-----|-----|-------------------|
| Thin file / no hit | 18,400 | 14.2% | 0.79 | 41.2 | Yes |
| Thick file | 81,600 | 8.9% | 0.87 | 56.8 | Yes |
| Salaried | 72,000 | 8.1% | 0.88 | 57.9 | Yes |
| Self-employed | 28,000 | 14.8% | 0.81 | 44.1 | Deciles 6/7 inverted |
| Metro | 61,000 | 9.2% | 0.86 | 54.2 | Yes |
| Non-metro | 39,000 | 11.4% | 0.83 | 48.6 | Yes |
| New-to-bank | 34,000 | 13.1% | 0.80 | 42.7 | Yes |

The self-employed row is the finding. The pooled AUC looks fine; that segment has weaker discrimination *and* a rank-order inversion. Options: a segment-specific scorecard, segment-specific cutoffs, or additional characteristics that work for self-employed applicants (bank-statement cash-flow features rather than salary-credit regularity). Segments to always cut by: file thickness, income type, tenure with the lender, channel, geography, and product.

**This is also where fair-lending analysis lives** — the same table cut by proxied demographic groups, which is a legally distinct exercise (§10.5) but methodologically identical.

## 8.9 Calibration — discrimination's forgotten sibling

Discrimination asks "does the model rank correctly?" **Calibration asks "are the predicted probabilities right?"** You need calibration for pricing, limit-setting, IFRS 9/CECL expected-loss provisioning, and any $EL$ calculation. A model can rank perfectly and be systematically off by 2× on the level.

**Calibration plot.** Bin by predicted PD, plot mean predicted vs observed bad rate, compare to the 45° line. This is the primary tool — the tests below are summaries of it.

**Calibration slope and intercept (weak calibration).** Regress the observed outcome on the predicted logit: $\text{logit}(y) \sim a + b \cdot \text{logit}(\hat p)$. Perfect calibration is $a = 0$, $b = 1$. $b < 1$ means predictions are too extreme (overfit); $a \ne 0$ means a systematic level shift — which is exactly what a changing base rate produces, and it is fixable by adjusting the intercept alone without touching the coefficients.

**Hosmer–Lemeshow test.**

$$\hat{C} = \sum_{g=1}^{G} \frac{(O_g - E_g)^2}{E_g\left(1 - \frac{E_g}{n_g}\right)} \sim \chi^2_{G-2}$$

Group into $G$ (usually 10) bins by predicted probability; compare observed bads $O_g$ to expected $E_g$. **Volunteer the criticisms**: the result depends on the arbitrary choice of $G$ and on the grouping method (implementations disagree), and with large $N$ it rejects for trivially small, practically irrelevant miscalibration. On a 500,000-row credit sample HL will essentially always reject. Report it because validators expect it; **decide** on the calibration plot and the slope/intercept.

**Brier score.**

$$BS = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2$$

A proper scoring rule combining discrimination and calibration. Murphy's decomposition is $BS = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$, where $\text{Uncertainty} = \bar{y}(1-\bar{y})$ depends only on the base rate. **So Brier scores are not comparable across portfolios with different bad rates** — a 2%-bad-rate book will always show a better Brier than a 15% book regardless of model quality. Use the Brier Skill Score, $1 - BS/BS_{\text{ref}}$, against the base-rate-only model.

**Recalibration methods**, in increasing order of intervention: intercept-only shift (base-rate correction — the right move when rank-ordering holds but the level moved), Platt scaling (fit $a$ and $b$ on the logit), isotonic regression (non-parametric, flexible, needs a lot of data and can produce step functions that are awkward in a scorecard), and finally a full rebuild.

## 8.10 Swap-set analysis — the only honest way to compare two models

When replacing a champion, AUC comparison is not sufficient. What the business needs to know is **who changes decision**, at matched approval rates.

Hold the approve rate constant. Then every applicant falls into one of four cells:

| | New model: approve | New model: decline |
|---|---|---|
| **Old: approve** | Both approve | **Swap-out** |
| **Old: decline** | **Swap-in** | Both decline |

The whole argument lives in the two swap sets:

- **Swap-in** — declined by the old model, approved by the new. Their *predicted* bad rate should be materially below the portfolio cutoff bad rate. You have no observed outcomes for them (they were declined), so this is a model-vs-model claim and must be flagged as such.
- **Swap-out** — approved by the old model, declined by the new. **Here you do have outcomes**, because they were booked. If the swap-out set's *actual* observed bad rate is much higher than the swap-in set's predicted bad rate, the new model is genuinely better and you can prove it with data.

| Cell | Volume | Old model's view | Observed bad rate | Verdict |
|------|--------|------------------|-------------------|---------|
| Both approve | 58,200 | approve | 6.1% | — |
| Swap-out | 4,800 | approve | **19.4% (observed)** | New model correctly rejects these |
| Swap-in | 4,800 | decline | 7.2% (predicted, unobservable) | Claimed improvement |
| Both decline | 32,200 | decline | not observed | — |

Also report the **profile** of each swap set: if swap-ins are concentrated in one geography or one thin-file segment, that is a fair-lending question and a concentration-risk question, independent of whether the average bad rate is fine.

> **Spoken version:** *"AUC tells me the new model ranks better on average; the swap set tells me what actually changes. At a matched approve rate I look at who gets swapped out — and those I have real outcomes for, because we booked them. If the swap-out set defaulted at 19% and the swap-ins are predicted at 7%, that's a real, evidenced improvement rather than a metric improvement. And I always profile the swap sets by segment, because a swap-in population concentrated in one geography is a fair-lending conversation even if the average bad rate is fine."*

## 8.11 The validation report checklist

| # | Item | Pass criterion |
|---|------|----------------|
| 1 | Sample waterfall with every exclusion counted | Reconciles to the source population |
| 2 | Target definition + vintage curve justifying the window | Bad rate flat by window end |
| 3 | Feature audit for point-in-time availability | Every feature confirmed available at decision time |
| 4 | Per-variable bin table, WOE, IV, monotonicity flag | All monotonic or documented exception |
| 5 | Correlation matrix and VIF | VIF < 5 |
| 6 | Coefficient table with signs, SEs, p-values | All signs correct |
| 7 | AUC / Gini / KS on train, OOS, OOT | OOT degradation < 0.02–0.05 |
| 8 | Decile / gains table on OOT | Monotonic rank-ordering |
| 9 | Calibration plot + slope/intercept + Brier | Slope near 1, plot on diagonal |
| 10 | PSI of score, CSI of every characteristic | PSI < 0.10 at build |
| 11 | Segment-level metrics | No segment with broken rank-ordering |
| 12 | Fair-lending / disparate-impact analysis | AIR above threshold, or documented LDA search |
| 13 | Swap-set analysis vs champion | Swap-out observed bad rate > swap-in predicted |
| 14 | Reject-inference approach and sensitivity | Results shown with and without RI |
| 15 | Known limitations, stated explicitly | Written down before someone else finds them |

---

# 9. Reject Inference

---

## 9.1 The problem, stated precisely

You only observe repayment for applicants you **approved**. But the model must score everyone who applies. Let $A = 1$ denote approval and $Y$ the bad outcome. Training on booked accounts estimates:

$$P(Y = 1 \mid X, A = 1)$$

but the decision requires:

$$P(Y = 1 \mid X)$$

over the full through-the-door population. **Whether these differ is the entire question, and the answer is more subtle than "yes."**

**Case 1 — approval was a deterministic function of the observed $X$.** If the previous policy declined everyone with $X$ in some region and approved everyone else, then conditional on $X$, approval carries no extra information about $Y$: $P(Y \mid X, A=1) = P(Y \mid X)$. **There is no bias.** What you have instead is **range restriction** — no data at all in the declined region, so the model must extrapolate. That is a variance and extrapolation problem, not a bias problem, and reject inference does not fix it because no method can create information that was never observed.

**Case 2 — approval used information not in $X$.** Underwriter judgment, a document review, a previous model with variables you no longer have, a fraud flag. Now approval depends on unobserved $U$ that also predicts $Y$, so:

$$P(Y = 1 \mid X, A = 1) \ne P(Y = 1 \mid X)$$

Specifically, among approvals at a given $X$, the unobserved $U$ is favourable — so the observed bad rate understates the true bad rate for that $X$. **This is genuine selection bias (MNAR), and it is the case reject inference is trying to address.**

Being able to draw this distinction is the single strongest thing you can say about reject inference, because it explains *why* the methods below are unsatisfying: they are attempting to correct for something unobserved using only observed data.

## 9.2 The methods

| Method | Mechanics | Assumption it needs |
|--------|-----------|--------------------|
| **Parcelling** | Score rejects with the known-good/known-bad (KGB) model. Bucket by score band. Within each band, randomly assign good/bad in proportion to the accepted population's bad rate in that band — usually **inflated** by a judgemental factor (2–5×) to reflect "rejects are worse than accepts at the same score" | That the inflation factor is right. It is a guess, and the whole answer is sensitive to it |
| **Augmentation (reweighting)** | Fit an accept/reject model $P(A=1 \mid X)$; weight each accepted record by $1/P(A=1 \mid X)$ so the accepted sample is reweighted to resemble the through-the-door population | **MAR given $X$** — i.e. Case 1. But under Case 1 there was no bias to correct. Under Case 2, IPW does not fix the problem it is invoked for |
| **Fuzzy augmentation** | Each reject enters the training set **twice**: once labelled bad with weight $\hat{p}$, once labelled good with weight $1 - \hat{p}$, where $\hat p$ is the KGB-model PD (optionally adjusted upward) | Same as parcelling, but without the extra Monte-Carlo noise. Strictly better than parcelling on variance grounds |
| **Bivariate probit / Heckman two-step** | Jointly model a selection equation (approve/decline) and an outcome equation (default) with correlated normal errors, correlation $\rho$. Estimating $\rho$ is exactly estimating the selection bias | Needs an **exclusion restriction**: a variable affecting approval but not default. In credit, essentially every approval variable also predicts default, so no valid instrument exists. Identification then comes entirely from the assumed bivariate normality — a functional-form assumption doing the work of an instrument |
| **Randomised approval / test-and-learn** | Approve a small random sample of applicants who would have been declined; observe their real outcomes | **The only method that produces genuinely new information.** Costs real money in known-bad losses |

## 9.3 The honest limitations

**Every method except randomised approval is circular.** Parcelling and fuzzy augmentation label rejects using a model fit on accepts, then feed those labels back to train a model on the combined sample. The reject labels contain no information that was not already in the accept-based model — they *propagate* its assumptions into the reject region rather than testing them. If the KGB model is wrong about the reject region, reject inference makes it wrong with more confidence and a larger apparent sample size.

**It cannot be validated where it claims to help.** You can measure the reject-inferred model's performance on the accepted population; you cannot measure it on the rejects, because you have no outcomes there. So the claim "reject inference improved my model" is untestable in exactly the region the method was invoked for. The apparent AUC change is measured on accepts, where reject inference typically *lowers* it slightly.

**Heckman's exclusion restriction essentially never exists in credit.** Anything that influenced the approval decision is, by construction, something the lender believed predicts default. Without an instrument, the model is identified only by the bivariate-normal assumption, and results are known to be sensitive to it.

**So why do it at all?** Two legitimate reasons: (1) **Regulatory expectation** — validators expect the through-the-door population to be considered, and documenting a reject-inference approach with a sensitivity analysis is a compliance requirement. (2) **Extrapolation stability** — including reject records, even with soft labels, stabilises the model's behaviour in the low-score region and reduces wild extrapolation (recall Applicant C in §7.4). It is a regularisation-shaped benefit, not an information-shaped one.

**What I would actually push for:** a small, deliberately-budgeted randomised approval programme — approve a random 1–2% of marginal declines just above the auto-decline floor, accept the known loss as a **data acquisition cost**, and treat it as the only unbiased source of information about the reject region. It also generates the counterfactual data needed to measure the false-positive cost that never shows up in the P&L (§1.5).

> **Spoken version:** *"Reject inference is trying to estimate P(bad | X) over the through-the-door population when you only observe outcomes for approvals. The first thing I'd separate is whether the old policy was a deterministic function of the variables I have — if it was, there's no bias, just range restriction, and no amount of inference creates data. The genuine bias case is when underwriters used judgement I can't observe. And that's exactly the case none of the standard methods fix, because parcelling and fuzzy augmentation label rejects using a model fit on accepts, so they propagate the assumption rather than test it. Heckman needs an exclusion restriction that doesn't exist in credit. I do reject inference because validators expect it and because it stabilises extrapolation in the low-score region — but I present results with and without it, and the only thing that genuinely fixes it is buying the data with a small randomised approval programme."*

---

# 10. Governance & Regulation

---

## 10.1 SR 11-7 and model risk management

**SR 11-7** (Federal Reserve / OCC supervisory guidance on Model Risk Management, 2011) is the reference framework in US banking, and its concepts have been widely adopted elsewhere. Three pillars:

| Pillar | Content |
|--------|---------|
| **1. Model development, implementation and use** | Documented conceptual soundness; the model's design rationale, data, assumptions and limitations written down; testing of the implementation, not just the maths |
| **2. Model validation** | Independent review with three components: (a) evaluation of **conceptual soundness**, (b) **ongoing monitoring** including process verification and benchmarking, (c) **outcomes analysis** — back-testing predictions against realised outcomes |
| **3. Governance, policies and controls** | A model inventory, clear roles and ownership, model risk appetite, internal audit oversight, and approval workflows |

The load-bearing phrase is **"effective challenge"**: review by people with the competence, incentive and *authority* to push back. In practice this is the three-lines-of-defence structure — the model developer (1st line), independent model validation (2nd line), internal audit (3rd line).

**What this means concretely for a scorecard:** an independent validator will re-derive your results, challenge your bad definition, re-run the OOT split, ask why every excluded variable was excluded, and demand the limitations section. Building the documentation *as you go* is the difference between a two-week and a two-month validation cycle.

## 10.2 Documentation — the Model Development Document

At minimum: business purpose and intended use (and explicitly, **prohibited** uses); data sources, extraction logic, and the sample waterfall; target definition with vintage and roll-rate justification; the full variable list including rejected variables and why; binning tables for every characteristic; coefficient table with diagnostics; the full validation battery; fair-lending analysis; implementation specification (the exact scoring formula, tie-breaking, and default-bin handling); monitoring plan with thresholds and owners; and a limitations section written by the developer.

**The limitations section is the one that earns credibility.** Write down what the model cannot do — that it was trained in a benign credit cycle, that it extrapolates poorly below score 480, that the thin-file segment has weaker discrimination — before a validator finds it. A model developer who volunteers limitations is trusted on everything else.

## 10.3 Challenger models

Maintain a **champion** (in production) and one or more **challengers** (scored in parallel, no decisions). Challengers serve three purposes: they benchmark whether the champion has decayed, they are the pre-validated replacement when it has, and they are the SR 11-7 "benchmarking" element of ongoing monitoring.

In credit the standard pair is a logistic scorecard champion and a GBM challenger. Running the GBM in shadow gives you a continuous read on the interpretability/accuracy gap — and if that gap widens over time, it is evidence that the champion's linear-in-WOE structure is missing newly-emerged non-linearity, which is a genuine rebuild trigger.

## 10.4 Adverse action and the explainability duty

Under **ECOA / Regulation B** in the US, an applicant denied credit must receive a notice with the **specific principal reasons** for the denial (customarily up to four). Under **FCRA**, if a consumer report was used, additional disclosures and the key factors affecting the score are required. India's RBI fair-practices and digital-lending framework imposes an analogous duty to communicate the reasons for rejection.

**The scorecard makes this mechanical.** For each characteristic, compute the applicant's points shortfall against a reference — usually the **maximum attainable points** for that characteristic, or the neutral/population-average points:

$$\text{Shortfall}_j = \text{Points}_j^{\max} - \text{Points}_j^{\text{applicant}}$$

Rank the shortfalls, take the top 3–4, map each to a pre-approved plain-language reason code. Applicant A from §7.4:

| Characteristic | Points | Max points | Shortfall | Reason code |
|----------------|--------|------------|-----------|-------------|
| Max DPD last 12m | 107 | 126 | **19** | "Recent delinquency on one or more accounts" |
| Credit utilisation | 108 | 128 | **20** | "Proportion of balances to credit limits is too high" |
| Bureau score | 115 | 138 | **23** | "Credit bureau score below our requirement" |
| Hard enquiries | 107 | 118 | 11 | — |
| Age of oldest trade | 115 | 119 | 4 | — |

Top three reasons: bureau score, utilisation, recent delinquency. **This is exact** — it is arithmetic on the model itself, not a post-hoc attribution. That exactness is a large part of why scorecards persist in decisioning.

**Two practical requirements:** the reason codes must be pre-approved plain language (not variable names), and they must be **actionable** — telling someone "your utilisation is too high" tells them what to do; telling them "your device tenure segment is unfavourable" does not, which is one more argument against exotic behavioural features in a decisioning model.

## 10.5 Fair lending and disparate impact

Two legally distinct concepts:

| Concept | Definition | Defence |
|---------|------------|---------|
| **Disparate treatment** | Using a protected characteristic (or an explicit proxy) in the decision. Intentional | None. Do not do it |
| **Disparate impact** | A facially neutral policy that produces significantly worse outcomes for a protected group | A three-step burden-shifting framework: (1) plaintiff shows the disparity, (2) lender shows **business necessity**, (3) plaintiff may show a **less discriminatory alternative (LDA)** achieving the same business purpose |

**Protected bases under ECOA:** race, colour, religion, national origin, sex, marital status, age (with narrow exceptions), receipt of public assistance income, and exercise of rights under consumer credit protection law. India's RBI framework prohibits discrimination on religion, caste, gender and similar grounds.

**Measurement.** In US mortgage lending, demographic data is collected under HMDA. In most other lending it is not, so demographics are **proxied** — the standard is BISG (Bayesian Improved Surname Geocoding), combining surname and census-tract distributions. Then:

$$\text{Adverse Impact Ratio (AIR)} = \frac{\text{approval rate of protected group}}{\text{approval rate of reference group}}$$

The **four-fifths rule** (AIR < 0.80) is a screening heuristic imported from employment law, not a legal standard in lending — say that, because a lot of candidates state it as if it were binding. Also report the **standardised mean difference** in scores, and the disparity **at the actual cutoff** rather than in the aggregate, since where you cut determines the disparity.

**The LDA search is an affirmative obligation, not a defence you wait to need.** The standard technique: iteratively drop or re-specify the characteristics contributing most to the disparity, and measure the AUC cost of each. If dropping a variable reduces the disparity by 6 points at a cost of 0.003 AUC, you have found a less discriminatory alternative and you are expected to adopt it. Document the search either way — searching and finding nothing is a strong position; not searching is not.

## 10.6 Proxies for protected attributes

Removing protected attributes is necessary and nowhere near sufficient. Proxies are everywhere: ZIP code (strongly correlated with race in the US due to historical residential segregation), first/last name, university attended, employer, language preference, device price and model, and any geographic aggregate.

**How to detect a proxy:** regress the (proxied) protected attribute on each candidate feature and look at the achieved $R^2$ or AUC. A feature that predicts group membership at AUC 0.75 is a proxy regardless of what it is named. Then ask whether it has independent predictive power for default *after* controlling for the legitimate variables it is standing in for — usually a "geographic risk" feature is largely a stand-in for income and employment stability, which you can and should measure directly.

**ZIP code specifically** should be treated as presumptively unusable in US consumer lending decisioning. If geography matters, use it through an explicitly-justified economic variable (regional unemployment, cost-of-living index) rather than the raw geographic identifier — and expect to defend even that.

## 10.7 Monitoring cadence

| Frequency | What | Trigger threshold |
|-----------|------|-------------------|
| **Daily** | Scoring volume, null rates, score-distribution mean/SD, error rates | Any hard failure of a data-integrity gate halts scoring |
| **Monthly** | PSI of the score, CSI of every characteristic, approve rate by band, override rate, early-MOB delinquency by vintage | PSI > 0.10 investigate; > 0.25 escalate |
| **Quarterly** | Discrimination (AUC/KS/Gini) on matured accounts, rank-ordering by band, calibration (actual vs expected bad rate), segment-level metrics, champion vs challenger | AUC drop > 10% relative, any rank-order inversion, actual/expected outside [0.8, 1.2] |
| **Annually** | Full revalidation by independent validation; fair-lending re-run; documentation refresh | Formal re-approval by the model risk committee |
| **Event-driven** | Policy change, new acquisition channel, data-source change, bureau score version change, macro shock | Immediate impact assessment before the change goes live |

---

# 11. Production Concerns

---

## 11.1 Score drift and retraining triggers

Distinguish three things that are routinely conflated:

| Type | What changed | Detected by | Typical remedy |
|------|--------------|-------------|----------------|
| **Data drift (covariate shift)** | $P(X)$ moved; $P(Y \mid X)$ unchanged | PSI, CSI | Often nothing — the model is still correct, the population moved. Check the cutoff still hits the target approve rate |
| **Concept drift** | $P(Y \mid X)$ changed | Actual vs expected bad rate by band; falling AUC on matured cohorts | Recalibrate if rank-ordering holds; rebuild if it does not |
| **Data quality failure** | A feed broke, a schema changed, a sentinel value appeared | Null-rate spikes, lumpy non-monotonic PSI concentrated in one or two bands, CSI spike on a single variable | Fix the pipeline. **Never** retrain on broken data |

**The ordering rule that saves you:** always ask "is this a data quality problem?" before "is this drift?" A broken bureau feed looks exactly like a population shift in PSI. That is why data-integrity gates run *before* the PSI computation in the pipeline.

**Rebuild triggers** (any one is sufficient to open an investigation; two or more usually mean rebuild):
- PSI > 0.25 sustained for two or more consecutive months and not explained by a known policy change
- AUC or KS on matured cohorts down > 10% relative to the development OOT figure
- Rank-order inversion between adjacent score bands, persisting across two quarters
- Actual/expected bad-rate ratio outside [0.8, 1.2] at portfolio level
- A material data-source change (bureau score version, new furnisher, a variable being deprecated)
- Scheduled refresh (most lenders rebuild application scorecards every 18–36 months regardless)

## 11.2 Champion / challenger in production

| Mode | Mechanics | Use when |
|------|-----------|----------|
| **Shadow scoring** | Challenger scores every application; no decisions taken | Always. It is free and it accumulates the evidence you need |
| **Swap-set simulation** | Compare decisions offline at matched approve rates (§8.10) | Before any promotion |
| **Randomised split** | A small % of traffic decisioned by the challenger | When you need real outcomes, and you have the risk appetite. Note the outcomes take a full performance window to read — this is not an A/B test with a one-week readout |
| **Phased rollout** | Promote by segment or by geography, monitoring each | Standard promotion path |

**The credit-specific constraint:** the readout takes 6–12 months because that is the performance window. You cannot run a two-week experiment. So the leading indicators — early-MOB delinquency of the challenger-approved cohort, swap-set composition, score-distribution stability — carry most of the decision weight, and the final outcome measurement is confirmatory.

## 11.3 Economic cycle sensitivity, PIT vs TTC

A model trained on 2012–2019 US or 2015–2019 India data learned a **benign** credit cycle. When the cycle turns:

**Discrimination degrades gracefully; calibration breaks immediately.** The variables that separate good from bad borrowers — delinquency history, utilisation, enquiry intensity — keep separating in a downturn. What changes is the *level*: everyone's PD rises, so a model calibrated to a 4% portfolio bad rate now systematically under-predicts against an 8% realised rate. **Rank-ordering is robust; calibration is fragile.** That is the single most useful sentence in this section, and the practical consequence is: **recalibrate before you rebuild.** An intercept shift fixes a level problem in an afternoon; a rebuild takes a quarter and needs revalidation.

| Framing | Definition | Where it is required |
|---------|------------|----------------------|
| **Point-in-time (PIT)** | PD reflects *current* conditions; rises in a downturn | IFRS 9 / CECL expected credit loss (explicitly forward-looking, macro-conditioned); origination decisioning |
| **Through-the-cycle (TTC)** | PD reflects a long-run average across the cycle; stable | Basel IRB regulatory capital — deliberately stable to avoid **procyclicality**, where capital requirements spike exactly when banks can least afford them |

Most application scorecards are naturally somewhere in between: the *ranking* is close to TTC (relative riskiness is stable), while the *calibration* is PIT (anchored to the bad rate of the development window). Saying that precisely is a strong signal in an interview.

**Mitigations:** include a full cycle in the development sample where data allows; hold out a stressed period as an additional OOT window if you have one; overlay an explicit macro adjustment for provisioning (a scalar or a regression on unemployment/GDP) rather than baking macro variables into the applicant-level scorecard; and stress-test — apply a 1.5×/2× bad-rate multiplier and check the cutoff still clears the profit hurdle.

## 11.4 Override tracking

An **override** is a decision that contradicts the score.

| Type | Definition | Concern |
|------|------------|---------|
| **High-side override** | Approve someone the score declined | Direct credit loss risk; also a fair-lending risk if overrides are not evenly distributed |
| **Low-side override** | Decline someone the score approved | Lost revenue; often driven by policy rules or fraud flags rather than judgement |

**Track two things: the rate and the performance.**

The **rate** should sit under about 5%. Above 10% the organisation is not really using the model — either the score is not trusted, or policy rules are doing the actual deciding and the score is theatre.

The **performance** is the more interesting metric. Back-test the high-side overrides: how did the accounts underwriters approved against the score actually perform?

- If overrides perform **worse** than the score predicted → the overrides are destroying value, and this is a training/incentive issue.
- If overrides perform **better** than the score predicted → **the underwriters know something the model does not.** That is a feature-discovery signal. Interview the underwriters, find out what they are looking at (a document, a call, a pattern in the bank statement), and get it into the next model.

That second case is one of the most valuable and least-used feedback loops in credit modelling, and it is a genuinely memorable thing to say in an interview.

**Fair-lending angle:** override authority is human discretion, which is where disparate treatment risk concentrates. Override rates must be monitored by proxied demographic group, not just in aggregate.

---

# 12. Python Implementations

---

## 12.1 WOE / IV calculator

```python
import numpy as np
import pandas as pd


def woe_iv_table(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: list | None = None,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Compute the WOE/IV bin table for one characteristic.

    Convention: WOE = ln(%Good / %Bad), so positive WOE means lower risk.
    target must be 1 = bad, 0 = good.

    alpha is the additive (Jeffreys) smoothing pseudocount, applied so that a
    bin with zero bads or zero goods yields a finite WOE instead of +/-inf.
    Prefer preventing empty bins via minimum-size constraints; smoothing is a
    backstop, not a substitute.
    """
    work = df[[feature, target]].copy()

    if bins is not None:
        work["bin"] = pd.cut(work[feature], bins=bins, include_lowest=True)
        work["bin"] = work["bin"].cat.add_categories(["MISSING"])
    else:
        work["bin"] = work[feature].astype("object")

    # Missing and special values get their own bin -- never imputed.
    work["bin"] = work["bin"].fillna("MISSING")

    grouped = work.groupby("bin", observed=True, dropna=False)[target].agg(
        n="count", bads="sum"
    )
    grouped["goods"] = grouped["n"] - grouped["bads"]

    k = len(grouped)
    total_goods = grouped["goods"].sum()
    total_bads = grouped["bads"].sum()

    grouped["pct_good"] = (grouped["goods"] + alpha) / (total_goods + alpha * k)
    grouped["pct_bad"] = (grouped["bads"] + alpha) / (total_bads + alpha * k)

    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv_contrib"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    grouped["bad_rate"] = grouped["bads"] / grouped["n"]
    grouped["pct_total"] = grouped["n"] / grouped["n"].sum()
    grouped["iv"] = grouped["iv_contrib"].sum()

    return grouped.reset_index()


def is_monotonic(woe_values: np.ndarray, tol: float = 1e-9) -> bool:
    """True if WOE is monotonically increasing or decreasing across bins."""
    diffs = np.diff(np.asarray(woe_values, dtype=float))
    return bool(np.all(diffs >= -tol) or np.all(diffs <= tol))


def screen_by_iv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    min_iv: float = 0.02,
    suspicious_iv: float = 0.5,
) -> pd.DataFrame:
    """Rank candidate variables by IV and flag both ends of the range.

    Anything above `suspicious_iv` is flagged for a leakage review -- it is not
    automatically dropped, because some behavioural variables legitimately land
    there, but it does not enter the model until someone has signed off.
    """
    rows = []
    for feat in features:
        # Numeric features get fine-classed first; categoricals used as-is.
        if pd.api.types.is_numeric_dtype(df[feat]) and df[feat].nunique() > 20:
            edges = np.unique(np.nanpercentile(df[feat], np.linspace(0, 100, 21)))
            edges[0], edges[-1] = -np.inf, np.inf
            table = woe_iv_table(df, feat, target, bins=list(edges))
        else:
            table = woe_iv_table(df, feat, target)

        iv = float(table["iv"].iloc[0])
        rows.append(
            {
                "feature": feat,
                "iv": iv,
                "n_bins": len(table),
                "monotonic": is_monotonic(table["woe"].values),
                "verdict": (
                    "DROP — not predictive" if iv < min_iv
                    else "REVIEW — check leakage" if iv > suspicious_iv
                    else "KEEP"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("iv", ascending=False)
```

## 12.2 Monotonic binner

```python
from sklearn.tree import DecisionTreeClassifier


def monotonic_bin(
    x: pd.Series,
    y: pd.Series,
    max_bins: int = 6,
    min_bin_frac: float = 0.05,
    min_bads_per_bin: int = 30,
) -> list[float]:
    """Find monotonic bin edges for a numeric characteristic.

    Two stages:
      1. A shallow decision tree proposes cut points that are informative about
         the target (subject to a minimum leaf size).
      2. Adjacent bins that violate monotonicity of the bad rate, or that are
         too small, are merged pairwise until the constraints hold. This is
         pool-adjacent-violators applied to the bad rate.

    Returns bin edges suitable for pandas.cut, with -inf / +inf endpoints.
    Missing values are handled outside this function -- they get their own bin.
    """
    mask = x.notna()
    x_clean, y_clean = x[mask].values, y[mask].values
    n = len(x_clean)

    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=max(int(min_bin_frac * n), 50),
        random_state=42,
    )
    tree.fit(x_clean.reshape(-1, 1), y_clean)

    thresholds = tree.tree_.threshold[tree.tree_.feature != -2]
    edges = [-np.inf] + sorted(float(t) for t in thresholds) + [np.inf]

    def bin_stats(edge_list):
        idx = np.digitize(x_clean, edge_list[1:-1])
        return pd.DataFrame(
            {
                "n": np.bincount(idx, minlength=len(edge_list) - 1),
                "bads": np.bincount(idx, weights=y_clean, minlength=len(edge_list) - 1),
            }
        ).assign(bad_rate=lambda d: d["bads"] / d["n"].replace(0, np.nan))

    # Merge until monotonic in bad rate and every bin meets the size floors.
    while len(edges) > 3:
        stats = bin_stats(edges)
        rates = stats["bad_rate"].values

        too_small = (stats["n"] < min_bin_frac * n) | (stats["bads"] < min_bads_per_bin)
        if too_small.any():
            # Merge the smallest offending bin into its closest-rate neighbour.
            j = int(stats.loc[too_small, "n"].idxmin())
        else:
            diffs = np.diff(rates)
            increasing = (diffs >= 0).sum() >= (diffs < 0).sum()
            violations = np.where(diffs < 0 if increasing else diffs > 0)[0]
            if len(violations) == 0:
                break
            # Merge across the smallest violation -- least information lost.
            j = int(violations[np.argmin(np.abs(diffs[violations]))])

        drop_at = min(max(j, 0), len(edges) - 3) + 1
        edges.pop(drop_at)

    return edges
```

## 12.3 KS statistic and the gains table

```python
def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact KS: max gap between the cumulative bad and good distributions.

    y_score is the predicted probability of bad (higher = riskier).
    """
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]

    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)

    return float(np.max(np.abs(cum_bad - cum_good)))


def gains_table(y_true: np.ndarray, y_score: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    """Decile gains/lift table -- the artefact credit teams actually read.

    Band 1 is the riskiest. Reports bad rate, lift, cumulative capture and the
    KS gap per band, so the KS value and the decile where it occurs are both
    visible rather than just the scalar.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["band"] = pd.qcut(df.index, q=n_bands, labels=range(1, n_bands + 1))

    out = df.groupby("band", observed=True)["y"].agg(n="count", bads="sum")
    out["goods"] = out["n"] - out["bads"]
    out["bad_rate"] = out["bads"] / out["n"]
    out["lift"] = out["bad_rate"] / df["y"].mean()
    out["cum_pct_bad"] = out["bads"].cumsum() / out["bads"].sum()
    out["cum_pct_good"] = out["goods"].cumsum() / out["goods"].sum()
    out["ks_gap"] = (out["cum_pct_bad"] - out["cum_pct_good"]) * 100

    return out.reset_index()
```

## 12.4 PSI and CSI

```python
def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    breakpoints: np.ndarray | None = None,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> tuple[float, pd.DataFrame]:
    """Population Stability Index, with the per-band contribution table.

    The band edges MUST be frozen at model build time and reused for every
    monitoring run. Re-deriving deciles from the current sample each month
    makes PSI structurally near-zero and useless -- a mistake that is
    surprisingly common and produces a monitoring system that never alarms.

    Returns (psi, contribution_table). Read the table, not just the scalar:
    a smooth directional shift and a spike in one band mean different things.
    """
    if breakpoints is None:
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)
    edges = np.concatenate([[-np.inf], breakpoints[1:-1], [np.inf]])

    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)

    contrib = (act_pct - exp_pct) * np.log(act_pct / exp_pct)

    table = pd.DataFrame(
        {
            "band": range(1, len(exp_pct) + 1),
            "expected_pct": exp_pct,
            "actual_pct": act_pct,
            "diff": act_pct - exp_pct,
            "ln_ratio": np.log(act_pct / exp_pct),
            "contribution": contrib,
        }
    )
    return float(contrib.sum()), table


def calculate_csi(
    dev_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    woe_bins: dict[str, list[float]],
) -> pd.DataFrame:
    """CSI for every characteristic, using the frozen development WOE bins.

    PSI says the score moved; CSI says which variable moved it. Always run both
    -- PSI alone is an alarm with no diagnosis.
    """
    rows = []
    for feature, edges in woe_bins.items():
        dev_counts = pd.cut(dev_df[feature], bins=edges).value_counts(normalize=True, sort=False)
        cur_counts = pd.cut(cur_df[feature], bins=edges).value_counts(normalize=True, sort=False)

        e = np.clip(dev_counts.values, 1e-4, None)
        a = np.clip(cur_counts.values, 1e-4, None)
        csi = float(np.sum((a - e) * np.log(a / e)))

        rows.append(
            {
                "feature": feature,
                "csi": csi,
                "status": (
                    "STABLE" if csi < 0.10
                    else "MONITOR" if csi < 0.25
                    else "SHIFTED"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("csi", ascending=False)
```

## 12.5 Scorecard scaling

```python
def build_scorecard(
    coefficients: dict[str, float],
    intercept: float,
    woe_tables: dict[str, pd.DataFrame],
    base_score: int = 600,
    base_odds: float = 30.0,
    pdo: int = 20,
) -> pd.DataFrame:
    """Convert logistic coefficients on WOE inputs into an additive points table.

    Assumes the model predicts P(bad) from WOE = ln(%Good / %Bad), so every
    coefficient should be negative. `odds` throughout means good:bad, so a
    higher score is lower risk.

        Points_ij = -Factor * beta_j * WOE_ij + (Offset - Factor * intercept) / k
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    k = len(coefficients)
    base_allocation = (offset - factor * intercept) / k

    if any(b > 0 for b in coefficients.values()):
        raise ValueError(
            "Positive coefficient on a WOE variable -- check for multicollinearity "
            "before scaling; the points for that characteristic would be inverted."
        )

    rows = []
    for feature, beta in coefficients.items():
        for _, row in woe_tables[feature].iterrows():
            rows.append(
                {
                    "characteristic": feature,
                    "attribute": row["bin"],
                    "woe": row["woe"],
                    "beta": beta,
                    "points": round(base_allocation - factor * beta * row["woe"]),
                }
            )

    return pd.DataFrame(rows)


def score_to_pd(score: float, base_score: int = 600, base_odds: float = 30.0, pdo: int = 20) -> float:
    """Invert the scorecard scaling to recover the predicted PD."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    log_odds_good = (score - offset) / factor
    return float(1.0 / (1.0 + np.exp(log_odds_good)))
```

---

# 13. Interview Questions with Strong Answers

---

### Q1: "Walk me through how you'd build a credit scorecard from scratch."

> "I'd sequence it by leverage, not by convenience. First, the target definition — I use roll-rate analysis to find the DPD bucket where the cure rate collapses, and vintage curves to find the month where cumulative bad rates flatten, which sets the performance window. That decision drives everything else, so I don't delegate it.
>
> Second, the sample frame: an observation point per account, features built strictly from before it, target strictly after, indeterminates excluded and counted, split by customer and then by time with the most recent months held out as out-of-time.
>
> Third, the variable pipeline: fine-class everything, compute WOE and IV, drop below 0.02, flag above 0.5 for a leakage review, coarse-class to monotonic bins with a minimum of 5% of the population and 30 bads per bin. Then correlation filter, VIF under 5, and a business review of every survivor.
>
> Fourth, fit logistic regression on the WOE columns, confirm every coefficient is negative, and check stability across folds. Fifth, scale to points with a chosen PDO, base score and base odds, and build the human-readable scorecard.
>
> Sixth — and this is the part that takes the longest — the validation battery: AUC, Gini, KS on train, out-of-sample and out-of-time; decile rank-ordering; calibration; PSI and CSI; metrics within every segment; swap-set analysis against the incumbent; and fair-lending testing. Then the monitoring plan and the documentation, which I write as I go rather than at the end."

---

### Q2 (trick): "0.94 AUC on a credit model is suspiciously high. Where's the leakage?"

> "It's the right question to ask, and I ask it of my own models. Three things make 0.94 defensible here, and one thing would make me abandon it.
>
> First, it was a **behavioural** scorecard, not an application scorecard. I was scoring existing customers using their own repayment history with us plus full bureau data. Application scorecards live at 0.70–0.82 because you're predicting a stranger; behavioural scorecards live at 0.85–0.95 because past repayment behaviour to you is nearly a leading indicator. If I'd claimed 0.94 on an application scorecard I'd expect to be disbelieved.
>
> Second, the target was 30+ DPD over a six-month window. That's an easier target than ever-90 over twelve months — some of what I'm predicting is administrative lateness with clear precursors, not deep credit distress. I'd say that plainly rather than let the number stand unqualified.
>
> Third, the leakage checks: I audited every feature for whether it exists for a declined applicant at decision time, I reconstructed features as-of the observation date rather than reading current state from the warehouse, I fitted the WOE bins on training data only, and I investigated every variable with IV above 0.5.
>
> And the thing that would have made me abandon it: if train, validation and out-of-time AUC had all been identical at 0.94. They were 0.95, 0.94, 0.94 — a small, believable degradation. A model that doesn't degrade at all out-of-time is usually reading the answer."

---

### Q3: "Why logistic regression when XGBoost beats it?"

> "It does beat it — about two AUC points in my benchmark, 0.96 versus 0.94, and I'd say that rather than pretend the gap doesn't exist. The question is what those two points cost in a decisioning model.
>
> Three costs. Exact decomposability: the scorecard's contribution per characteristic *is* the model, so adverse-action reasons are arithmetic. SHAP gives an attribution that depends on a background distribution, and two defensible configurations can rank an applicant's decline reasons differently — a validator will find that. Monotonicity: with WOE inputs it's inherited by construction; with a GBM I have to impose constraints and then prove each one is correctly signed. And stability: refit a GBM on three more months and individual predictions move materially, whereas a forty-variable scorecard moves by a point or two — which matters because the policy team has set limits and prices against those points.
>
> Where I'd absolutely use the GBM: fraud, collections prioritisation, marketing, early-warning triage — anywhere the adverse-action obligation doesn't bind. And the pattern I actually like is using the GBM as a feature discoverer: read its SHAP interaction values, find the two or three interactions it's exploiting, hand-build them as new characteristics, WOE-bin them, and put them in the scorecard. You recover part of the lift inside an artefact you can still defend."

---

### Q4: "KS vs AUC vs Gini — when do they disagree?"

> "Gini and AUC never disagree — Gini is exactly two-times-AUC-minus-one, so they're the same statistic in different units. Europe and India report Gini, US tech reports AUC. If someone asks me the difference between them, part of what they're testing is whether I'll invent one.
>
> KS is genuinely different. AUC integrates performance across every threshold; KS is the single maximum gap between the cumulative bad and good distributions. So two models with identical AUC can have different KS if their ROC curves cross — one concentrating separation in the middle, the other spreading it into the tails.
>
> There's a relationship worth knowing: for a concave ROC, Gini is always at least KS. You can see it geometrically — the minimal concave ROC achieving a given max gap is two line segments, and that triangle's Gini equals KS exactly; any real curve through the same point encloses more area. So Gini is an upper bound on KS, and the Gini-over-KS ratio tells you whether the model's power is concentrated at one cut point or distributed. Around 1.0 means it's essentially one good threshold; 1.2 to 1.4 is healthy.
>
> Practically I report all three, but I make the decision on the bad-rate capture at the cutoff we actually operate. A KS that peaks at the 30th percentile is irrelevant to me if we cut at the 10th."

---

### Q5 (trick): "One of your variables has an IV of 0.8. Good news?"

> "No — that's an alarm, not a trophy. IV above 0.5 means one variable nearly separates goods from bads on its own, and the base rate of that being leakage rather than genius is high.
>
> My checklist: Is it a restatement of the target — something like max-DPD-ever or account status? Is it timestamped after the observation point, which is the classic warehouse-stores-current-state failure? Is it a consequence of the decision rather than an input to it, like collections contact count? Is the IV concentrated in one thin bin holding two percent of the population, in which case it's small-sample noise rather than signal? And does it hold out-of-time — a leaked variable keeps its IV because the leak is structural, whereas a noisy one collapses, so that test actually discriminates between the two causes.
>
> Sometimes it survives all of that. On a behavioural scorecard, max DPD in the last six months legitimately hits 0.6 to 0.9 because recent delinquency is close to a leading indicator of near-term delinquency. But surviving the checklist is a conclusion I have to reach, not an assumption I start from."

---

### Q6: "What exactly is WOE, and why does it make logistic regression work better?"

> "WOE is the log ratio of the share of goods to the share of bads in a bin. The reason it works is worth deriving: the empirical log-odds of good within a bin equal the WOE plus the population log-odds — that's just algebra on the definition. So the WOE value *is* the bin's log-odds, up to a constant.
>
> Logistic regression assumes the log-odds are linear in the inputs. WOE hands it inputs where that assumption is satisfied exactly, by construction. There's a nice consequence: fit a univariate logistic regression on a single WOE variable and you must get a coefficient of exactly minus one and an intercept equal to the sample log-odds. I use that as a unit test on my WOE code.
>
> Three other benefits: everything lands on a common scale so continuous and categorical variables mix without dummy explosion; outliers are immune because binning collapses the tails; and missing values get an empirically-measured bin instead of an imputation.
>
> The cost I'd volunteer: WOE is supervised — it uses the target. So it must be fitted on training data only and frozen, and with thin bins it will overfit, which is exactly why the minimum-bads-per-bin rule exists."

---

### Q7: "Why do you require monotonic binning? What breaks without it?"

> "Four reasons, and they're independent. Interpretability: the scorecard has to complete the sentence 'higher utilisation, fewer points' — a row saying 50-to-80 percent scores worse than 80-to-100 percent is unusable to a credit officer. Regulatory defensibility: an adverse-action reason has to survive 'would this applicant have been approved with a worse value?' Under monotonicity the answer is always no; without it, it can be yes, and that's an arbitrariness argument handed to a regulator. Business sense as a specification test: domain knowledge says utilisation and enquiries are monotonically bad, so if my empirical binning disagrees, the likely causes are a thin-bin artefact, sentinel codes contaminating the numeric range, or a genuine sub-population mixture — all three worth finding. And robustness: non-monotonic bins are usually fitting noise, and when they invert out-of-time they don't just lose accuracy, they reverse the direction of points for a slice of the population.
>
> The counter-argument I'd volunteer: some relationships are genuinely non-monotonic. Age is the classic U-shape, and zero utilisation can signal a dormant customer rather than a prudent one. The right handling isn't to force monotonicity through the anomaly — it's to split the variable, so a separate zero-utilisation indicator plus monotonic binning of the positive range. What you never do is silently flatten a real U-shape and lose the signal."

---

### Q8 (trick): "You need the model now, but the performance window is twelve months. What do you do?"

> "You never have a mature target when you need one — that's the permanent condition of credit modelling, not an edge case.
>
> First, I'd challenge the twelve months. Vintage curves tell me where cumulative bad rates flatten. If 90% of eventual bads have emerged by month nine, a nine-month window with a documented censoring adjustment is defensible and buys me a quarter.
>
> Second, use a **proxy target with a bridge**. Build on an early indicator that matures fast — ever-30 by month six — and estimate the roll rate from 30 to 90 on historical cohorts. The model ranks on the fast target; the roll rate translates the rank into the 90-DPD probability the business needs for pricing. I'd be explicit that this imports the assumption that the roll rate is stable, and I'd monitor it.
>
> Third, **partially-matured cohorts with survival methods** — a discrete-time hazard model uses accounts that have only reached month four without pretending they're good, which a naive binary target does.
>
> Fourth, if there's genuinely no history, start with a **policy-rules-plus-bureau-score** decision, book deliberately conservatively, and treat the first six months as data acquisition. Score with the vendor bureau score, keep the population wider than you'd like so you actually observe the range, and rebuild on your own data.
>
> What I would not do is train on three months of performance and present the AUC as if the window were closed. That systematically understates risk, because the slow-emerging bads are all sitting in the good class."

---

### Q9 (trick): "What's reject inference, and why is it philosophically dubious?"

> "You only see outcomes for people you approved, but the model scores everyone who applies. Reject inference is the family of methods for estimating the bad probability over the full through-the-door population.
>
> The first distinction I'd draw is the one most people skip. If the old approval policy was a deterministic function of the variables I already have, then conditional on those variables approval carries no extra information about the outcome — there's no bias, just range restriction, meaning no data in the declined region. That's an extrapolation problem, and no inference method creates information that was never observed. The genuine bias case is when underwriters used judgement I can't observe — a document, a phone call, a prior model's variables. Then the unobserved factor is favourable among approvals and the observed bad rate understates the truth.
>
> And that's exactly the case the standard methods don't fix. Parcelling and fuzzy augmentation label rejects using a model fit on accepts, then train on the combination — so they propagate the accept-model's assumptions into the reject region rather than test them. Inverse-probability augmentation assumes missing-at-random, which is the case where there was no bias to begin with. Heckman needs an exclusion restriction — something affecting approval but not default — and in credit essentially nothing qualifies, because anything that influenced approval was believed to predict default. So identification comes from the bivariate-normality assumption doing the work of an instrument.
>
> The deepest problem is that it can't be validated where it claims to help. I can measure the reject-inferred model on accepts; I have no outcomes on rejects. So I do it — validators expect it, and it genuinely stabilises extrapolation in the low-score region — but I present results with and without it, and I'd argue for a small randomised approval programme on marginal declines. Buying the data is the only thing that actually resolves it."

---

### Q10 (trick): "Your PSI is 0.30. What do you do?"

> "Not retrain. Retraining is the last step, not the first.
>
> Step one, is it real? A broken bureau feed or a schema change looks exactly like a population shift. So I check null rates, row counts, value ranges, and whether any sentinel codes started appearing. A data-quality failure usually produces a lumpy PSI concentrated in one or two bands; a genuine mix shift is smooth and directional. That's why the integrity gates run before the PSI computation.
>
> Step two, which variables? PSI is an alarm; CSI is the diagnosis. If one characteristic carries most of the drift, that's a specific, findable cause — a new bureau score version, a data provider changing a definition, a variable's population changing. If it's spread evenly across characteristics, it's a genuine population shift.
>
> Step three, what changed in the business? A new acquisition channel, a marketing campaign into a new segment, a policy cutoff change, a competitor exit. In my experience most large PSI moves have a mundane business explanation that a five-minute conversation surfaces.
>
> Step four — and this is the important one — **PSI is not a performance metric.** It measures input drift, not accuracy. So I go and check whether the model still works: rank-ordering on matured cohorts, actual versus expected bad rate by band, AUC on whatever has matured. It's entirely possible to have PSI at 0.30 with the model performing correctly, because a shifted population is not a wrong model.
>
> Then the action depends on what I found. Data problem: fix the pipeline. Population shifted but rank-ordering holds and calibration is off: recalibrate the intercept — that's an afternoon, not a quarter. Rank-ordering has broken or is inverting between bands: rebuild. And in every case I'd check whether the cutoff still produces the intended approve rate, because a shifted score distribution silently changes the approval rate even when the model is fine."

---

### Q11: "How do you explain a decline to a customer?"

> "The scorecard makes it mechanical, which is a large part of why it's still the standard. For each characteristic I compute the applicant's points shortfall against the maximum attainable points for that characteristic, rank the shortfalls, take the top three or four, and map each to a pre-approved plain-language reason code.
>
> So for a real applicant it comes out as: bureau score below our requirement, proportion of balances to credit limits too high, recent delinquency on one or more accounts. That's exact arithmetic on the model itself, not a post-hoc attribution — which matters legally, because under Regulation B I have to state the specific principal reasons, and 'specific' is easier to defend when the decomposition is the model rather than an approximation of it.
>
> Two things I'd add. The reason codes have to be plain language that a person can act on — 'your utilisation is too high' tells someone what to do; 'your device tenure segment is unfavourable' doesn't, which is one more argument against exotic behavioural variables in a decisioning model. And if a credit report was used, FCRA adds its own disclosure obligations on top of ECOA's, so the notice has to satisfy both."

---

### Q12 (trick): "Your model is accurate, but it declines more applicants in one ZIP code. What now?"

> "Accuracy is not a defence against disparate impact. That's the first thing to be clear about — a facially neutral model that produces significantly worse outcomes for a protected group is actionable regardless of how well it predicts, and the burden shifts to demonstrating business necessity and then to whether a less discriminatory alternative exists.
>
> So, in order. Measure it properly: proxy the demographics — BISG is the standard where demographic data isn't collected — and compute the adverse impact ratio at the cutoff we actually use, not in aggregate, since where you cut determines the disparity. I'd note that the four-fifths rule is a screening heuristic imported from employment law, not a legal standard in lending.
>
> Then find the driver. Which characteristics contribute most to the score gap between groups? ZIP code in US consumer lending I'd treat as presumptively unusable — it's strongly correlated with race because of historical residential segregation, so it's a proxy regardless of what it's named. But even without raw geography, features can carry geographic information: a regional aggregate, a branch identifier, sometimes a device or channel variable.
>
> Then the less-discriminatory-alternative search, which is an affirmative obligation and not a defence I wait to need. Iteratively drop or re-specify the highest-contributing characteristics and measure the AUC cost of each. If dropping a variable cuts the disparity by six points at a cost of 0.003 AUC, I've found an LDA and I'm expected to adopt it. If every alternative costs real performance, I document the search — searching and finding nothing is a defensible position; not searching is not.
>
> And I'd escalate rather than resolve it alone. This is a legal and compliance decision with a modelling input, not a modelling decision. My job is to give them an accurate disparity measurement, an honest driver analysis, and a menu of alternatives with their performance costs."

---

### Q13: "How do you choose the bad definition?"

> "From the roll-rate matrix, not from convention. I look at the transition probabilities between delinquency buckets and find where the cure rate collapses and the roll-forward rate approaches one. If 30-DPD accounts cure 60% of the time but 60-DPD accounts cure 6% and roll to 90 at 73%, then the point where delinquency stops being noise and becomes default is somewhere around 60. That's empirical justification, and it's a much better answer than 'ninety is the Basel standard.'
>
> Then I trade it against the window. A stricter definition needs a longer performance window and produces fewer events, which costs statistical power and delays every retrain. At Fibe we used 30+ over six months because it was a behavioural scorecard on short-tenure products and we needed a target that matured fast enough to retrain on. I'd defend that, and I'd also say plainly that a 30-DPD target inflates apparent performance relative to 90-DPD on the same book, and that the PD it produces can't be dropped into an expected-loss calculation that assumes Basel default without a roll-rate bridge."

---

### Q14: "Explain vintage analysis and roll-rate analysis, and what each is for."

> "They answer different questions. Vintage analysis groups accounts by booking month and plots cumulative bad rate against months on book. It does two things: it tells me where the bad rate flattens, which sets the minimum performance window; and it lets me compare recent cohorts to older ones at the same maturity, which is the earliest honest read on portfolio quality I have. If the 2025-Q1 vintage is at 4.0% bad at month six when the 2024-Q1 vintage was at 2.4%, that's actionable now — nine months before the twelve-month number exists.
>
> Roll-rate analysis is a monthly transition matrix between delinquency buckets. It tells me the probability an account moves from current to 1-29, from 30-59 to 60-89, and so on. Its main use is justifying the DPD threshold in the bad definition, by showing where cure rates collapse. Its second use is short-horizon loss forecasting — you can propagate the current bucket distribution forward through the matrix.
>
> Together: roll rates tell me *what* bad means; vintage curves tell me *how long* I have to wait to see it."

---

### Q15: "Your AUC held steady but the portfolio lost money. How?"

> "Discrimination and calibration are different properties, and only one of them held.
>
> AUC measures rank-ordering — whether risky applicants score worse than safe ones. It's completely invariant to a monotone transformation of the scores. So if every applicant's true PD doubled in a downturn, the ranking would be identical and AUC would be unchanged, while the model systematically under-predicts losses across the board. The pricing, the limits and the provisions were all set against the old level, so the book loses money with a perfectly healthy AUC.
>
> This is the general pattern in a turning cycle: rank-ordering is robust, calibration is fragile. The variables that separate good borrowers from bad ones keep separating; what moves is the level.
>
> How I'd have caught it: actual versus expected bad rate by score band, monitored quarterly. If every band's actual is roughly 1.8 times expected while the ordering is intact, that's a pure calibration break and the fix is an intercept shift — an afternoon, not a rebuild. If the ratio varies wildly across bands or the ordering inverts, the model itself has broken and needs rebuilding.
>
> The other candidate explanation I'd check: the loss might not be a PD problem at all. If LGD deteriorated because collections got worse or collateral values fell, or if EAD rose because of limit utilisation before default, the PD model can be blameless. Expected loss has three factors and only one of them is my model."

---

### Q16: "PD, LGD, EAD — how do they combine, and which would you model first?"

> "Expected loss is the product: PD times LGD times EAD. For a ₹50,000 loan with an 8% PD, 85% LGD and ₹45,000 exposure at default, that's about ₹3,060 of expected loss, and the loan only makes sense if the revenue over its expected life clears that plus funding and servicing cost.
>
> I'd model PD first, for two reasons. It varies most across applicants — LGD and EAD are much more product- and process-driven, determined by collections effectiveness, collateral and limit policy rather than by who the borrower is. And it's the component the approve/decline decision actually controls: declining an applicant sets their contribution to zero, whereas nothing about the underwriting decision changes how much you recover after default.
>
> The one caution: the three aren't independent. In a downturn PD rises *and* recoveries fall, so LGD rises at the same time — the correlation is exactly why stress scenarios use downturn LGD rather than through-the-cycle LGD. Multiplying an average PD by an average LGD understates tail loss."

---

### Q17: "How do you set the cutoff?"

> "Not by maximising KS, which is the common wrong answer. KS gives you the point of maximum separation between goods and bads; the business wants the point of maximum profit, and those coincide only by accident.
>
> I write down the expected profit function: for each applicant above the cutoff, the probability of good times the expected margin, minus the probability of bad times LGD times EAD. Sweep the cutoff, plot profit, and find the maximum. That requires the business to state a margin assumption and an LGD assumption explicitly, which is itself valuable — those numbers are usually implicit and inconsistent across teams.
>
> Then I bring the trade-off table rather than a single number: at each candidate cutoff, the approve rate, the expected bad rate, the volume, and the expected profit. Because in practice the decision is rarely a pure profit maximisation — there's a growth target, an underwriting capacity constraint on the manual-review band, and a risk appetite ceiling on portfolio bad rate. My job is to make the trade-off legible so the business chooses with its eyes open, not to hand them one number.
>
> And in practice it's not one cutoff but three: auto-approve above one score, auto-decline below another, manual review in between, with the review band sized to the underwriting team's actual capacity."

---

### Q18: "What is swap-set analysis and why isn't AUC enough?"

> "AUC tells me the new model ranks better on average. It doesn't tell me what changes, and the business decision is entirely about what changes.
>
> So I hold the approve rate constant and classify every applicant into four cells: approved by both, declined by both, and the two swap sets. Swap-outs were approved by the old model and declined by the new one; swap-ins are the reverse.
>
> The asymmetry is what makes this powerful. For swap-outs I have **real observed outcomes**, because the old model approved them and we booked them. For swap-ins I have nothing, because they were declined — so any claim about them is model-versus-model and has to be flagged as such.
>
> That means the honest evidence lives in the swap-out set. If the accounts my new model rejects and my old model approved defaulted at 19% while the portfolio cutoff bad rate is around 8%, that's a real, evidenced improvement, not a metric improvement.
>
> I'd always add the segment profile of each swap set. If the swap-ins are concentrated in one geography or one thin-file segment, that's both a fair-lending conversation and a concentration-risk conversation, independent of whether the average bad rate looks fine."

---

### Q19: "Point-in-time versus through-the-cycle PD — what's the difference and when does each matter?"

> "A point-in-time PD reflects current conditions and moves with the cycle: it rises in a downturn. A through-the-cycle PD reflects a long-run average and is deliberately stable.
>
> The distinction is regulatory, not academic. Basel IRB capital wants something closer to through-the-cycle precisely to avoid procyclicality — if capital requirements spiked in every downturn, banks would be forced to deleverage at exactly the worst moment, which amplifies the downturn. IFRS 9 and CECL want the opposite: explicitly forward-looking, macro-conditioned, point-in-time expected credit loss.
>
> Most application scorecards sit in between, and being precise about *how* is the useful part: the **ranking** is close to through-the-cycle, because relative riskiness between borrowers is fairly stable across the cycle, while the **calibration** is point-in-time, anchored to the bad rate of whatever window you trained on. That's why the practical response to a cycle turn is recalibration rather than a rebuild — the ordering survives, the level doesn't.
>
> For provisioning I'd overlay the macro adjustment as an explicit scalar or a regression on unemployment, rather than putting macro variables inside the applicant-level scorecard. Keeping the applicant model and the macro conditioning separate means I can update the macro view monthly without revalidating the scorecard."

---

### Q20: "How do you handle missing bureau data?"

> "Missing gets its own bin with its own empirically-measured WOE. I don't impute, and the reason is both statistical and ethical.
>
> Statistically, missingness is informative. 'No bureau record' has an observable bad rate, and in my experience it lands somewhere between the mid and low bureau bands — nothing like the population median. Imputing the median tells the model this person is average, which is a fabrication.
>
> Ethically, it's the population I understand least, and it's disproportionately young, immigrant and underbanked applicants. Imputing a credit history for people who don't have one, and then declining them on it, is exactly the pattern fair-lending review is designed to catch.
>
> I'd also distinguish types of missing, because they mean different things: a true no-hit at the bureau, a structural not-applicable like utilisation for someone with no revolving line, and a sentinel code for a suppressed value. Bureau feeds are full of codes like minus-one and nine-nines, and if those leak into numeric binning they'll be treated as extreme values and destroy both the cut points and the monotonicity. So the first step is always mapping special values to named categories before anything numeric happens.
>
> And if the thin-file population is large — which in Indian fintech it is, often 25 to 40 percent — I'd validate the model *within* that segment separately, and consider a dedicated thin-file scorecard on transaction and cash-flow features if the pooled model underperforms there."

---

### Q21: "A coefficient on one of your WOE variables came out positive. What happened?"

> "That's a bug report, not a finding. With WOE defined as log of percent-good over percent-bad and a model predicting probability of bad, every coefficient must be negative — higher WOE means more good, which means lower risk.
>
> The overwhelmingly likely cause is multicollinearity. Bureau variables are extravagantly correlated — number of accounts, number of open accounts, number of active trade lines all measure nearly the same thing. When two near-duplicates are in the model together, the fit can assign a large negative coefficient to one and a positive one to the other, because only the difference between them is being identified. I'd check VIF, and if it's above five I'd drop the lower-IV member of the pair or combine them into a single characteristic.
>
> Other candidates: a genuine suppressor effect, which is real but rare and needs a story before I'd believe it; a sign error in my WOE code, which the univariate unit test would catch — a single-variable fit must return exactly minus one; or quasi-separation, where a bin has almost no bads and the coefficient is being driven by a handful of observations, which shows up as an enormous standard error.
>
> What I wouldn't do is ship it. A positive coefficient inverts the points for that characteristic — the scorecard would award more points for worse values, which is both wrong and indefensible in an adverse-action notice."

---

### Q22: "How would you actually get a GBM into production at a regulated lender without getting fired?"

> "By not putting it on the decisioning path first.
>
> Step one, deploy it where the adverse-action obligation doesn't bind: fraud detection, collections prioritisation, marketing and pre-qualification, early-warning triage on the existing book. Same modelling skill, materially lighter governance burden, and you build institutional experience with the tooling and the monitoring.
>
> Step two, run it as a permanent shadow challenger against the scorecard champion. No decisions, full scoring, full monitoring. You accumulate a real evidence base on the accuracy gap — and if that gap widens over time, that's itself a rebuild trigger for the champion, because it means the linear-in-WOE structure is missing newly-emerged non-linearity.
>
> Step three, if you want it on the decisioning path, do the governance work up front rather than arguing about it afterwards: monotonic constraints on every characteristic with a documented sign, a fixed SHAP configuration with the background distribution frozen and versioned so reason codes are reproducible, stability testing across refreshes, and a comparison showing the reason codes agree with the scorecard's on the overlapping population.
>
> Step four, and honestly the highest-value move: use it as a feature discoverer for the scorecard. Read the SHAP interaction values, find the two or three interactions it's exploiting, hand-build those as characteristics, WOE-bin them, and put them into the logistic model. You get a real share of the lift inside an artefact that's already approved.
>
> The framing that matters in the room: I'm not arguing the GBM is unusable, I'm sequencing so that each step earns the right to the next one with evidence."

---

### Q23: "Out-of-sample versus out-of-time — why does credit insist on OOT?"

> "Out-of-sample is a random hold-out from the same period, so it tests generalisation to new *individuals*. Out-of-time is a later period entirely excluded from training, so it tests generalisation to a new *population and environment*.
>
> Credit data is non-stationary in ways an out-of-sample split structurally cannot detect. Between the training window and deployment, the applicant mix changes because marketing opened a new channel, the policy changes because someone moved a cutoff and rewrote who's in the population, the macro environment moves, and the bureau itself changes — new score versions, new furnishers, new reporting rules. A random hold-out holds all of that constant and will report a perfectly happy number for a model that's about to fail.
>
> Building it correctly matters: the OOT window has to be excluded from training, from feature selection, *and* from binning. That last one is where people cheat without noticing — if the WOE bins were fitted on the full history, the OOT number is contaminated.
>
> What I expect to see: a small degradation, AUC down one or two points. That's healthy. Zero degradation is suspicious. And more than five points means the model is fitting period-specific structure rather than durable risk relationships."

---

### Q24: "Your model was trained on 2015–2019 data and now we're in a recession. What happens?"

> "Rank-ordering mostly survives; calibration breaks immediately. That's the pattern, and it dictates the response.
>
> The variables that separate good borrowers from bad ones — delinquency history, utilisation, enquiry intensity, account vintage — keep separating in a downturn. Someone at 90% utilisation with two recent missed payments is still riskier than someone at 10% with a clean file. What changes is the level: every PD rises, so a model calibrated to a 4% portfolio bad rate systematically under-predicts against an 8% realised rate. And every downstream number — pricing, limits, provisions — was set against the old level.
>
> So the immediate response is recalibration, not rebuild. An intercept shift corrects a pure level error, and I can do it as soon as I have enough realised outcomes to estimate the new base rate. A rebuild takes a quarter plus revalidation, and I'd only start one if the actual-versus-expected ratio varies wildly across bands or if rank-ordering starts inverting — both signs that the relationships themselves changed, not just the level.
>
> Second, I'd expect *some* differential deterioration. A recession doesn't hit uniformly — if it's concentrated in particular sectors, borrowers in those sectors get worse in a way employment-type variables may not capture. So I'd cut performance by segment, not just look at the pooled number.
>
> Third, I'd flag the structural limitation openly: the model never saw a downturn, so its behaviour in one is an extrapolation. That belongs in the limitations section of the model document, written before the downturn rather than after — and it's a strong argument for stress testing at build time, applying a 1.5x or 2x bad-rate multiplier and checking the cutoff still clears the profit hurdle."

---

### Q25: "How do you monitor a scorecard in production, and what triggers a rebuild?"

> "Layered by frequency. Daily: scoring volume, null rates, score distribution mean and standard deviation, and hard data-integrity gates that halt scoring on failure. Monthly: PSI on the score, CSI on every characteristic, approve rate by band, override rate, and early-months-on-book delinquency by vintage. Quarterly, once accounts have matured: AUC, KS, rank-ordering by band, actual versus expected bad rate, segment-level metrics, and the champion-versus-challenger comparison. Annually: full independent revalidation and a fair-lending re-run.
>
> The ordering rule that matters: always ask 'is this a data quality problem?' before 'is this drift?' A broken bureau feed looks exactly like a population shift in PSI. That's why the integrity gates run before the PSI computation, not after.
>
> Rebuild triggers, any one of which opens an investigation: PSI above 0.25 sustained for two consecutive months without a known policy explanation; AUC or KS on matured cohorts down more than 10% relative to the development OOT figure; a rank-order inversion between adjacent bands persisting across two quarters; actual-over-expected bad rate outside 0.8 to 1.2 at portfolio level; a material data source change like a bureau score version; or the scheduled refresh, which for most application scorecards is every 18 to 36 months regardless of whether anything looks wrong.
>
> And the distinction that saves the most work: a shifted population with intact rank-ordering usually needs a recalibration, not a rebuild. People conflate those and spend a quarter solving an afternoon's problem."

---

### Q26: "How would you build a scorecard for a brand-new product with no historical data?"

> "You can't, so the honest answer is about sequencing rather than modelling.
>
> Phase one, launch on policy rules plus a vendor bureau score. Conservative cutoffs, small limits, tight eligibility. This isn't a model, it's a defensible starting policy, and I'd say so rather than dress it up.
>
> Phase two — and this is the part people skip — deliberately book **wider than optimal** for a limited period. If I only approve the safest 20%, I will never observe the risk relationships in the other 80%, and my first real model will be trained on a range-restricted sample that can't extrapolate. Booking wider costs known losses; it buys the only data that lets me build anything. I'd budget it explicitly as data acquisition and get the number agreed in advance, because it will look like a mistake in month three and I want the rationale on record.
>
> Phase three, borrow structure. If there's an adjacent product with history, I can transfer the bad definition, the binning policy and often the characteristic set, and refit coefficients as soon as I have enough events. That's much faster than starting from nothing.
>
> Phase four, use a fast-maturing proxy target — ever-30 at month three or four — to get a first model out, then migrate to the real target as cohorts mature.
>
> Throughout, I'd run vintage curves obsessively, because with no model the vintage curve is the only early read on whether the policy is working."

---

### Q27: "What's the Hosmer-Lemeshow test, and why do people distrust it?"

> "It's a calibration test. Group applicants into typically ten bins by predicted probability, compare observed bads to expected bads in each bin, and form a chi-square statistic on roughly G-minus-2 degrees of freedom. A significant result means the predicted probabilities don't match observed frequencies.
>
> Three reasons it's distrusted. It depends on an arbitrary choice of the number of groups, and the answer genuinely changes with it — ten versus twenty groups can flip the conclusion. Implementations differ in how they handle ties and grouping, so two software packages disagree on the same data. And most importantly it's a null-hypothesis test, so its power scales with sample size: on a 500,000-row credit sample it will reject for a miscalibration far too small to matter to any decision. Rejecting is the default outcome, which makes it nearly uninformative.
>
> So I report it because validators expect to see it, and I *decide* on the calibration plot — predicted versus observed by bin against the 45-degree line — plus the calibration slope and intercept from regressing outcome on predicted logit. A slope below one tells me predictions are too extreme; an intercept shift tells me the base rate moved, which is the fixable case. Those give me an effect size and a direction, which a p-value doesn't."

---

### Q28: "The business wants to increase approvals by 10%. What's your answer?"

> "My answer is a table, not a yes or a no. The model tells you what happens; the risk appetite decides whether to do it.
>
> The table: at each candidate cutoff, the approve rate, the expected bad rate on the incremental population, the expected loss, the expected revenue, and the net. Approving 10% more means reaching into the next score band down, and I can tell them precisely what that band's expected bad rate is — say the current marginal band runs at 8% and the next runs at 14%. Then it's an explicit trade: this much volume, this much incremental loss, this net.
>
> Three things I'd raise unprompted. First, the incremental population is where my model is least reliable, because it's closer to the region the old policy declined and my training data thins out there. So the predicted bad rate on the swap-in population carries wider error bars than the number suggests, and I'd say so.
>
> Second, there are usually better routes to the same goal than moving the cutoff. Improving the model's ranking lets you approve more at the *same* bad rate — that's strictly better. So is reducing the manual-review band by automating decisions the model is confident about, or fixing a conversion drop-off in the application funnel. If the real goal is volume, the cutoff is only one lever and it's the one with the most direct risk cost.
>
> Third, I'd want to phase it and measure it. Move the cutoff for a randomised portion of traffic, watch early-months-on-book delinquency against the predicted rate, and only roll it out fully once the leading indicators confirm the projection. Given the performance window, the full readout takes months — so I'd design the measurement before the change, not after."

---

### Q29: "How do you detect that a feature is a proxy for a protected attribute?"

> "Directly measure it. Take the protected attribute — proxied via BISG where it isn't collected — and model it from the candidate feature. If a feature predicts group membership at, say, 0.75 AUC, it's a proxy regardless of what it's called. That's a simple, quantitative screen I'd run over every candidate variable, not just the obviously suspicious ones.
>
> Then the second question, which is the one that actually decides it: does the feature have independent predictive power for default *after* controlling for the legitimate variables it's standing in for? Usually a 'geographic risk score' is largely a stand-in for income and employment stability, which I can measure directly. If adding it on top of those adds little AUC but a lot of disparity, that's an easy drop and it's exactly what the less-discriminatory-alternative search is looking for.
>
> The features I'd treat as presumptively unusable in US consumer lending decisioning: ZIP code and any raw geographic identifier, name-derived features, university attended, employer identity, device price and model. If geography genuinely matters, route it through an explicitly justified economic variable like regional unemployment rather than the raw identifier — and expect to defend even that.
>
> The thing I'd be careful not to overclaim: removing proxies doesn't guarantee a fair model. Disparity can survive through the legitimate variables themselves, because credit history is itself shaped by historical inequity. That's a real limitation, it belongs in the documentation, and it's why the LDA search is an ongoing obligation rather than a one-time check."

---

### Q30: "Your scorecard has 45 variables. Isn't that too many? Or too few?"

> "It's in the normal band — most production scorecards land somewhere between 10 and 20 characteristics per scorecard, and 45 is on the high side but not unusual for a behavioural model with multiple data sources. But the count isn't the interesting question; how you got there is.
>
> The case for fewer: each characteristic is a row a human reads, a variable that can break in the pipeline, and a source of instability across refreshes. Marginal variables — IV around 0.03, contributing a point or two of score range — add operational risk and essentially no discrimination. There's a real argument for pruning anything that doesn't earn its maintenance cost.
>
> The case for these 45: they came out of a funnel from 1,000-plus candidates through IV screening, correlation filtering, VIF under five, monotonicity validation and a business review. Every survivor is individually predictive, not redundant with another, monotonic, and someone from credit policy signed off that it makes sense. That's a defensible provenance.
>
> What I'd actually do to answer it properly is measure it: refit with the top 15 by contribution and compare OOT AUC and KS. If the 45-variable model beats the 15-variable one by 0.002 AUC, ship the 15 — the simpler model is worth more than that. If it's 0.02, the extra variables are earning their keep. I'd rather answer that question with a number than with a philosophy."

---

# 14. Key Takeaways & Cheat Sheet

---

## Formula reference

| Quantity | Formula |
|----------|---------|
| **Expected loss** | $EL = PD \times LGD \times EAD$ |
| **WOE** | $WOE_i = \ln\left(\%\text{Good}_i / \%\text{Bad}_i\right)$ |
| **Bin log-odds identity** | $\ln(g_i/b_i) = WOE_i + \ln(G/B)$ |
| **Information Value** | $IV = \sum_i (\%\text{Good}_i - \%\text{Bad}_i)\,WOE_i = D_{KL}(G \Vert B) + D_{KL}(B \Vert G)$ |
| **Smoothed WOE** | $\ln\frac{(g_i+\alpha)/(G+\alpha k)}{(b_i+\alpha)/(B+\alpha k)}$, $\alpha = 0.5$ |
| **Scorecard scale** | $\text{Score} = \text{Offset} + \text{Factor}\times\ln(\text{odds})$ |
| **Factor / Offset** | $\text{Factor} = PDO/\ln 2$; $\text{Offset} = \text{Base} - \text{Factor}\ln(\text{Base Odds})$ |
| **Attribute points** | $\text{Points}_{ij} = -\text{Factor}\,\beta_j WOE_{ij} + \frac{\text{Offset}-\text{Factor}\,\beta_0}{k}$ |
| **Gini** | $\text{Gini} = 2 \times AUC - 1$ |
| **KS** | $KS = \max_s \lvert F_{\text{bad}}(s) - F_{\text{good}}(s) \rvert$ |
| **Gini/KS bound** | $\text{Gini} \ge KS$ for a concave ROC (equality only for a two-segment ROC) |
| **PSI / CSI** | $\sum_i (\%A_i - \%E_i)\ln(\%A_i/\%E_i)$ |
| **VIF** | $VIF_j = 1/(1 - R_j^2)$ |
| **Brier** | $BS = \frac{1}{N}\sum(\hat p_i - y_i)^2$; $BS = \text{Rel} - \text{Res} + \text{Unc}$ |

## Threshold reference

| Metric | Green | Amber | Red |
|--------|-------|-------|-----|
| **IV** | 0.1 – 0.5 | 0.02 – 0.1 | < 0.02 (drop) or > 0.5 (leakage check) |
| **PSI / CSI** | < 0.10 | 0.10 – 0.25 | > 0.25 |
| **VIF** | < 5 | 5 – 10 | > 10 |
| **AUC (application)** | 0.75+ | 0.70 – 0.75 | < 0.70, or > 0.90 without explanation |
| **AUC (behavioural)** | 0.85+ | 0.80 – 0.85 | > 0.95 without explanation |
| **KS** | 40 – 60 | 20 – 40 | < 20, or > 75 (leakage check) |
| **OOT degradation** | 0.01 – 0.02 | 0.02 – 0.05 | > 0.05, **or exactly 0.00** |
| **Override rate** | < 5% | 5 – 10% | > 10% |
| **Actual / expected bad rate** | 0.9 – 1.1 | 0.8 – 1.2 | outside 0.8 – 1.2 |

## The ten sentences worth memorising

1. **Target definition is the highest-leverage decision** — it sets the performance ceiling, defines what the cutoff means economically, and is nearly impossible to change later.
2. **WOE puts every variable on the log-odds scale by construction**, which is exactly the scale logistic regression assumes — hence the exact $\beta = -1$ univariate unit test.
3. **IV is the Jeffreys divergence** between the good and bad distributions across bins; it is always non-negative and zero only when the variable carries no information.
4. **IV above 0.5 is guilty until proven innocent**, and the sharpest test is whether the IV survives out-of-time — a structural leak keeps its IV, noise collapses.
5. **Monotonicity is inherited by construction in a WOE scorecard and must be imposed and proven in a GBM.** That is the real interpretability argument, not "trees are black boxes."
6. **Gini equals two-AUC-minus-one exactly; KS is a different animal**, and for a concave ROC Gini is an upper bound on KS.
7. **Rank-ordering is robust, calibration is fragile** — so in a turning cycle, recalibrate before you rebuild.
8. **PSI measures input drift, not accuracy.** Always ask "is this a data quality problem?" before "is this drift?", and check whether the model is actually still working before touching it.
9. **Reject inference cannot be validated where it claims to help**, because you have no outcomes in the region it addresses. It buys extrapolation stability and regulatory compliance, not information. Only randomised approval buys information.
10. **The false-positive cost is invisible in the P&L**, which biases every lender toward over-conservatism and starves the next model of training data in exactly the region it needs it.

## Related files

| File | What it adds |
|------|--------------|
| `projects/P01_fibe_behaviour_scorecard.md` | The project narrative: 1,000+ variables, 0.94 AUC, Knime automation, MLflow, PSI/CSI monitoring |
| `projects/P06_axio_address_default_prediction.md` | Adjacent default-prediction work |
| `learning/19_classical_ml_algorithms.md` | Logistic regression, regularisation, tree ensembles, feature selection mechanics |
| `learning/14_evaluation_metrics.md` | ROC/PR curves, threshold selection, imbalance metrics |
| `learning/36_explainability_and_interpretability.md` | SHAP mechanics and the limits of post-hoc attribution |
| `learning/33_causal_inference_and_experimentation.md` | Selection bias, IPW, and the machinery behind reject inference |
| `learning/24_statistics_and_ab_testing.md` | Hypothesis testing, power, and why HL rejects on large samples |

---

*Document prepared for Rahul Sharma — Data Scientist / ML Engineer interview preparation. Anchored on the Fibe (EarlySalary) credit-risk scorecard: CIBIL/Experian/transaction/behavioural data, 1,000+ variables screened via WOE/IV and monotonic binning, 0.94 ROC-AUC out-of-time validated.*
