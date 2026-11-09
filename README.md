# 🧠 LLM Hallucination Detection & Mitigation

[![Dataset](https://img.shields.io/badge/Dataset-HaluEval%20QA-green?style=flat-square&logo=huggingface)](https://github.com/RUCAIBox/HaluEval)
[![Generator](https://img.shields.io/badge/Generation%20LLM-openai%2Fgpt--oss--20b%20(Groq)-purple?style=flat-square)](https://groq.com/)
[![NLI Verifier](https://img.shields.io/badge/Verifier%20NLI-DeBERTa--v3--base--mnli--fever--anli-red?style=flat-square&logo=huggingface)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
[![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-blue?style=flat-square)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.14-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Final%20Test%20Evaluation%20Complete-success?style=flat-square)]()

An empirical research and engineering project studying, detecting, and mitigating hallucinations in Large Language Model (LLM) Question-Answering (QA) systems using the **HaluEval QA** benchmark across 400 evaluated questions (`dev`, `val`, and `test` splits).

---

## 📌 Project Overview

Large Language Models frequently hallucinate plausible-sounding but factually incorrect information when answering queries exclusively from parametric memory. This project systematically investigates the transition from closed-book generation to Retrieval-Augmented Generation (RAG) and grounded prompting, establishing baseline measurements and error taxonomies before introducing automated claim verification and a multi-round dynamic self-correction loop.

### ⚙️ Evaluated Configurations

| Config | 🏷️ Configuration Name | 📝 Description | 🚦 Status |
| :---: | --- | --- | :---: |
| **A** | **Closed-Book Baseline** | The LLM answers directly from its internal parametric weights with zero external knowledge. | ✅ **Complete** (`dev`, `test`) |
| **B** | **Standard RAG** | Top-$K$ ($K=3$) evidence passages retrieved via dense vector similarity and provided in the prompt context. | ✅ **Complete** (`dev`, `val`, `test`) |
| **C** | **Grounded RAG + Abstention** | The prompt strictly limits the LLM to supplied evidence, forcing an explicit `"Insufficient evidence"` abstention if ungrounded. | ✅ **Complete** (`dev`, `val`, `test`) |
| **D** | **RAG + Verification & Mitigation** | Starts from Config C. Answers that C abstained on or that the detector flags as unsupported trigger multi-round dynamic re-retrieval ($K=5, 10$) and regeneration, verified with NLI + lexical scoring. | ✅ **Complete** (`dev`, `val`, `test`); Tuned on `val`, evaluated on `test` |

---

## 🔬 Headline Findings (Final Evaluation on 200 Test Questions)

1. **Retrieval Provides Massive Gains Over Closed-Book Baseline:**
   - Closed-book baseline (**Config A**) answered only **35%** correctly, hallucinating on **65%** of queries.
   - Plain RAG (**Config B**) boosted verified accuracy to **79%** ($+43.5\%$, 95% CI: $[+36.0\%, +51.5\%]$), but still produced confident hallucinations on **21%** of questions.
2. **Grounded Prompting Slashes Hallucinations with Negligible Accuracy Loss:**
   - Grounded RAG with abstention (**Config C**) cut wrong answers by **14.5 percentage points** (down to **7%**, 95% CI: $[4\%, 10\%]$) by abstaining on ungrounded queries (17%), at a statistically insignificant accuracy cost ($-3.0\%$, 95% CI: $[-8.0\%, +2.0\%]$).
3. **Dynamic Mitigation Loop Recovers Abstentions Without Corrupting Grounded Answers:**
   - The mitigation loop (**Config D, Keep policy**) recovered valid answers from abstentions, achieving **80% verified accuracy** ($+4.0\%$ over Config C, 95% CI: $[+1.5\%, +7.0\%]$) and reducing abstentions to **12%**.
   - **Zero Regression on Grounded Answers:** None of the 151 answers that Config C answered correctly were corrupted by the loop.
   - Across 400 pooled questions (`dev`, `val`, `test`), Config D recovered **19 net correct answers** while keeping the hallucination rate identical to Config C.
4. **Pre-Set Criteria Verdict:**
   - **Secondary Criterion MET:** Config D dominated plain RAG, matching accuracy (0.795 vs 0.785) with **62% fewer wrong answers** (0.080 vs 0.215, 95% CI diff: $[-19.0\%, -8.5\%]$).
   - **Primary Criterion NOT MET:** The loop produced 16 wrong answers on test vs Config C's 14 (pre-set rule required no increase in errors), introducing 2 erroneous accepted retries.
5. **Mitigation Strongly Beats Verification-Only Baselines:**
   - Retrying with expanded evidence ($K=5, 10$) proved vastly superior to passive rejection ("verify-and-abstain"), which discarded 7.5% of correct answers in RAG.

---

## 📊 Comprehensive Experimental Results

All experiments were conducted with greedy decoding ($T = 0.0$) using `openai/gpt-oss-20b` via the Groq API, embedded with `all-MiniLM-L6-v2` (384d), and verified with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` running locally on CPU.

### 1. 📈 Answer Quality Across All Three Splits (`dev`, `val`, `test`: 400 Questions Total)

*Hallucination Proxy:* **Wrong and answered** represents cases where the model chose not to abstain yet produced an incorrect response.

| Split | Configuration | Strict Contains | Lenient Contains | Verified Correct | Abstained | Wrong & Answered (Lenient) | Wrong & Answered (Verified) |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dev`** | **A. Baseline** | 0.31 | 0.31 | **0.33** | 0.01 | 0.68 | **0.66** |
| (100 Qs) | **B. Plain RAG (Top-3)** | 0.71 | 0.77 | **0.83** | 0.02 | 0.21 | **0.15** |
| | **C. Grounded RAG (Top-3)** | 0.72 | 0.76 | **0.79** | 0.17 | 0.07 | **0.04** |
| | **D. Loop (Keep Best)** | **0.78** | **0.82** | **0.85** | 0.11 | 0.07 | **0.04** |
| | **D. Loop (Abstain if Flagged)** | 0.71 | 0.75 | 0.78 | 0.20 | 0.05 | **0.02** |
| **`val`** | **B. Plain RAG (Top-3)** | 0.79 | 0.81 | **0.87** | 0.01 | 0.18 | **0.12** |
| (100 Qs) | **C. Grounded RAG (Top-3)** | 0.75 | 0.77 | **0.83** | 0.12 | 0.11 | **0.05** |
| | **D. Loop (Keep Best)** | **0.80** | **0.82** | **0.88** | 0.09 | 0.09 | **0.03** |
| | **D. Loop (Abstain if Flagged)** | 0.76 | 0.78 | 0.83 | 0.14 | 0.08 | **0.03** |
| **`test`** | **A. Baseline** | 0.30 | 0.31 | **0.35** | 0.01 | 0.69 | **0.65** |
| (200 Qs) | **B. Plain RAG (Top-3)** | 0.71 | 0.73 | **0.79** | 0.00 | 0.27 | **0.21** |
| | **C. Grounded RAG (Top-3)** | 0.69 | 0.71 | **0.76** | 0.17 | 0.12 | **0.07** |
| | **D. Loop (Keep Best)** | **0.71** | **0.74** | **0.80** | 0.12 | 0.13 | **0.08** |
| | **D. Loop (Abstain if Flagged)** | 0.68 | 0.71 | 0.76 | 0.17 | 0.12 | **0.07** |

---

### 2. 🔬 Statistical Rigor & Paired Differences (Test Split: 200 Questions, 5,000 Bootstrap Resamples)

Evaluated using 95% percentile bootstrap intervals with [`src/final_eval.py`](file:///Users/thrivikrampudi/Downloads/Projects/LLM%20Hallucination%20Detection%20%26%20Mitigation/src/final_eval.py):

| Paired Difference ($X - Y$) | $\Delta$ Verified Correct [95% CI] | $\Delta$ Wrong-and-Answered [95% CI] | Statistical Interpretation |
| --- | :---: | :---: | --- |
| **D Keep vs C Grounded** | **+0.040** $[+0.015, +0.070]$ | **+0.010** $[0.000, +0.025]$ | Significant accuracy boost from abstention recovery; small non-significant error drift. |
| **D Keep vs B Plain RAG** | **+0.010** $[-0.035, +0.055]$ | **-0.135** $[-0.190, -0.085]$ | Matches RAG accuracy while **slashing errors by 13.5 percentage points** ($p < 0.001$). |
| **C Grounded vs B Plain RAG** | **-0.030** $[-0.080, +0.020]$ | **-0.145** $[-0.200, -0.095]$ | Grounding eliminates 14.5 points of error with no statistically significant loss of accuracy. |
| **B Plain RAG vs A Baseline** | **+0.435** $[+0.360, +0.515]$ | **-0.430** $[-0.510, -0.355]$ | Retrieval provides enormous +43.5% net accuracy and eliminates 43% of confabulations. |
| **D Keep vs A Baseline** | **+0.445** $[+0.370, +0.520]$ | **-0.565** $[-0.635, -0.495]$ | Overall pipeline beats baseline by +44.5% accuracy and **reduces error by 56.5 points**. |

---

### 3. ⚡ Lexical Overlap, Latency & Token Efficiency

| Configuration | Split | Exact Match (EM) | Token F1 | Avg LLM Latency | Avg Retrieval Latency | Avg Tokens / Query |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Baseline** | `dev` | 0.250 | 0.301 | 0.73 s | — | 290 |
| **B. Plain RAG (Top-3)** | `dev` | 0.660 | 0.731 | 0.77 s | ~0.015 s | 433 |
| **C. Grounded RAG (Top-3)** | `dev` | 0.630 | 0.717 | 0.69 s | ~0.015 s | 434 |
| **A. Baseline** | `test` | 0.260 | 0.355 | 0.98 s | — | 311 |
| **B. Plain RAG (Top-3)** | `test` | 0.630 | 0.739 | 0.87 s | ~0.015 s | 446 |
| **C. Grounded RAG (Top-3)** | `test` | 0.615 | 0.702 | 0.79 s | ~0.015 s | 457 |
| **D. Mitigation Loop** | `test` | — | — | ~3.8 s (triggered) | ~0.030 s | 810 (1.8x Config C) |

---

### 4. 🔎 Dense Retrieval Performance (10,040 Chunk Knowledge Index)

Retriever: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, 120-word chunking, exact cosine similarity).

| Metric | @1 | @3 | @5 | @10 | 🎯 MRR@10 |
| --- | :---: | :---: | :---: | :---: | :---: |
| **Recall@K** (Source passage in top $K$) | 0.830 | **0.900** | 0.920 | 0.940 | **0.871** |
| **AnswerInChunk@K** (Gold string in top $K$) | 0.810 | **0.860** | 0.900 | 0.920 | — |

---

### 5. 🛡️ Verification Engine & Threshold Tuning (`results/chosen_detector.json`)

Questions and answers are rewritten into atomic declarative claims via LLM ([`src/make_claims.py`](file:///Users/thrivikrampudi/Downloads/Projects/LLM%20Hallucination%20Detection%20%26%20Mitigation/src/make_claims.py)). The detector evaluates claims against retrieved evidence using `DeBERTa-v3-base-mnli-fever-anli` NLI entailment and novel **Lexical Support** scoring ([`src/detector.py`](file:///Users/thrivikrampudi/Downloads/Projects/LLM%20Hallucination%20Detection%20%26%20Mitigation/src/detector.py)).

The hallucination score is $1 - \text{support}$; flagged if score $\ge \tau$.

#### 5.1 Validation-Driven Selection Protocol

Selected on 187 pooled Config B and C validation answers by maximizing $F_{0.5}$ (prioritizing precision over recall to avoid costly false-alarm regenerations):
$$\text{Chosen Detector: } \text{mean}(\text{NLI}, \text{lexical}), \quad \text{Threshold } \tau = 0.55, \quad F_{0.5} = 0.520$$

#### 5.2 Detector Generalization Across Splits

The classifier's discriminative ability generalized remarkably well across independent splits:

- **HaluEval Labeled Pairs AUC:** **0.771** (`dev`, 200 claims) $\to$ **0.779** (`val`, 200 claims) $\to$ **0.766** (`test`, 400 claims).
- **Asymmetric Filtering Utility:**
  - On **Plain RAG answers (Config B)**: The detector acts as an effective filter (precision 0.52, recall 0.48 pooled), catching ungrounded guesses when retrieval fails.
  - On **Grounded answers (Config C)**: Precision is low (0.22 pooled, 0.10 test) because grounded errors are mostly supported by evidence (synonyms, formatting) rather than ungrounded fabrications.

---

### 6. ⚔️ Mitigation vs Verification-Only Baselines (Experiments E5 & E6)

Computed offline via [`src/verify_only.py`](file:///Users/thrivikrampudi/Downloads/Projects/LLM%20Hallucination%20Detection%20%26%20Mitigation/src/verify_only.py) to isolate the value of active mitigation (retrieval expansion + regeneration) versus passive abstention ("verify and abstain"):

| Configuration | Test Correct | Test Wrong | Test Abstained | Pooled (400) Correct | Pooled Wrong | Pooled Abstained |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B. Plain RAG** | 0.785 | 0.215 | 0.000 | 0.815 | 0.177 | 0.007 |
| **B + Verify (Passive Abstain)** | 0.710 | 0.140 | 0.150 | 0.735 | 0.092 | 0.172 |
| **C. Grounded RAG** | 0.755 | 0.070 | 0.175 | 0.782 | 0.058 | 0.160 |
| **C + Verify (Passive Abstain)** | 0.710 | 0.065 | 0.225 | 0.730 | 0.043 | 0.228 |
| **D. Mitigation Loop (Keep)** | **0.795** | **0.080** | **0.125** | **0.830** | **0.058** | **0.113** |
| **D. Mitigation Loop (Abstain)** | 0.755 | 0.075 | 0.170 | 0.780 | 0.050 | 0.170 |

**Key Takeaways:**

- **Passive verification hurts accuracy:** "B + Verify" discarded 7.5% of correct answers to eliminate 7.5% of errors. On Config C, passive verification lost 4.5% accuracy to remove only 0.5% errors.
- **Active mitigation dominates:** Config D (Keep) achieved **8.5 percentage points higher accuracy** than verification-only ($p < 0.001$), proving that dynamic retrieval expansion is essential.

---

## 💡 Key Insights & In-Depth Error Taxonomy

### 1. 🔄 Where Does the Mitigation Loop Succeed?

Across 400 pooled questions, the loop was triggered for 81 questions (64 abstentions, 17 flagged answers):

- **Abstention Recovery (+19 Net Correct Answers):** 16 of 64 abstentions were converted into verified-correct answers by retrieving 5 or 10 chunks (e.g. Mincome prime minister $\to$ *"15th"*, Rihanna album $\to$ *"Loud"*, occupation $\to$ *"model"*).
- **Zero Regression:** In all 400 questions, not a single answer that Config C answered correctly was degraded.
- **Limitation on Wrong Answers:** The loop fixed 3 wrong answers but introduced 3 new ones (net 0 change in errors).

### 2. 🔍 Post-Hoc Audit of Remaining Test Errors (16 Wrong Answers in Config D Keep)

Detailed manual inspection ([`results/report.md`](file:///Users/thrivikrampudi/Downloads/Projects/LLM%20Hallucination%20Detection%20%26%20Mitigation/results/report.md#appendix-d-post-hoc-reading-of-the-16-wrong-answers-after-the-loop-test-keep-policy)) revealed:

- **10 Benign Metric Artifacts (62.5%):**
  - *Synonyms / Aliases:* *"United States"* for *"American"*, *"Torah"* for *"Law of Moses"*, *"Arctic Sea"* for *"Arctic Ocean"*.
  - *Granularity / Paraphrase:* *"Washington, D.C."* for *"Capitol Hill"*, *"Director of Mad Max 2"* for *"Created by director George Miller"*.
  - *Formatting / Ordinals:* *"14th"* for *"14"*, *"Jim Diamond"* for *"James Aaron Diamond"*.
- **6 Genuine Errors (37.5%):**
  - 4 from original generation (e.g. *"Pilot"* for astronaut).
  - 2 introduced by mitigation retries (*"Matt Maiellaro"* for Dave Willis; *"Las Vegas"* for Paradise casino).

### 3. 🧩 The Missing Entity Phenomenon in NLI

Raw DeBERTa NLI generated 37 false alarms on dev because Wikipedia passages omit entity names or start mid-sentence with pronouns (*"It was founded in 1996 by..."*, *"its long-time leader..."*). The claim explicitly names the entity, breaking coreference for NLI.
The novel **Lexical Support** metric ([`src/detector.py`](file:///Users/thrivikrampudi/Downloads/Projects/LLM%20Hallucination%20Detection%20%26%20Mitigation/src/detector.py)) resolved this by checking novel content word overlap, cutting false alarms by over 50%.

---

## 🏗️ End-to-End System Architecture

```
                                  ┌───────────────────────────┐
                                  │ 📚 HaluEval QA (10k rows) │
                                  └─────────────┬─────────────┘
                                                │
                                 Word Chunking (120 words, 20% overlap)
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │    10,040 Chunks Index    │
                                  │   all-MiniLM-L6-v2 (384d) │
                                  └─────────────┬─────────────┘
                                                │
                   User Query                   │
                        │                       ▼
                        ├───────────►  Exact Cosine Retrieval (NumPy)
                        │                       │ Top-3 Evidence Chunks
                        │                       ▼
                        │             ┌───────────────────┐
                        └────────────►│ Prompt Formatting │
                                      └─────────┬─────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │      Groq API ⚡       │
                                    │ (openai/gpt-oss-20b)  │
                                    └───────────┬───────────┘
                                                │
                            ┌───────────────────┴───────────────────┐
                            ▼                                       ▼
                Config B: Direct Answer                 Config C: Grounded + Abstain
                                                                    │
                            ┌───────────────────────────────────────┘
                            │
                            ├── Abstained ("Insufficient evidence") ──────┐
                            ▼                                             │
                   Answered Candidates                                    │
                            │                                             │
                            ▼                                             │
               ┌───────────────────────────┐                              │
               │   src/make_claims.py      │                              │
               │ (Atomic Claim Rewriting)  │                              │
               └────────────┬──────────────┘                              │
                            │ Declarative Fact                            │
                            ▼                                             │
               ┌───────────────────────────┐                              │
               │   Dual Verification Gate  │                              │
               │   DeBERTa-v3 NLI + Lexical│                              │
               │   mean(NLI, Lexical)      │                              │
               └────────────┬──────────────┘                              │
                            │                                             │
               ┌────────────┴────────────┐                                │
               ▼                         ▼                                │
     Score < 0.55 (Verified)   Score ≥ 0.55 (Flagged)                     │
               │                         │                                │
               ▼                         └────────────────┐               │
          Accept Answer                                   ▼               ▼
                                            ┌───────────────────────────────────┐
                                            │  Config D: Mitigation Pipeline    │
                                            ├───────────────────────────────────┤
                                            │ 🔄 Round 1: Top-5 Chunks (K=5)    │
                                            │    Regenerate -> Verify Claim     │
                                            │    If Verified -> ACCEPT          │
                                            │ 🔄 Round 2: Top-10 Chunks (K=10)  │
                                            │    Regenerate -> Verify Claim     │
                                            │    If Verified -> ACCEPT          │
                                            │ 🛡️ Fallback: Keep Best Supported  │
                                            └───────────────────────────────────┘
```

---

## 🛠️ Summary of Engineering Challenges Solved

| # | 🚧 Challenge | 🔍 Root Cause | 💡 Engineering Solution |
| :---: | --- | --- | --- |
| 1 | 🛑 **API Quota Limits** | Gemini free tier limited to 20 req/day. | Switched to Groq API (`openai/gpt-oss-20b`), built persistent SHA-256 disk cache (`data/llm_cache.jsonl`). |
| 2 | ⏳ **Reasoning Token Starvation** | Thinking tokens consumed the full `max_tokens` allocation. | Lowered reasoning effort to `low`, added auto-retry handler that doubles `max_tokens` up to 2048. |
| 3 | 💥 **Apple Silicon FAISS Segfault** | PyTorch + FAISS conflict with Python 3.14 on macOS ARM64. | Vectorized exact cosine similarity using NumPy (`q @ emb.T`), identical results in milliseconds. |
| 4 | 🚦 **Rate Limit Violations (HTTP 429)** | Groq 8,000 tokens/minute ceiling. | Built dynamic sliding-window rate limiter (2.2s delay + 6,500 TPM window) with exponential backoff. |
| 5 | ⚖️ **Evaluation Metric Bias** | Strict string matching marked legitimate variations as wrong. | Built multi-tier metrics (`metrics.py`, `rescore.py`) with lenient normalization and short-form review. |
| 6 | 🧩 **NLI Coreference Breakdown** | Passages omit entity names or use pronouns, triggering false NLI neutrality. | Engineered a novel **Lexical Support** metric (`src/detector.py`), cutting false alarms by >50%. |
| 7 | 🏷️ **Benchmark Label Noise** | Benchmark hallucinated answers contained true facts. | Diagnostic audit (`src/detector_analysis.py`) identifying mislabeled ground truth. |
| 8 | 🎯 **Threshold Tuning Uncertainty** | Validation sample had only 29 wrong answers; 12 were short-form artifacts. | Fixed selection rule in advance ($F_{0.5}$ on pooled answers), confirmed stability with sensitivity check (`src/check_detector.py`). |
| 9 | 🔄 **Mitigation Fallback Policy** | Naive rejection of flagged answers discards valid answers. | Implemented "Keep Best-Supported" policy, recovering 19 answers across 400 questions with 0 regressions. |
| 10 | 🧪 **Grounded Error Masking** | Grounded answers errors are supported by retrieved evidence (synonyms, aliases). | Formulated experiments E5/E6 (`src/verify_only.py`) demonstrating active mitigation superiority. |

---

## 📂 Project Structure

```
.
├── README.md                  # Comprehensive project documentation & final evaluation report
├── app.py                     # Streamlit web demonstration interface (in development)
├── .env.example               # Environment configuration template
├── data/
│   ├── qa_data.json           # Raw HaluEval QA dataset (10,000 rows)
│   ├── all.jsonl              # Cleaned dataset with permanent IDs
│   ├── dev.jsonl              # Development split (100 questions)
│   ├── val.jsonl              # Validation split (300 questions; first 100 used for tuning)
│   ├── test.jsonl             # Test split (500 questions; first 200 used for final evaluation)
│   ├── chunks_120.jsonl       # 10,040 knowledge passages (120-word chunks)
│   ├── emb_120.npy            # Precomputed 384-dimensional embeddings (float32, 10040x384)
│   ├── llm_cache.jsonl        # Persistent SHA-256 prompt response cache
│   ├── baseline_dev.jsonl     # Predictions for Config A (dev)
│   ├── baseline_test.jsonl    # Predictions for Config A (test)
│   ├── rag_b_k3_dev.jsonl     # Predictions for Config B (dev)
│   ├── rag_b_k3_val.jsonl     # Predictions for Config B (val)
│   ├── rag_b_k3_test.jsonl    # Predictions for Config B (test)
│   ├── rag_c_k3_dev.jsonl     # Predictions for Config C (dev)
│   ├── rag_c_k3_val.jsonl     # Predictions for Config C (val)
│   ├── rag_c_k3_test.jsonl    # Predictions for Config C (test)
│   ├── claims_dev.jsonl       # Declarative claims extracted via LLM (dev)
│   ├── claims_val.jsonl       # Declarative claims extracted via LLM (val)
│   ├── claims_test.jsonl      # Declarative claims extracted via LLM (test)
│   ├── verified_dev.jsonl     # Dev claims annotated with NLI probabilities
│   ├── verified_val.jsonl     # Val claims annotated with NLI probabilities
│   ├── verified_test.jsonl    # Test claims annotated with NLI probabilities
│   ├── mitigate_dev.jsonl     # Full Config D mitigation execution trace (dev)
│   ├── mitigate_val.jsonl     # Full Config D mitigation execution trace (val)
│   └── mitigate_test.jsonl    # Full Config D mitigation execution trace (test)
├── results/
│   ├── report.md              # Complete final research report
│   ├── chosen_detector.json   # Frozen detector specification (mean variant, threshold 0.55)
│   ├── retrieval_dev_120.csv  # Per-question retrieval ranks and top-1 chunks
│   ├── error_analysis_dev.csv # Cross-configuration categorical error matrix
│   ├── rescored_dev.csv       # Re-scored dev predictions
│   ├── rescored_val.csv       # Re-scored val predictions
│   └── rescored_test.csv      # Re-scored test predictions
└── src/
    ├── load_data.py           # Initial dataset inspection script
    ├── make_splits.py         # Dataset cleaning, deduplication, and split generator
    ├── build_index.py         # Text chunking and SentenceTransformer embedder
    ├── retrieve.py            # Vector retrieval engine (NumPy cosine / FAISS)
    ├── eval_retrieval.py      # Recall@K, AnswerInChunk@K, and MRR evaluation
    ├── show_retrieval.py      # Visual inspector for top-K retrieved evidence
    ├── llm.py                # Groq API client with disk caching, retry, and rate throttling
    ├── test_llm.py            # Smoke test script for Groq API
    ├── metrics.py             # Multi-tier evaluation metrics and token normalizers
    ├── labels.py              # Consolidated evaluation labels (lenient & verified)
    ├── run_baseline.py        # Evaluates Config A (Closed-book baseline)
    ├── run_rag.py             # Evaluates Config B and Config C across any split
    ├── compare.py             # Prints consolidated multi-configuration metric tables
    ├── error_analysis.py      # Generates categorical outcome and transition matrices
    ├── review_errors.py       # Terminal tool for inspecting baseline failures
    ├── rescore.py             # Re-evaluates outputs with lenient & short-form metrics
    ├── make_claims.py         # Generates atomic declarative claims for NLI checking
    ├── nli.py                 # DeBERTa-v3 NLI inference engine (CPU, batched)
    ├── verify.py              # Evaluates claims against retrieved & oracle evidence
    ├── detector.py            # Feature extractor computing NLI, lexical, and hybrid scores
    ├── detect_eval.py         # Evaluates detector AUC, accuracy, precision, recall, and F1
    ├── tune_threshold.py      # Grid-search threshold tuning selecting maximum F0.5
    ├── check_detector.py      # Label sensitivity check and cross-split detector validation
    ├── detector_analysis.py   # Diagnoses detector false alarms, misses, and label noise
    ├── detector_variants.py   # Compares scoring combinations on dev
    ├── mitigate.py            # Config D multi-round dynamic mitigation loop
    ├── mitigation_analysis.py # Parses mitigation execution traces by round and status
    ├── final_eval.py          # Bootstrap confidence intervals and paired difference testing
    └── verify_only.py         # Verification-only offline evaluation (Experiments E5 and E6)
```

---

## 🏃 Complete Reproduction Workflow

Follow these steps to reproduce the experiments from scratch:

### 🔹 Stage 1: Data Preparation & Retrieval Indexing

```bash
# 1. Download HaluEval QA, deduplicate, and split into dev (100), val (300), test (500)
python src/make_splits.py

# 2. Chunk knowledge passages into 120-word windows and build 384-d dense embeddings
python src/build_index.py 120

# 3. Evaluate retrieval performance (Recall@K, AnswerInChunk@K, MRR@10)
python src/eval_retrieval.py dev 120
```

### 🔹 Stage 2: Baseline & RAG Evaluation (Dev Split)

```bash
# 4. Run closed-book baseline (Config A)
python src/run_baseline.py dev

# 5. Run standard RAG (Config B, top-3) and grounded RAG (Config C, top-3)
python src/run_rag.py dev b 3
python src/run_rag.py dev c 3
python src/rescore.py dev
```

### 🔹 Stage 3: Claim Extraction & Hallucination Verification (Dev Split)

```bash
# 6. Extract atomic declarative claims from QA pairs
python src/make_claims.py dev

# 7. Run DeBERTa-v3 NLI verification against retrieved & oracle evidence
python src/verify.py dev

# 8. Evaluate initial detector variants and analyze false alarms
python src/detect_eval.py dev 0.5
python src/detector_analysis.py
```

### 🔹 Stage 4: Validation Runs & Threshold Tuning (Val Split)

```bash
# 9. Run Config B and Config C on validation subset
python src/run_rag.py val b 3
python src/run_rag.py val c 3
python src/rescore.py val

# 10. Extract claims and verify with NLI on validation subset
python src/make_claims.py val
python src/verify.py val

# 11. Grid-search threshold tuning selecting maximum F0.5 on validation answers
python src/tune_threshold.py val

# 12. Perform sensitivity check with corrected labels and cross-split validation
python src/check_detector.py val
python src/check_detector.py dev
```

### 🔹 Stage 5: Final Evaluation & Verification Baselines (Test Split: 200 Questions)

```bash
# 13. Run baseline, Config B, and Config C on test split
python src/run_baseline.py test 200
python src/run_rag.py test b 3 200
python src/run_rag.py test c 3 200
python src/rescore.py test

# 14. Run the mitigation loop on test split
python src/mitigate.py test 200

# 15. Extract claims and verify on test split
python src/make_claims.py test 200
python src/verify.py test

# 16. Run final bootstrap statistical evaluation (95% CIs and paired tests)
python src/final_eval.py test

# 17. Run verification-only offline evaluation (Experiments E5 and E6)
python src/verify_only.py dev val test
```

---

## 🗺️ Roadmap & Current Status

- [x] **Config A (Closed-Book Baseline):** Evaluated on `dev` (33%) and `test` (35%).
- [x] **Config B (Standard RAG):** Evaluated on `dev` (83%), `val` (87%), and `test` (79%).
- [x] **Config C (Grounded RAG + Abstention):** Evaluated on `dev` (79%), `val` (83%), and `test` (76%), slashing hallucinations by 67%.
- [x] **Multi-Tier Metrics & Error Taxonomy:** Developed strict, lenient, and manually verified scoring tiers.
- [x] **Atomic Claim Extraction:** Automated declarative rewriting via LLM (`src/make_claims.py`).
- [x] **NLI Verification Engine:** Local deployment of `DeBERTa-v3-base-mnli-fever-anli` (`src/nli.py`, `src/verify.py`).
- [x] **Hybrid Verification Scoring:** Developed Lexical Support metric to mitigate pronoun/coreference false alarms (`src/detector.py`).
- [x] **Validation Threshold Tuning:** Rigorously selected $\text{mean}$ variant and $\tau = 0.55$ maximizing $F_{0.5}$ on validation (`results/chosen_detector.json`).
- [x] **Config D Mitigation Pipeline:** Built and verified multi-round self-correction loop on `dev`, `val`, and `test`, recovering 19 correct answers across 400 questions with 0 regressions.
- [x] **Final Test-Scale Evaluation (200 Questions):** Completed with bootstrap 95% confidence intervals and paired difference testing (`src/final_eval.py`).
- [x] **Verification-Only Baselines (E5 & E6):** Proved active mitigation superiority over passive rejection (`src/verify_only.py`).
- [ ] **Publication Plots & Figures:** Generate publication-quality figures for paper/slides.
- [ ] **Streamlit Demo Application (`app.py`):** Interactive web dashboard for live QA, evidence inspection, claim decomposition, and real-time mitigation visualization.

---

## 📜 Citation & Acknowledgements

This project utilizes the [HaluEval](https://github.com/RUCAIBox/HaluEval) benchmark:

```bibtex
@inproceedings{li2023halueval,
  title={HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models},
  author={Li, Junyi and Cheng, Xiaoxue and Zhao, Wayne Xin and Nie, Jian-Yun and Wen, Ji-Rong},
  booktitle={Proceedings of EMNLP},
  year={2023}
}
```

Built using [Sentence-Transformers](https://www.sbert.net/), [DeBERTa-v3 (Moritz Laurer)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli), [Groq Cloud](https://groq.com/), and [Hugging Face](https://huggingface.co/).

---

## 👥 Team Members

- **Thrivikram** (23BCS100)
- **Kuladeep** (23BDS021)
- **Harshith** (23BDS058)
- **Sadan** (23BDS048)
- **Chaitra** (23BDS016)
