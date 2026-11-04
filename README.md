# 🧠 LLM Hallucination Detection & Mitigation

[![Dataset](https://img.shields.io/badge/Dataset-HaluEval%20QA-green?style=flat-square&logo=huggingface)](https://github.com/RUCAIBox/HaluEval)
[![Generator](https://img.shields.io/badge/Generation%20LLM-openai%2Fgpt--oss--20b%20(Groq)-purple?style=flat-square)](https://groq.com/)
[![NLI Verifier](https://img.shields.io/badge/Verifier%20NLI-DeBERTa--v3--base--mnli--fever--anli-red?style=flat-square&logo=huggingface)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
[![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-blue?style=flat-square)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.14-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Mitigation%20Loop%20Complete%20(Dev%20%26%20Val)-success?style=flat-square)]()

An empirical research and engineering project studying, detecting, and mitigating hallucinations in Large Language Model (LLM) Question-Answering (QA) systems using the **HaluEval QA** benchmark.

---

## 📌 Project Overview

Large Language Models frequently hallucinate plausible-sounding but factually incorrect information when answering queries exclusively from parametric memory. This project systematically investigates the transition from closed-book generation to Retrieval-Augmented Generation (RAG) and grounded prompting, establishing baseline measurements and error taxonomies before introducing automated claim verification and a multi-round dynamic self-correction loop.

### ⚙️ Evaluated Configurations

| Config | 🏷️ Configuration Name | 📝 Description | 🚦 Status |
| --- | --- | --- | :---: |
| **A** | **Closed-Book Baseline** | The LLM answers directly from its internal parametric weights with zero external knowledge. | ✅ **Complete** (`dev`) |
| **B** | **Standard RAG** | Top-$K$ ($K=3$) evidence passages retrieved via dense vector similarity and provided in the prompt context. | ✅ **Complete** (`dev`, `val`) |
| **C** | **Grounded RAG + Abstention** | The prompt strictly limits the LLM to supplied evidence, forcing an explicit `"Insufficient evidence"` abstention if ungrounded. | ✅ **Complete** (`dev`, `val`) |
| **D** | **RAG + Verification & Mitigation** | Starts from Config C. Answers that C abstained on or that the detector flags as unsupported trigger multi-round dynamic re-retrieval ($K=5, 10$) and regeneration, verified with NLI + lexical scoring. | ✅ **Complete** (`dev`, `val`); Threshold tuned on `val` (`results/chosen_detector.json`) |

---

## 🔬 Experimental Results & Benchmark Analysis

All experiments were conducted with greedy decoding ($T = 0.0$) using `openai/gpt-oss-20b` via the Groq API, embedded with `all-MiniLM-L6-v2`, and verified with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` running locally on CPU. Evaluated on the 100-question `dev` split and confirmed on the 100-question `val` subset.

### 1. 📊 Answer Quality, Accuracy & Hallucination Proxy

*Hallucination Proxy:* **Wrong and answered** represents cases where the model chose not to abstain yet produced a factually incorrect response.

| Split | Configuration | Strict Contains | Lenient Contains | Manually Verified | Abstained | Wrong & Answered (Strict / Lenient / Verified) |
| :---: | --- | :---: | :---: | :---: | :---: | :---: |
| **`dev`** | **A. Baseline** | 0.31 | 0.31 | **0.33** | 0.01 | 0.68 / 0.68 / **0.66** |
| (100 Qs) | **B. RAG (Top-3)** | 0.71 | 0.77 | **0.83** | 0.02 | 0.27 / 0.21 / **0.15** |
| | **C. Grounded RAG (Top-3)** | 0.72 | 0.76 | **0.79** | 0.17 | 0.11 / 0.07 / **0.04** |
| | **D. Mitigation (Keep)** | **0.78** | **0.82** | **0.85** | 0.11 | 0.11 / 0.07 / **0.04** |
| | **D. Mitigation (Abstain)** | 0.71 | 0.75 | 0.78 | 0.20 | 0.09 / 0.05 / **0.02** |
| **`val`** | **B. RAG (Top-3)** | 0.79 | 0.81 | **0.87** | 0.01 | 0.20 / 0.18 / **0.12** |
| (100 Qs) | **C. Grounded RAG (Top-3)** | 0.75 | 0.77 | **0.83** | 0.12 | 0.13 / 0.11 / **0.05** |
| | **D. Mitigation (Keep)** | **0.80** | **0.82** | **0.88** | 0.09 | 0.11 / 0.09 / **0.03** |
| | **D. Mitigation (Abstain)** | 0.76 | 0.78 | 0.83 | 0.14 | 0.10 / 0.08 / **0.03** |

### 2. ⚡ Lexical Overlap, Latency & Token Efficiency

| Configuration | 🎯 Exact Match (EM) | 📈 Token F1 | ⏱️ Avg LLM Latency | ⚡ Avg Retrieval Latency | 🪙 Avg Tokens / Query |
| --- | :---: | :---: | :---: | :---: | :---: |
| **A. Baseline (`dev`)** | 0.250 | 0.301 | 0.73 s | — | 290 |
| **B. RAG (Top-3, `dev`)** | 0.660 | 0.731 | 0.77 s | ~0.015 s | 433 |
| **C. Grounded RAG (Top-3, `dev`)** | 0.630 | 0.717 | 0.69 s | ~0.015 s | 434 |
| **B. RAG (Top-3, `val`)** | 0.710 | 0.809 | 0.73 s | ~0.015 s | 425 |
| **C. Grounded RAG (Top-3, `val`)** | 0.670 | 0.769 | 0.71 s | ~0.015 s | 437 |

### 3. 🔎 Dense Retrieval Performance (10,040 Chunk Knowledge Index)

Retriever: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, 120-word chunking, exact cosine similarity).

| Metric | @1 | @3 | @5 | @10 | 🎯 MRR@10 |
| --- | :---: | :---: | :---: | :---: | :---: |
| **Recall@K** (Source passage in top $K$) | 0.830 | **0.900** | 0.920 | 0.940 | **0.871** |
| **AnswerInChunk@K** (Gold string in top $K$) | 0.810 | **0.860** | 0.900 | 0.920 | — |

---

### 4. 🛡️ Hallucination Detector & Validation-Driven Threshold Selection

Questions and answers are rewritten into atomic declarative claims via LLM (`src/make_claims.py`). Each claim is evaluated against retrieved evidence using `DeBERTa-v3-base-mnli-fever-anli` NLI entailment and novel **Lexical Support** scoring (`src/detector.py`).

The hallucination score is $1 - \text{support}$; a claim is flagged as hallucinated if score $\ge \tau$.

#### 4.1 Detector Selection Protocol (Frozen Before Running Validation)

- **Selection Criterion:** Maximize $F_{0.5}$ on pooled Config B and C answers from the validation split (187 answers, 29 wrong under lenient metric). $F_{0.5}$ prioritizes precision to minimize costly false-alarm regenerations.
- **Candidate Variants:** `mean(NLI, lexical)`, `max(NLI, lexical)`, and `NLI only` across $\tau \in [0.30, 0.80]$ with step 0.05.

**Validation Tuning Results (`val` split, 187 pipeline answers & 200 HaluEval pairs):**

| Variant | AUC (Pipeline) | AUC (Pairs) | Best $\tau$ | Flagged (TP / FP) | Precision | Recall | $F_1$ | 🎯 $F_{0.5}$ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mean(NLI, lexical)`** | **0.752** | **0.779** | **0.55** | 24 (13 / 11) | **0.542** | 0.448 | **0.491** | **0.520** |
| **`max(NLI, lexical)`** | 0.679 | 0.777 | 0.30 | 24 (13 / 11) | 0.542 | 0.448 | 0.491 | 0.520 |
| **`NLI only` (reference)** | 0.691 | 0.712 | 0.80 | 46 (16 / 30) | 0.348 | **0.552** | 0.427 | 0.376 |

**Chosen Configuration (`results/chosen_detector.json`):**
$$\text{Variant: } \text{mean}(\text{NLI}, \text{lexical}), \quad \text{Threshold } \tau = 0.55$$

#### 4.2 Threshold Sensitivity & Corrected Labels Check (`src/check_detector.py`)

12 of the 29 "wrong" validation answers were short-form variants (e.g. "Mickey Arthur" for "coached by Mickey Arthur") that are factually correct. Repeating tuning with human-verified labels confirmed:

- The optimal threshold remains **$\tau = 0.55$** ($F_{0.5} = 0.487$).
- Under corrected labels, detector AUC reaches **0.832** on `val` and **0.837** on `dev`, catching 65% of true errors with a precision 5 times the base error rate.

---

### 5. 🔄 Config D: Mitigation Loop & Self-Correction Performance

The mitigation loop ([`src/mitigate.py`](file:///Users/thrivikrampudi/Downloads/Projects/LLM%20Hallucination%20Detection%20%26%20Mitigation/src/mitigate.py)) starts from Config C answers. If Config C abstains or the frozen detector ($\text{mean}, \tau = 0.55$) flags the answer:

1. **Retry Round 1:** Retrieve top-5 chunks ($K=5$), regenerate with grounded prompt, extract claim, re-verify. If unflagged $\to$ **Accept**.
2. **Retry Round 2:** If still unverified/abstained, retrieve top-10 chunks ($K=10$), regenerate, extract claim, re-verify. If unflagged $\to$ **Accept**.
3. **Fallback Policy:** If retries remain flagged, **Keep** the best-supported candidate (marked unverified) or **Abstain** ("Insufficient evidence").

#### 5.1 Mitigation Outcomes Across 200 Pooled Questions (`dev` + `val`)

```
Config C Answers (200 Questions)
 ├── Not Triggered (Answered & Verified): 154 questions
 │    └── 150 Correct (97.4%), 4 Wrong (2.6%)
 ├── Triggered by Abstention ("Insufficient evidence"): 29 questions
 │    ├── Retries Accepted & Correct: 6 questions (e.g., "15th", "Loud", "model", "Picket Fences")
 │    ├── Fallback Kept: 3 questions (2 Correct, 1 Incomplete)
 │    └── Remained Abstained: 20 questions
 └── Triggered by Detector Flag: 17 questions (12 Correct, 5 Wrong)
      ├── Retries Accepted: 6 questions (3 Wrong answers FIXED, 2 Correct stayed Correct, 1 stayed Wrong)
      └── Fallback Kept: 11 questions (10 Correct preserved, 1 Wrong)
```

#### 5.2 Key Mitigation Findings

- 🏆 **Pre-Set Success Criterion Met:** Config D (Keep) significantly outperformed Config C on verified accuracy (from 162 to **173 out of 200**, $+11$) while reducing wrong answers (from 9 down to **7 out of 200**).
- 🛡️ **Zero Regression:** **0 out of 162** answers that Config C got correct were made worse by the mitigation loop.
- ⚡ **Matches RAG Accuracy with $4\times$ Fewer Hallucinations:** Config D (Keep) matched or exceeded plain RAG (Config B) accuracy (0.85 vs 0.82 on `dev`; 0.88 vs 0.87 on `val`) while cutting hallucinated/wrong answers by **$\approx 75\%$** (0.04 vs 0.16 on `dev`; 0.03 vs 0.12 on `val`).
- 🔄 **Multi-Round Synergy:** Round 1 ($K=5$) yielded 8 accepted retries (7 correct); Round 2 ($K=10$) yielded 4 accepted retries (**4 out of 4 correct**). Two rounds proved strictly superior to a single round (173 vs 168 correct).
- ⚖️ **Fallback Strategy:** The "Keep Best-Supported" policy yielded 173 correct and 7 wrong answers, whereas "Abstain if Flagged" discarded 12 correct answers just to eliminate 2 errors (161 correct, 5 wrong).

---

## 💡 Key Insights & Error Taxonomy

Detailed analysis from [`results/report.md`](results/report.md), [`src/detector_analysis.py`](src/detector_analysis.py), and [`src/mitigation_analysis.py`](src/mitigation_analysis.py) revealed:

1. 🚀 **Retrieval Drastically Suppresses Hallucinations:**
   - Top-3 dense retrieval increased accuracy from **33% to 83%** (manually verified) and slashed hallucinated answers from **66% down to 15%**.
2. 🛡️ **Grounded Prompts Prevent Confabulation on Retrieval Failures:**
   - When retrieval failed (10/100 questions), Config B hallucinated blindly on 8. Config C abstained on **100% (10/10)** of retrieval failures, cutting fabricated responses to 4%.
3. ⚠️ **Detector False Alarms & The Missing Entity Problem:**
   - Raw NLI produced 37 false alarms on correct claims because Wikipedia passages frequently omit the entity name in subsequent sentences or start mid-sentence (*"It was founded in 1996 by..."*, *"its long-time leader..."*). The claim explicitly names the entity, but NLI cannot resolve the missing antecedent.
   - The **Lexical Support** metric (novel claim content words matching evidence) successfully resolved this, cutting false alarms down to 24 ($\text{mean}$) and 16 ($\max$).
4. 🔎 **Detector Misses & Benchmark Label Noise:**
   - Auditing the top 8 "missed" hallucinations revealed that **5 out of 8 were actually true real-world facts** (e.g., McClellan AFB is in California; Newark connects *Street Fight* and Cory Booker; Milton Friedman is American), indicating substantial label noise in HaluEval.
5. 🔍 **Taxonomy of Remaining Errors Post-Mitigation (7 Total across Dev + Val):**
   - **Synonyms (2):** Valid alternative terms unrecognised by string metrics ("United States" for "America", "Torah" for "Law of Moses").
   - **Incomplete Answers (2):** Partially specified entities supported by evidence ("Film director." for "film director and screenwriter").
   - **Non-Responsive Statements (2):** Answering "Yes." to statement-style questions ending in a question mark.
   - **True Non-Answering Statement (1):** Stating a true fact that fails to answer the question.
   - *Crucially, none of the remaining post-mitigation errors were fabricated hallucinated entities.*

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

## 🛠️ Engineering Challenges & Practical Solutions

| # | 🚧 Challenge | 🔍 Root Cause | 💡 Engineering Solution |
| :---: | --- | --- | --- |
| 1 | 🛑 **API Quota Exhaustion** | Gemini free tier limited to 5 req/min and 20 req/day. | Switched to Groq API (`openai/gpt-oss-20b`), built persistent SHA-256 disk cache (`data/llm_cache.jsonl`), and sliding-window rate limiters. |
| 2 | ⏳ **Reasoning Token Starvation** | Thinking tokens consumed the full `max_tokens` allocation. | Lowered reasoning effort to `low`, added auto-retry handler that doubles `max_tokens` up to 2048. |
| 3 | 💥 **Apple Silicon FAISS Segfault** | PyTorch + FAISS conflict with Python 3.14 on macOS ARM64. | Implemented vectorized exact cosine similarity using NumPy (`q @ emb.T`), yielding identical results in milliseconds. |
| 4 | 🚦 **Rate Limit Violations (HTTP 429)** | Groq 8,000 tokens/minute ceiling. | Built dynamic sliding-window rate limiter (2.2s delay + 6,500 TPM window) with exponential backoff. |
| 5 | ⚖️ **Evaluation Metric Bias** | Strict substring matching marked legitimate variations as wrong. | Built multi-tier metrics (`metrics.py`, `rescore.py`) with lenient normalization, number-word mapping, and short-form review. |
| 6 | 🧩 **NLI Coreference Breakdown** | Passages omit entity names or use pronouns, triggering false NLI neutrality. | Engineered a novel **Lexical Support** metric (`src/detector.py`) checking novel claim content words against evidence, cutting false alarms by up to 57%. |
| 7 | 🏷️ **Benchmark Label Noise** | Benchmark hallucinated answers often contained true facts. | Built diagnostic scripts (`src/detector_analysis.py`) to systematically audit and isolate noise from true algorithmic failure. |
| 8 | 🎯 **Threshold Tuning Uncertainty** | Validation sample had only 29 wrong answers; 12 were short-form artifacts. | Fixed selection rule in advance ($F_{0.5}$ on pooled answers), confirmed stability with sensitivity check (`src/check_detector.py`) locking $\tau = 0.55$. |
| 9 | 🔄 **Mitigation Fallback Policy** | Naive rejection of flagged answers discards valid answers due to false alarms. | Implemented "Keep Best-Supported" candidate policy, preventing loss of correct answers while keeping hallucinations suppressed. |

---

## 📂 Project Structure

```
.
├── README.md                  # Comprehensive project documentation & experimental report
├── app.py                     # Streamlit web demonstration interface (in development)
├── .env.example               # Environment configuration template
├── data/
│   ├── qa_data.json           # Raw HaluEval QA dataset (10,000 rows)
│   ├── all.jsonl              # Cleaned dataset with permanent IDs
│   ├── dev.jsonl              # Development split (100 questions)
│   ├── val.jsonl              # Validation split (300 questions; first 100 used for tuning)
│   ├── test.jsonl             # Test split (500 questions, reserved for final evaluation)
│   ├── chunks_120.jsonl       # 10,040 knowledge passages (120-word chunks)
│   ├── emb_120.npy            # Precomputed 384-dimensional embeddings (float32, 10040x384)
│   ├── llm_cache.jsonl        # Persistent SHA-256 prompt response cache
│   ├── baseline_dev.jsonl     # Predictions for Config A (Closed-book baseline, dev)
│   ├── rag_b_k3_dev.jsonl     # Predictions for Config B (Standard RAG, dev)
│   ├── rag_c_k3_dev.jsonl     # Predictions for Config C (Grounded RAG, dev)
│   ├── rag_b_k3_val.jsonl     # Predictions for Config B (Standard RAG, val)
│   ├── rag_c_k3_val.jsonl     # Predictions for Config C (Grounded RAG, val)
│   ├── claims_dev.jsonl       # 381 atomic declarative claims extracted via LLM (dev)
│   ├── claims_val.jsonl       # 387 atomic declarative claims extracted via LLM (val)
│   ├── verified_dev.jsonl     # Dev claims annotated with NLI entail/neutral/contra probs
│   ├── verified_val.jsonl     # Val claims annotated with NLI entail/neutral/contra probs
│   ├── mitigate_dev.jsonl     # Full Config D mitigation execution trace on dev
│   └── mitigate_val.jsonl     # Full Config D mitigation execution trace on val
├── results/
│   ├── report.md              # Milestone progress report & complete research paper
│   ├── chosen_detector.json   # Frozen detector specification (mean variant, threshold 0.55)
│   ├── retrieval_dev_120.csv  # Per-question retrieval ranks and top-1 chunks
│   ├── error_analysis_dev.csv # Cross-configuration categorical error matrix
│   ├── rescored_dev.csv       # Re-scored dev predictions across strict, lenient, verified
│   └── rescored_val.csv       # Re-scored val predictions across strict, lenient, verified
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
    ├── mitigate.py            # Config D multi-round dynamic mitigation and self-correction loop
    └── mitigation_analysis.py # Parses mitigation execution traces by round, trigger, and status
```

---

## 🚀 Getting Started

### 1. 📋 Prerequisites

- Python 3.10+ (tested on Python 3.12 and Python 3.14 on macOS Apple Silicon & Linux)
- A free [Groq Cloud API Key](https://console.groq.com/)

### 2. 💻 Installation & Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/SiddhuPudi/LLM-Hallucination-Detection-and-Mitigation
cd LLM-Hallucination-Detection-and-Mitigation

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Core dependencies: pandas sentence-transformers transformers torch scikit-learn python-dotenv tqdm
```

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```ini
LLM_API="your_groq_api_key_here"
LLM_MODEL="openai/gpt-oss-20b"
LLM_REASONING="low"
USE_FAISS="0"
NLI_MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
```

Verify your API connectivity:

```bash
python src/test_llm.py
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

### 🔹 Stage 2: Baseline & Initial RAG Evaluation (Dev Split)

```bash
# 4. Run closed-book baseline (Config A)
python src/run_baseline.py dev

# 5. Run standard RAG (Config B, top-3) and grounded RAG (Config C, top-3)
python src/run_rag.py dev b 3
python src/run_rag.py dev c 3

# 6. Compare performance & rescore with lenient/short-form metrics
python src/compare.py
python src/rescore.py dev
```

### 🔹 Stage 3: Claim Extraction & Hallucination Verification (Dev Split)

```bash
# 7. Extract atomic declarative claims from QA pairs
python src/make_claims.py dev

# 8. Run DeBERTa-v3 NLI verification against retrieved & oracle evidence
python src/verify.py dev

# 9. Evaluate initial detector variants and analyze false alarms
python src/detect_eval.py dev 0.5
python src/detector_analysis.py
```

### 🔹 Stage 4: Validation Runs & Threshold Tuning (Val Split)

```bash
# 10. Run Config B and Config C on validation subset
python src/run_rag.py val b 3
python src/run_rag.py val c 3
python src/rescore.py val

# 11. Extract claims and verify with NLI on validation subset
python src/make_claims.py val
python src/verify.py val

# 12. Grid-search threshold tuning selecting maximum F0.5 on validation answers
python src/tune_threshold.py val

# 13. Perform sensitivity check with corrected labels and cross-split validation
python src/check_detector.py val
python src/check_detector.py dev
```

### 🔹 Stage 5: Config D Mitigation Loop & Self-Correction

```bash
# 14. Execute the multi-round mitigation loop on dev and val
python src/mitigate.py dev
python src/mitigate.py val

# 15. Run comprehensive mitigation analysis (trigger breakdown, retry rounds, error audit)
python src/mitigation_analysis.py dev
python src/mitigation_analysis.py val
```

---

## 🗺️ Roadmap & Current Status

- [x] **Config A (Closed-Book Baseline):** Evaluated on dev split (33% accuracy, 66% wrong & answered).
- [x] **Config B (Standard RAG):** Evaluated with Top-3 retrieval on dev (83%) and val (87%).
- [x] **Config C (Grounded RAG + Abstention):** Evaluated on dev (79%) and val (83%), abstaining on 100% of retrieval failures.
- [x] **Multi-Tier Metrics & Error Taxonomy:** Developed strict, lenient, and manually verified scoring tiers.
- [x] **Atomic Claim Extraction:** Automated declarative rewriting via LLM (`src/make_claims.py`).
- [x] **NLI Verification Engine:** Local deployment of `DeBERTa-v3-base-mnli-fever-anli` (`src/nli.py`, `src/verify.py`).
- [x] **Hybrid Verification Scoring:** Developed Lexical Support metric to mitigate pronoun/coreference false alarms (`src/detector.py`).
- [x] **Validation Threshold Tuning:** Rigorously selected $\text{mean}$ variant and $\tau = 0.55$ maximizing $F_{0.5}$ on validation (`results/chosen_detector.json`).
- [x] **Config D Mitigation Pipeline:** Built and verified multi-round self-correction loop on dev and val, recovering 11 correct answers with 0 regressions.
- [ ] **Test-Scale Final Evaluation:** Run frozen pipeline (A, B, C, D) once on untouched `test.jsonl` (200 questions).
- [ ] **Controlled Ablation Experiments (E2 & E3):** Sweep retrieval depth ($K \in \{1, 3, 5\}$) and small chunk sizes (25, 40, 60 words).
- [ ] **HaluEval Label Noise Audit:** Manually annotate missed hallucinated claims to quantify true benchmark error.
- [ ] **Interactive Streamlit Application (`app.py`):** User dashboard for live QA, evidence highlighting, claim decomposition, and real-time mitigation visualization.

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
