# 🧠 LLM Hallucination Detection & Mitigation

[![Course](https://img.shields.io/badge/Course-Large%20Language%20Models%20(IIIT%20Dharwad)-orange?style=flat-square)](#-team--course-information)
[![Dataset](https://img.shields.io/badge/Dataset-HaluEval%20QA-green?style=flat-square&logo=huggingface)](https://github.com/RUCAIBox/HaluEval)
[![Generator](https://img.shields.io/badge/Generation%20LLM-openai%2Fgpt--oss--20b%20(Groq)-purple?style=flat-square)](https://groq.com/)
[![NLI Verifier](https://img.shields.io/badge/Verifier%20NLI-DeBERTa--v3--base--mnli--fever--anli-red?style=flat-square&logo=huggingface)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
[![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-blue?style=flat-square)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.14-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Detector%20Built%20%26%20Evaluated%20(Dev)-success?style=flat-square)]()

An empirical research and engineering project studying, detecting, and mitigating hallucinations in Large Language Model (LLM) Question-Answering (QA) systems using the **HaluEval QA** benchmark.

---

## 📌 Project Overview

Large Language Models frequently hallucinate plausible-sounding but factually incorrect information when answering queries exclusively from parametric memory. This project systematically investigates the transition from closed-book generation to Retrieval-Augmented Generation (RAG) and grounded prompting, establishing baseline measurements and error taxonomies before introducing automated verification and self-correction.

### ⚙️ Evaluated Configurations

| Config | 🏷️ Configuration Name | 📝 Description | 🚦 Status |
| --- | --- | --- | :---: |
| **A** | **Closed-Book Baseline** | The LLM answers directly from its internal parametric weights with zero external knowledge. | ✅ **Complete** (`dev`) |
| **B** | **Standard RAG** | Top-$K$ ($K=3$) evidence passages retrieved via dense vector similarity and provided in the prompt context. | ✅ **Complete** (`dev`) |
| **C** | **Grounded RAG + Abstention** | The prompt strictly limits the LLM to supplied evidence, forcing an explicit `"Insufficient evidence"` abstention if ungrounded. | ✅ **Complete** (`dev`) |
| **D** | **RAG + Verification & Mitigation** | Answers are rewritten into atomic declarative claims, verified against evidence using Natural Language Inference (NLI) and novel lexical support scoring, and unsupported claims trigger dynamic re-retrieval / regeneration. | 🔬 **Detector Built & Evaluated** (`dev`); Threshold tuning on `val` & mitigation loop in progress |

---

## 🔬 Key Experimental Results (Dev Split: 100 Questions)

All experiments were conducted with greedy decoding ($T = 0.0$) using `openai/gpt-oss-20b` via the Groq API, embedded with `all-MiniLM-L6-v2`, and verified with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` running locally on CPU.

### 1. 📊 Answer Quality, Accuracy & Hallucination Proxy

*Hallucination Proxy:* **Wrong and answered** represents cases where the model chose not to abstain yet produced a factually incorrect response.

| Configuration | Strict Contains | Lenient Contains | Manually Verified | Abstained | Wrong & Answered (Strict / Lenient / Verified) |
| --- | :---: | :---: | :---: | :---: | :---: |
| **A. Baseline** | 0.31 | 0.31 | **0.33** | 0.01 | 0.68 / 0.68 / **0.66** |
| **B. RAG (Top-3)** | 0.71 | 0.77 | **0.83** | 0.02 | 0.27 / 0.21 / **0.15** |
| **C. Grounded RAG (Top-3)** | 0.72 | 0.76 | **0.79** | 0.17 | 0.11 / 0.07 / **0.04** |

### 2. ⚡ Lexical Overlap, Latency & Token Efficiency

| Configuration | 🎯 Exact Match (EM) | 📈 Token F1 | ⏱️ Avg LLM Latency | ⚡ Avg Retrieval Latency | 🪙 Avg Tokens / Query |
| --- | :---: | :---: | :---: | :---: | :---: |
| **A. Baseline** | 0.250 | 0.301 | 0.73 s | — | 290 |
| **B. RAG (Top-3)** | 0.660 | 0.731 | 0.77 s | ~0.015 s | 433 |
| **C. Grounded RAG (Top-3)** | 0.630 | 0.717 | 0.69 s | ~0.015 s | 434 |

### 3. 🔎 Dense Retrieval Performance (10,040 Chunk Knowledge Index)

Retriever: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, 120-word chunking, exact cosine similarity).

| Metric | @1 | @3 | @5 | @10 | 🎯 MRR@10 |
| --- | :---: | :---: | :---: | :---: | :---: |
| **Recall@K** (Source passage in top $K$) | 0.830 | **0.900** | 0.920 | 0.940 | **0.871** |
| **AnswerInChunk@K** (Gold string in top $K$) | 0.810 | **0.860** | 0.900 | 0.920 | — |

---

### 4. 🛡️ Hallucination Detection Performance (Config D Verification & Detector Evaluation)

Questions and generated answers were transformed into atomic declarative statements using `src/make_claims.py` (381 total dev claims). Each claim was evaluated against retrieved evidence and oracle evidence using `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (`src/nli.py`) alongside novel lexical support scoring (`src/detector_variants.py`).

The hallucination score is defined as $1 - \text{support}$; claims with score $\ge \tau$ (placeholder $\tau = 0.5$) are flagged as hallucinated.

#### 4.1 Evaluation on HaluEval Labeled Pairs (200 Dev Claims: Right vs Hallucinated)

| Scoring Method | 🎯 AUC (Retrieved Ev.) | 🎯 AUC (Oracle Ev.) | Precision | Recall | F1 Score |
| --- | :---: | :---: | :---: | :---: | :---: |
| **NLI Only ($1 - P(\text{entail})$)** | 0.732 | 0.792 | 0.664 | **0.730** | 0.695 |
| **Lexical Support Only** | 0.724 | 0.755 | 0.713 | 0.670 | 0.691 |
| **$\max(\text{NLI}, \text{Lexical})$** | 0.759 | 0.807 | **0.771** | 0.540 | 0.635 |
| **$\text{mean}(\text{NLI}, \text{Lexical})$** | **0.771** | **0.830** | 0.742 | 0.690 | **0.715** |

*Note on Label Noise:* Two pairs had identical right and hallucinated claims in HaluEval. Excluding them improves NLI-only AUC to **0.735** (retrieved) and **0.796** (oracle).

**Confusion Matrices on Pairs (Retrieved Evidence, $\tau = 0.5$):**

| Scoring Variant | Right Claims Kept (True Neg) | Right Claims Flagged (False Alarm) | Hallucinated Caught (True Pos) | Hallucinated Missed (False Neg) |
| --- | :---: | :---: | :---: | :---: |
| **NLI Only** | 63 | 37 | **73** | 27 |
| **$\max(\text{NLI}, \text{Lexical})$** | **84** | **16** | 54 | 46 |
| **$\text{mean}(\text{NLI}, \text{Lexical})$** | 76 | 24 | 69 | 31 |

#### 4.2 Detector on Pipeline's Own Answers (Flagging Incorrect Answers under Lenient Metric)

| Pipeline Configuration | Scoring Method | 🎯 AUC (Retrieved Ev.) | Precision | Recall | F1 Score |
| --- | --- | :---: | :---: | :---: | :---: |
| **Config B** (98 answered, 21 wrong) | NLI Only | 0.678 | 0.333 | 0.571 | 0.421 |
| | Lexical Support Only | 0.724 | 0.444 | **0.571** | 0.500 |
| | $\max(\text{NLI}, \text{Lexical})$ | **0.745** | **0.625** | 0.476 | **0.541** |
| | $\text{mean}(\text{NLI}, \text{Lexical})$ | **0.745** | 0.480 | **0.571** | 0.522 |
| **Config C** (83 answered, 7 wrong) | NLI Only | 0.582 | 0.125 | 0.429 | 0.194 |
| | Lexical Support Only | **0.746** | 0.278 | **0.714** | **0.400** |
| | $\max(\text{NLI}, \text{Lexical})$ | 0.733 | **0.375** | 0.429 | **0.400** |
| | $\text{mean}(\text{NLI}, \text{Lexical})$ | 0.731 | 0.286 | 0.571 | 0.381 |

#### 4.3 Entailment Probability Distributions

| Group / Source | Mean $P(\text{entailment})$ [Retrieved Ev.] | Mean $P(\text{entailment})$ [Oracle Ev.] |
| --- | :---: | :---: |
| **HaluEval Right Claims** | **0.635** | **0.747** |
| **HaluEval Hallucinated Claims** | 0.277 | 0.282 |
| **Pipeline Generated Correct Answers** | **0.710** | **0.791** |
| **Pipeline Generated Incorrect Answers** | 0.501 | 0.614 |

---

## 💡 Key Insights & Error Taxonomy

Detailed analysis from [`results/report.md`](results/report.md), [`results/error_analysis_dev.csv`](results/error_analysis_dev.csv), and [`src/detector_analysis.py`](src/detector_analysis.py) revealed key empirical findings:

1. 🚀 **Retrieval Drastically Suppresses Hallucinations:**
   - Introducing top-3 retrieved passages boosted accuracy from **33% to 83%** (manually verified) and slashed hallucinated/wrong answers from **66% down to 15%**.
2. 🛡️ **Grounded Prompts Prevent Confabulation on Retrieval Failures:**
   - In 10 of the 100 dev questions, dense retrieval failed to place the source passage in the top 3.
   - Config B guessed blindly on 8 of these 10 (hallucinating).
   - **Config C abstained on 100% (10 out of 10) of retrieval failures.**
   - Overall, Config C traded off 4 points of accuracy (83% $\to$ 79%) to drastically reduce fabricated answers from **15% down to 4%**.
3. 🔍 **Decomposition of Config B's 15 Remaining Errors:**
   - **8 cases (53%):** Retrieval failures where the model guessed blindly.
   - **3 cases (20%):** Evidence was present but misread/misattributed.
   - **2 cases (13%):** Incomplete answers (e.g., year provided instead of full date).
   - **1 case (7%):** Yes/No question answered inversely.
   - **1 case (7%):** Malformed question in dataset.
4. 📏 **Multi-Tier Evaluation & Metric Under-Counting:**
   - Strict string matching falsely penalized valid outputs due to Unicode hyphens (e.g., non-breaking hyphens), number-word variations ("8" vs "eight"), and legitimate shorter forms ("Bangladesh" for "People's Republic of Bangladesh").
   - Three-level evaluation framework:
     - **Strict:** Substring containment after punctuation/stopword removal.
     - **Lenient:** Token-set inclusion handling Unicode normalization and number-word conversions.
     - **Manually Verified:** Adjudication of short-form variants and edge cases.
5. ⚠️ **Detector False Alarms & The Missing Entity Problem:**
   - Using NLI alone flagged 37 of 100 correct claims as hallucinations.
   - **Root Cause:** 6 of the 8 worst false alarms occurred because HaluEval Wikipedia passages start mid-sentence or omit entity names, relying on pronouns (*"It was founded in 1996 by..."*, *"its long-time leader..."*, *"He is the chairman..."*). The claim explicitly names the entity, but the premise contains only an unresolved pronoun, causing the NLI model to predict `neutral`.
   - **Solution:** Formulating a **Lexical Support** metric (checking whether novel content words in the claim appear in the evidence). Blending NLI with lexical support via $\text{mean}$ cut false alarms from 37 down to 24; $\max$ reduced them to 16.
6. 🔎 **Detector Misses & Benchmark Label Noise:**
   - An audit of the top 8 "missed" hallucinations (hallucinated claims with $\approx 1.0$ entailment) revealed that **5 out of 8 were actually factually true real-world statements** (e.g., McClellan AFB is in California; Newark connects *Street Fight* and Cory Booker; Milton Friedman is American).
   - 2 were true statements answering a related aspect of the topic.
   - This proves that apparent detector misses are often driven by HaluEval label noise rather than verifier failure.
7. ⚖️ **Faithfulness vs Gold Correctness:**
   - The detector measures faithfulness to provided evidence rather than strict agreement with a gold string. Incomplete answers (e.g., a year instead of full date) are fully supported by evidence and are rightly recognized as faithful.

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
                            │                                       │
                            └───────────────────┬───────────────────┘
                                                │ Non-abstained Answers
                                                ▼
                                  ┌───────────────────────────┐
                                  │   src/make_claims.py      │
                                  │ (Atomic Claim Rewriting)  │
                                  └─────────────┬─────────────┘
                                                │ Claim: Short Declarative Fact
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Dual Verification Gate  │
                                  └──────┬─────────────┬──────┘
                                         │             │
                    Premise vs Hypothesis│             │ Novel Content Tokens
                                         ▼             ▼
                      ┌──────────────────────┐    ┌──────────────────────┐
                      │      DeBERTa-v3      │    │    Lexical Support   │
                      │       NLI Model      │    │    Token Overlap     │
                      └──────────┬───────────┘    └──────────┬───────────┘
                                 │                           │
                                 └─────────────┬─────────────┘
                                               ▼
                                  ┌───────────────────────────┐
                                  │   Hybrid Scoring Engine   │
                                  │   mean / max(NLI, Lexical)│
                                  └────────────┬──────────────┘
                                               │
                                  ┌────────────┴────────────┐
                                  ▼                         ▼
                        Score < τ (Supported)     Score ≥ τ (Hallucination)
                                  │                         │
                                  ▼                         ▼
                             Pass Answer          [Config D Mitigation Loop]
                                                  🔄 Re-retrieval (K=3 -> 5)
                                                  🔄 Prompt Regeneration
```

---

## 🛠️ Engineering Challenges & Practical Solutions

| 🚧 Challenge | 🔍 Root Cause | 💡 Engineering Solution |
| --- | --- | --- |
| 🛑 **API Quota Exhaustion** | Gemini free tier limited to 5 req/min and 20 req/day. | Switched to Groq API (`openai/gpt-oss-20b`), built persistent SHA-256 disk cache (`data/llm_cache.jsonl`), and rate throttlers. |
| ⏳ **Reasoning Token Starvation** | Internal thinking tokens consumed the entire `max_tokens` allocation. | Lowered reasoning effort to `low`, added auto-retry handler that doubles `max_tokens` up to 2048. |
| 💥 **Apple Silicon FAISS Segfault** | PyTorch + FAISS conflict with Python 3.14 on macOS ARM64. | Implemented vectorized exact cosine similarity using NumPy (`q @ emb.T`), yielding identical results for 10k vectors in milliseconds. |
| 🚦 **Rate Limit Violations (HTTP 429)** | Groq 8,000 tokens/minute ceiling. | Built dynamic sliding-window rate limiter (2.2s delay + 6,500 TPM window) with exponential backoff. |
| ⚖️ **Evaluation Metric Bias** | Strict substring matching marked legitimate variations as wrong. | Built multi-tier metrics (`metrics.py`, `rescore.py`) with lenient normalization, number-word mapping, and short-form review. |
| 🧩 **NLI Coreference Breakdown** | Evidence passages omit entity names or use pronouns, triggering false NLI neutrality. | Engineered a novel **Lexical Support** metric (`src/detector_variants.py`) checking novel claim content words against evidence, cutting false alarms by up to 57%. |
| 🏷️ **Benchmark Label Noise** | Benchmark hallucinated answers often contained true facts. | Built diagnostic scripts (`detector_analysis.py`) to systematically audit and isolate noise from true algorithmic failure. |

---

## 📂 Project Structure

```
.
├── README.md                  # Comprehensive project documentation & experimental report
├── app.py                     # Streamlit web demonstration interface (in progress)
├── .env.example               # Environment configuration template
├── data/
│   ├── qa_data.json           # Raw HaluEval QA dataset (10,000 rows)
│   ├── all.jsonl              # Cleaned dataset with permanent IDs
│   ├── dev.jsonl              # Development split (100 questions)
│   ├── val.jsonl              # Validation split (300 questions, reserved for threshold tuning)
│   ├── test.jsonl             # Test split (500 questions, reserved for final evaluation)
│   ├── chunks_120.jsonl       # 10,040 knowledge passages (120-word chunks)
│   ├── emb_120.npy            # Precomputed 384-dimensional embeddings (float32, 10040x384)
│   ├── llm_cache.jsonl        # Persistent SHA-256 prompt response cache
│   ├── baseline_dev.jsonl     # Predictions for Config A (Closed-book baseline)
│   ├── rag_b_k3_dev.jsonl     # Predictions for Config B (Standard RAG)
│   ├── rag_c_k3_dev.jsonl     # Predictions for Config C (Grounded RAG)
│   ├── claims_dev.jsonl       # 381 atomic declarative claims extracted via LLM
│   └── verified_dev.jsonl     # Claims annotated with NLI entailment, neutral, and contra probs
├── results/
│   ├── report.md              # Milestone progress report & complete research paper
│   ├── retrieval_dev_120.csv  # Per-question retrieval ranks and top-1 chunks
│   ├── error_analysis_dev.csv # Cross-configuration categorical error matrix
│   └── rescored_dev.csv       # Re-scored predictions across strict, lenient, and verified tiers
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
    ├── run_baseline.py        # Evaluates Config A (Closed-book baseline)
    ├── run_rag.py             # Evaluates Config B and Config C (RAG generation)
    ├── compare.py             # Prints consolidated multi-configuration metric tables
    ├── error_analysis.py      # Generates categorical outcome and transition matrices
    ├── review_errors.py       # Terminal tool for inspecting baseline failures
    ├── rescore.py             # Re-evaluates outputs with lenient & short-form metrics
    ├── make_claims.py         # Generates atomic declarative claims for NLI checking
    ├── nli.py                 # DeBERTa-v3 NLI inference engine (CPU, batched)
    ├── verify.py              # Evaluates claims against retrieved & oracle evidence
    ├── detect_eval.py         # Evaluates detector AUC, accuracy, precision, recall, and F1
    ├── detector_analysis.py   # Diagnoses detector false alarms, misses, and label noise
    └── detector_variants.py   # Hybrid detector combining NLI with lexical support
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

## 🏃 Reproduction & Execution Workflow

Follow these steps to reproduce the experiments from scratch:

### 🔹 Step 1: Data Preparation & Splitting

Downloads HaluEval QA, deduplicates rows, assigns persistent IDs, and produces `dev` (100), `val` (300), and `test` (500) splits:

```bash
python src/make_splits.py
```

### 🔹 Step 2: Build the Knowledge Base Index

Chunks all 10,000 knowledge rows into 120-word windows with 20% overlap (10,040 chunks) and precomputes 384-d normalized embeddings:

```bash
python src/build_index.py 120
```

### 🔹 Step 3: Evaluate Retrieval Quality

Measures Recall@1, 3, 5, 10, AnswerInChunk@K, and MRR@10:

```bash
python src/eval_retrieval.py dev 120
```

Inspect sample retrieval queries interactively:

```bash
python src/show_retrieval.py
```

### 🔹 Step 4: Run Config A (Closed-Book Baseline)

Executes zero-context QA generation on the dev split:

```bash
python src/run_baseline.py dev
```

### 🔹 Step 5: Run Config B & Config C (RAG Generation)

Run standard RAG (Config B) with top-3 retrieved passages:

```bash
python src/run_rag.py dev b 3
```

Run grounded RAG with explicit abstention (Config C):

```bash
python src/run_rag.py dev c 3
```

### 🔹 Step 6: Multi-Configuration Comparison

Displays a consolidated metrics table across Baseline, Config B, and Config C:

```bash
python src/compare.py
```

### 🔹 Step 7: Error Analysis & Category Tracking

Performs cross-model failure classification and saves the full transition matrix:

```bash
python src/error_analysis.py
```

### 🔹 Step 8: Multi-Tier Re-scoring

Applies lenient normalization, checks for short-form abbreviations, and exports `results/rescored_dev.csv`:

```bash
python src/rescore.py
```

### 🔹 Step 9: Atomic Claim Generation

Rewrites each question and answer into a short, self-contained declarative sentence:

```bash
python src/make_claims.py dev
```

### 🔹 Step 10: NLI Model Verification

Runs DeBERTa-v3 NLI on all claims against retrieved and oracle evidence:

```bash
python src/verify.py dev
```

### 🔹 Step 11: Evaluate Hallucination Detector

Computes ROC-AUC, accuracy, precision, recall, and confusion matrices at $\tau = 0.5$:

```bash
python src/detect_eval.py dev 0.5
```

### 🔹 Step 12: In-Depth Error & False Alarm Analysis

Diagnoses score distributions, examines right claims flagged (false alarms), and inspects missed hallucinations:

```bash
python src/detector_analysis.py
```

### 🔹 Step 13: Evaluate Hybrid Detector Variants (Lexical + NLI)

Evaluates NLI-only, lexical-only, $\max(\text{NLI}, \text{lexical})$, and $\text{mean}(\text{NLI}, \text{lexical})$ combinations:

```bash
python src/detector_variants.py
```

---

## 🗺️ Roadmap & Current Status

- [x] **Config A (Closed-Book Baseline):** Evaluated on dev split (33% accuracy, 66% wrong & answered).
- [x] **Config B (Standard RAG):** Evaluated with Top-3 retrieval (83% accuracy, 15% wrong & answered).
- [x] **Config C (Grounded RAG + Abstention):** Evaluated with Top-3 retrieval (79% accuracy, 4% wrong & answered, 100% abstention on retrieval failures).
- [x] **Multi-Tier Metrics & Error Taxonomy:** Developed strict, lenient, and manually verified scoring tiers.
- [x] **Atomic Claim Extraction:** Automated question-answer declarative rewriting via LLM (`src/make_claims.py`).
- [x] **NLI Verification Engine:** Local deployment of `DeBERTa-v3-base-mnli-fever-anli` (`src/nli.py`, `src/verify.py`).
- [x] **Hybrid Verification Scoring:** Developed Lexical Support metric to mitigate pronoun/coreference false alarms (`src/detector_variants.py`).
- [x] **Detector Evaluation on Dev:** Benchmarked across 200 labeled pairs and 181 pipeline-generated answers.
- [ ] **Validation Split Threshold Tuning:** Tune optimal threshold $\tau$ and select between $\text{mean}$ vs $\max$ using `data/val.jsonl` (300 questions).
- [ ] **Config D Automated Mitigation Loop:** When a claim is flagged, trigger automated dynamic re-retrieval ($K=3 \to 5$) and prompt regeneration.
- [ ] **HaluEval Label Noise Audit:** Manually annotate the 27 missed hallucinated claims to quantify benchmark label error.
- [ ] **Controlled Benchmark Experiments (E1 to E6):** Run full evaluation sweep on untouched `test.jsonl` across chunk sizes (40, 80, 120 words) and retrieval depths ($K \in \{1, 3, 5\}$).
- [ ] **Streamlit Demo Application (`app.py`):** Interactive web dashboard for live QA, evidence inspection, claim decomposition, and real-time hallucination scoring.

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
