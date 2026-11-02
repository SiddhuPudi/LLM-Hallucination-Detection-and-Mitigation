# 🧠 LLM Hallucination Detection & Mitigation

[![Dataset](https://img.shields.io/badge/Dataset-HaluEval%20QA-green?style=flat-square&logo=huggingface)](https://github.com/RUCAIBox/HaluEval)
[![Model](https://img.shields.io/badge/Model-openai%2Fgpt--oss--20b%20(Groq)-purple?style=flat-square)](https://groq.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-blue?style=flat-square)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Development%20Phase-yellow?style=flat-square)]()

An empirical research and engineering project studying, detecting, and mitigating hallucinations in Large Language Model (LLM) Question-Answering (QA) systems using the **HaluEval QA** benchmark.

---

## 📌 Project Overview

Large Language Models frequently hallucinate plausible-sounding but factually incorrect information when answering queries exclusively from parametric memory. This project systematically investigates the transition from closed-book generation to retrieval-augmented generation (RAG) and grounded prompting, establishing baseline measurements and error taxonomies before introducing automated verification and self-correction.

### ⚙️ Evaluated Configurations

| Config | 🏷️ Configuration Name | 📝 Description | 🚦 Status |
| --- | --- | --- | :---: |
| **A** | **Closed-Book Baseline** | The LLM answers directly from its internal parametric weights with zero external knowledge. | ✅ **Complete** (`dev`) |
| **B** | **Standard RAG** | Top-$K$ ($K=3$) evidence passages retrieved via dense vector similarity and provided in the prompt context. | ✅ **Complete** (`dev`) |
| **C** | **Grounded RAG + Abstention** | The prompt strictly limits the LLM to supplied evidence, forcing an explicit `"Insufficient evidence"` abstention if ungrounded. | ✅ **Complete** (`dev`) |
| **D** | **RAG + Verification & Mitigation** | Answers are decomposed into atomic claims, verified against evidence using Natural Language Inference (NLI), and unsupported claims trigger dynamic re-retrieval / regeneration. | ⏳ *In Progress* |

---

## 🔬 Key Experimental Results (Dev Split: 100 Questions)

All experiments were conducted with a greedy decoding temperature ($T = 0.0$) using `openai/gpt-oss-20b` via the Groq API.

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

## 💡 Key Insights & Error Taxonomy

Detailed analysis from [`results/report.md`](results/report.md) and [`results/error_analysis_dev.csv`](results/error_analysis_dev.csv) revealed several critical findings:

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
4. 📏 **Scoring Metric Under-Counting & Multi-Level Evaluation:**
   - Strict string matching falsely penalized valid outputs due to Unicode hyphens (e.g. non-breaking hyphens), number-word variations ("8" vs "eight"), and legitimate shorter forms ("Bangladesh" for "People's Republic of Bangladesh").
   - A three-level evaluation framework was established:
     - **Strict:** Direct substring containment after punctuation/stopword removal.
     - **Lenient:** Token-set inclusion handling Unicode normalization and number-word conversions.
     - **Manually Verified:** Human adjudication of short-form variants and edge cases.

---

## 🏗️ System Architecture & Engineering Solutions

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
                │                       │ Top-3 Chunks
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
                    ▼                                       ▼
       [In Progress: make_claims.py]            [In Progress: NLI Verification]
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                         Config D: Mitigation Loop 🔄
```

### 🛠️ Engineering Challenges Solved

| 🚧 Challenge | 🔍 Root Cause | 💡 Engineering Solution |
| --- | --- | --- |
| 🛑 **API Quota Exhaustion** | Gemini free tier limited to 5 req/min and 20 req/day. | Switched to Groq API (`openai/gpt-oss-20b`), implemented request/token throttling. |
| ⏳ **Reasoning Model Token Starvation** | Thinking tokens consumed the full `max_tokens` allocation. | Reduced reasoning effort to `low`, added auto-retry that doubles `max_tokens` up to 2048. |
| 💥 **Apple Silicon FAISS Segfault** | PyTorch + FAISS conflict with Python 3.14 on macOS ARM64. | Implemented vectorized exact cosine similarity using NumPy (`q @ emb.T`), identical for 10k vectors. |
| 🚦 **Rate Limit Violations (HTTP 429)** | Groq 8,000 tokens/minute ceiling. | Built dynamic sliding-window rate limiter (2.2s gap + 6,500 TPM window) with exponential backoff. |
| ⚖️ **Evaluation Bias** | Strict substring matching marked legitimate answers as wrong. | Built multi-tier metrics (`metrics.py`) with lenient normalization and short-form detection. |

---

## 📂 Project Structure

```
.
├── README.md                  # Project documentation & benchmark analysis
├── app.py                     # Streamlit web interface (in development)
├── .env.example               # Environment configuration template
├── data/
│   ├── qa_data.json           # Raw HaluEval QA dataset (10,000 rows)
│   ├── all.jsonl              # Cleaned dataset with permanent IDs
│   ├── dev.jsonl              # Dev split (100 questions)
│   ├── val.jsonl              # Validation split (300 questions, reserved)
│   ├── test.jsonl             # Test split (500 questions, reserved)
│   ├── chunks_120.jsonl       # 10,040 knowledge passages (120-word chunks)
│   ├── emb_120.npy            # Precomputed 384-dimensional embeddings (float32)
│   ├── llm_cache.jsonl        # Persistent SHA-256 prompt response cache
│   ├── baseline_dev.jsonl     # Output predictions for Config A
│   ├── rag_b_k3_dev.jsonl     # Output predictions for Config B
│   ├── rag_c_k3_dev.jsonl     # Output predictions for Config C
│   └── claims_dev.jsonl       # Atomic declarative claims for NLI verification
├── results/
│   ├── report.md              # Full academic milestone report & analysis
│   ├── retrieval_dev_120.csv  # Per-question retrieval ranks and top-1 chunks
│   ├── error_analysis_dev.csv # Cross-configuration categorical error matrix
│   └── rescored_dev.csv       # Re-scored predictions across metric tiers
└── src/
    ├── load_data.py           # Initial dataset inspection script
    ├── make_splits.py         # Dataset cleaning, deduplication, and split generator
    ├── build_index.py         # Text chunking and SentenceTransformer embedder
    ├── retrieve.py            # Vector retrieval engine (NumPy cosine / FAISS)
    ├── eval_retrieval.py      # Recall@K, AnswerInChunk@K, and MRR evaluation
    ├── show_retrieval.py      # Visual inspector for top-K retrieved evidence
    ├── llm.py                # Groq API client with caching, retry, and throttling
    ├── test_llm.py            # Smoke test script for Groq API
    ├── metrics.py             # Multi-tier evaluation metrics and token normalizers
    ├── run_baseline.py        # Evaluates Config A (Closed-book baseline)
    ├── run_rag.py             # Evaluates Config B and Config C
    ├── compare.py             # Aggregates and prints side-by-side comparison tables
    ├── error_analysis.py      # Generates categorical outcome and transition matrices
    ├── review_errors.py       # Terminal tool for inspecting baseline failures
    ├── rescore.py             # Re-evaluates outputs with lenient & short-form metrics
    └── make_claims.py         # Generates atomic declarative claims for NLI checking
```

---

## 🚀 Getting Started

### 1. 📋 Prerequisites

- Python 3.10+ (tested on Python 3.12 and Python 3.14 on macOS Apple Silicon / Linux)
- A free [Groq Cloud API Key](https://console.groq.com/)

### 2. 💻 Installation & Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/SiddhuPudi/LLM-Hallucination-Detection-and-Mitigation
cd "LLM Hallucination Detection & Mitigation"

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or: pip install pandas sentence-transformers requests python-dotenv tqdm
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
```

Verify your API connectivity:

```bash
python src/test_llm.py
```

---

## 🏃 Reproduction & Execution Workflow

Follow these steps to reproduce the experiments from scratch:

### 🔹 Step 1: Data Preparation & Splitting

Downloads HaluEval QA, deduplicates questions, assigns persistent IDs, and produces `dev` (100), `val` (300), and `test` (500) splits:

```bash
python src/make_splits.py
```

### 🔹 Step 2: Build the Evidence Corpus Index

Chunks all 10,000 knowledge rows into 120-word windows with 20% overlap, generating 10,040 passage chunks and precomputing 384-d normalized embeddings:

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

Run grounded RAG with abstention (Config C):

```bash
python src/run_rag.py dev c 3
```

### 🔹 Step 6: Multi-Configuration Comparison

Displays a consolidated metrics table across Baseline, Config B, and Config C:

```bash
python src/compare.py
```

### 🔹 Step 7: Error Analysis & Category Tracking

Performs cross-model failure classification and saves the full transition matrix to `results/error_analysis_dev.csv`:

```bash
python src/error_analysis.py
```

### 🔹 Step 8: Multi-Tier Re-scoring

Applies lenient normalization, checks for short-form abbreviations, and exports `results/rescored_dev.csv`:

```bash
python src/rescore.py
```

### 🔹 Step 9: Atomic Claim Generation (Detection Prep)

Generates atomic declarative claims for right answers, known hallucinations, Config B, and Config C answers for downstream NLI verification:

```bash
python src/make_claims.py dev
```

---

## 🗺️ Roadmap & Next Steps

1. 🛡️ **Config D: Automated Hallucination Verification & Mitigation:**
   - [x] Atomic claim extraction (`src/make_claims.py`).
   - [ ] Local NLI model deployment (`deberta-v3-small` / `roberta-large-snli`) to classify entailment vs. neutral/contradiction.
   - [ ] Automated self-correction loop: if a claim is unsupported, automatically expand retrieval ($K=3 \to 5$) or trigger prompt refinement and re-generation.
2. 🧪 **Controlled Evaluation across Benchmarks:**
   - Scale experiments across the untouched `val` (300) and `test` (500) splits.
   - Sweep chunk window sizes (40, 80, 120 words) and retrieval depths ($K \in \{1, 3, 5\}$).
3. 🖥️ **Interactive Demonstration:**
   - Complete `app.py` as an interactive Streamlit application enabling live QA, side-by-side evidence inspection, and claim verification visualization.

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

Built using [Sentence-Transformers](https://www.sbert.net/), [Groq Cloud](https://groq.com/), and [Hugging Face](https://huggingface.co/).

---

## 👥 Team Members

- **Thrivikram** (23BCS100)
- **Kuladeep** (23BDS021)
- **Harshith** (23BDS058)
- **Sadan** (23BDS048)
- **Chaitra** (23BDS016)
