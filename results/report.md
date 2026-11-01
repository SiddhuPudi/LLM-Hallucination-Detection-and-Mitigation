# LLM Hallucination Detection & Mitigation: Progress Report

**Course:** Large Language Models, IIIT Dharwad
**Team members:** Kuladeep (23BDS021), Thrivikram (23BCS100), Harshith (23BDS058), Sadan (23BDS048)
**Status as of:** 1 November 2026. Baseline and RAG generation stages are complete; hallucination detection and mitigation are not built yet.

---

## 1. Project summary

This project builds a question-answering system on the **HaluEval QA** dataset to study and reduce hallucinations in LLM answers. It compares four configurations:

| Config | Description | Status |
|---|---|---|
| A. Baseline | The LLM answers the question directly, with no external evidence. | Done (dev) |
| B. RAG | The top-3 evidence chunks are retrieved (embeddings + vector search) and given to the LLM. | Done (dev) |
| C. RAG + grounded prompt | The prompt tells the LLM to answer only from the evidence, or abstain. | Done (dev) |
| D. RAG + verification + mitigation | Claims in the answer are checked against the evidence with an NLI model; unsupported answers are regenerated. | Not built yet |

The systems will be compared on hallucination rate, answer accuracy, faithfulness, retrieval performance (Recall@K, MRR) and response latency. The project does not claim to eliminate hallucinations; it measures how much each stage improves reliability.

## 2. Work completed so far

1. **Environment.** Python virtual environment on a MacBook Air (8 GB RAM, no GPU) with the required libraries installed.
2. **Dataset.** HaluEval QA loaded and inspected (fields: `knowledge`, `question`, `right_answer`, `hallucinated_answer`). Cleaned by dropping empty fields and duplicate questions, leaving 10,000 rows.
3. **Fixed splits.** Shuffled once with seed 42 and saved three permanent splits: `dev` (100 questions, for development), `val` (300, for threshold tuning) and `test` (500, for final results). The `val` and `test` splits have not been used yet.
4. **LLM access.** Built `src/llm.py`, a reusable function that calls the LLM API with a disk cache (no prompt is ever paid for twice), throttling by requests and tokens per minute, retry handling for rate-limit and server errors, automatic retry when a reasoning model returns an empty answer, and logging of latency, model version and token usage.
5. **Metrics.** Built `src/metrics.py` (see Section 4.3).
6. **Baseline (Config A).** Ran the LLM on the dev split with no retrieval.
7. **Evidence corpus and retrieval.** Chunked the `knowledge` field of all 10,000 rows, embedded the chunks, built a searchable index and evaluated retrieval quality.
8. **RAG generation (Configs B and C).** Built `src/run_rag.py` and ran both prompt variants on the dev split with top-3 evidence. Built `src/compare.py` to put all configurations in one table.

## 3. LLM used

**Model:** `openai/gpt-oss-20b`, served through the free tier of the Groq API, with reasoning effort set to low. The same model is used for every configuration, so only the prompt and the evidence change between experiments. A small model was chosen deliberately: it makes enough mistakes on closed-book questions for retrieval and verification to show a measurable effect.

## 4. Method summary

### 4.1 Evidence corpus and retrieval

- **Corpus:** the `knowledge` field of all 10,000 cleaned rows (not only the dev, val and test rows), so the correct passage has to compete with many wrong ones.
- **Chunking:** word-based chunks of up to 120 words with a small overlap. The result was 10,040 chunks with an average of 55 words, so nearly every row stayed as a single chunk.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized, runs locally on CPU).
- **Search:** exact cosine similarity over all chunk vectors (numpy). FAISS is supported as an optional switch, but see problem 7 in Section 6.
- **Retrieval depth:** top-3 chunks per question.

### 4.2 Prompts

- **Config B (plain RAG):** the retrieved evidence is shown to the model with the instruction to use it to answer, giving only a short answer.
- **Config C (grounded + abstain):** the model is told to use only the evidence, not to use outside knowledge, and to reply exactly "Insufficient evidence" if the evidence does not contain the answer.

### 4.3 Metrics

- **Exact Match (EM), token F1:** standard answer-matching metrics after normalization.
- **Contains:** the gold answer appears inside the prediction. Added because the model often phrases answers differently from the gold text.
- **Abstained:** the answer says "unknown", "insufficient evidence" or similar.
- **Wrong and answered:** the model gave an answer (did not abstain) and the gold answer is not in it. This is used as a simple **hallucination-rate proxy** for now. Claim-level verification (not built yet) will add an evidence-based measure.
- **Evidence had answer:** the gold answer text appears in the retrieved evidence. This is a ceiling on what a faithful answer can achieve.
- **Retrieval:** Recall@K (a chunk from the question's own source row is in the top K), AnswerInChunk@K (the gold answer text is in the top K chunks) and MRR.
- **Cost:** average tokens per question and average API latency per question.

## 5. Results (dev split, 100 questions)

All numbers below come from our own runs on the 100-question dev split. They are development-stage results, not final results.

### 5.1 Retrieval (120-word chunks, 10,040-chunk index)

| Metric | Value |
|---|---|
| Recall@1 | 0.830 |
| Recall@3 | 0.900 |
| Recall@5 | 0.920 |
| Recall@10 | 0.940 |
| AnswerInChunk@1 / @3 / @5 / @10 | 0.810 / 0.860 / 0.900 / 0.920 |
| MRR@10 | 0.871 |

The correct passage is ranked first for 83% of questions and is in the top 3 for 90%. Top-3 was therefore chosen as the starting evidence size.

### 5.2 Answer quality, hallucination proxy and cost

| Config | Contains | EM | F1 | Abstained | Wrong and answered | Avg tokens | Avg API latency |
|---|---|---|---|---|---|---|---|
| A. Baseline | 0.31 | 0.25 | 0.301 | 0.01 | 0.68 | 290 | 0.73 s |
| B. RAG (top-3) | 0.71 | 0.66 | 0.731 | 0.02 | 0.27 | 433 | 0.77 s |
| C. RAG + grounded prompt (top-3) | 0.72 | 0.63 | 0.717 | 0.17 | 0.11 | 434 | 0.69 s |

Average retrieval time was about 0.01 to 0.02 s per question. In B and C, the retrieved evidence contained the gold answer for 86% of questions, and a chunk from the question's own source row was retrieved for 90%.

### 5.3 What the results show

- **Retrieval helps a lot.** Adding evidence raised the share of answers containing the gold answer from 0.31 to 0.71, and cut confidently wrong answers from 68% of questions to 27%. The size of this gain is far beyond what chance variation on 100 questions could explain.
- **The grounded prompt reduces confident wrong answers further without hurting correctness.** Moving from B to C, correct answers stayed essentially the same (0.71 vs 0.72, a difference of one question, which is within noise), while "wrong and answered" fell from 27% to 11%. The model instead abstained on 17% of questions. This trade-off (fewer fabricated answers in exchange for more "I don't know") is the behaviour we wanted.
- **Cost of RAG.** Prompts are about 50% longer (290 to about 433 tokens per question), while API latency is about the same. Retrieval itself adds only milliseconds.
- **Evidence limits accuracy.** The evidence contained the answer in 86% of cases, yet B was correct in 71%, so some wrong answers happen even when the right evidence was retrieved (the model misread it, or the question needs combining several passages). Separating these cases is part of the next error analysis.

## 6. Problems faced and how they were solved

| # | Problem | Cause | Solution |
|---|---|---|---|
| 1 | The first LLM choice (Gemini 3.8 Flash via Google AI Studio) could not run the baseline. | The free tier allows only 5 requests per minute and 20 requests per day, far below what 100+ questions per experiment need. | Checked the usage dashboard, stopped the run, and switched to a provider with a larger free quota (Groq). |
| 2 | Repeated HTTP 503 errors and 90-second read timeouts during the first baseline attempt. | The Gemini model was overloaded; the retry loop made things worse by spending the small daily quota on failed attempts. | Shortened timeouts, added smarter retry rules, and made the code stop with a clear message when a daily limit is reached. |
| 3 | Running a model locally (Ollama) was not an option. | The laptop has 8 GB of RAM and froze even with the smallest model. | Used a hosted API instead of a local model. |
| 4 | Groq returned a 404 "model not found" error for `llama-3.1-8b-instant`. | That model is not available on this account; the account lists only a few chat models. | Read the account's model list and chose `openai/gpt-oss-20b`. |
| 5 | Some answers came back empty. | `gpt-oss-20b` is a reasoning model; its internal thinking sometimes used up the whole token limit before producing an answer. | Set reasoning effort to low, and made the code retry automatically with a larger token limit. Empty answers are never saved to the cache. The retry worked every time it triggered in the baseline and RAG runs. |
| 6 | Free-tier limits constrain the size of the experiments. | The free plan allows about 200K tokens per day. Measured cost: about 290 tokens per baseline question and about 433 per RAG question, so each full 100-question dev run costs roughly 29K to 43K tokens. | Cache every call, keep retrieved chunks short, smoke-test on 10 questions before each full run, and plan the heavy experiments (on a 200-question subset of test first) across several days. |
| 7 | The FAISS vector search crashed Python with a segmentation fault. | A known kind of conflict between FAISS and PyTorch on Apple Silicon, made more likely by the very new Python 3.14 and FAISS 1.15.1. Building embeddings worked; only the parallel search crashed. | Switched to exact cosine search with numpy, which gives the same results as a FAISS flat index at this size (10K vectors). FAISS is kept as an optional switch and can be retried in a Python 3.12 environment. |
| 8 | The baseline metrics looked stricter than the model really was. | The model sometimes gives a correct answer in different words (for example "Bangladesh" vs "People's Republic of Bangladesh"). | Added the "contains" metric and manually reviewed 20 baseline errors: 16 were real wrong answers (mostly wrong named entities), 2 were correct but phrased differently, 1 was a partial date and 1 was an abstention. |

## 7. Observations and limitations

- **Small, single-run results.** Everything above is from 100 dev questions at temperature 0, one run. With 100 questions, a proportion near 0.7 has a margin of roughly plus or minus 9 percentage points, so small gaps (such as B vs C on correctness, or EM 0.66 vs 0.63) should not be treated as real differences. Only the large gaps (A vs B and C) are clear.
- **The hallucination proxy depends on gold-answer matching.** Correct answers phrased differently from the gold text are counted as wrong, so the absolute "wrong and answered" values are somewhat pessimistic. The manual review suggests this affects roughly a few of every 20 failures.
- **Easier-than-real retrieval.** The evidence corpus is built from the dataset's own `knowledge` field, so retrieval is easier than open-domain search. The retrieval numbers should be read with that in mind.
- **Retrieval failures follow a pattern.** Vague questions with no distinctive name (for example "Which Magnolia actor was also a Navy combat veteran?") often fail to retrieve the right passage, and multi-hop questions can rank a related passage first.
- **Reported latency is API call time only.** Wall-clock time per question was several seconds because the code deliberately pauses between calls to respect rate limits, and sometimes retries.
- **Abstention quality is not yet analysed.** It is not yet known how many of the 17 abstentions in Config C were appropriate (the evidence really lacked the answer) and how many were over-cautious.
- **Prompts were not tuned.** One prompt per configuration was used. Test and validation splits are untouched so far.

## 8. Next steps

1. **Error analysis of B and C on dev.** Using the saved flags, split wrong answers into retrieval misses, correct evidence ignored or misread, and phrasing differences; check how many of C's abstentions were appropriate.
2. **Hallucination detection (runs locally, no API quota):** claim extraction from answers, NLI-based verification against the retrieved evidence, an answer-level supported/hallucinated decision, and threshold selection on the validation split. Evaluate the detector with accuracy, precision, recall and F1.
3. **Mitigation loop (Config D):** when a claim is unsupported, retrieve more evidence or refine the prompt, regenerate, and verify again (at most one or two retries).
4. **Controlled experiments E1 to E6** on a fixed subset of the test split: baseline vs RAG, Top-K of 1, 3 and 5, chunk sizes (small sizes such as 25, 40 and 60 words, since passages are short), prompt variants, verification, and mitigation.
5. **Tables, plots and final error analysis.**
6. **Streamlit demo, final report and presentation.**