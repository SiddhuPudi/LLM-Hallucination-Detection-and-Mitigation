# LLM Hallucination Detection & Mitigation: Progress Report

**Course:** Large Language Models, IIIT Dharwad
**Team members:** Kuladeep (23BDS021), Thrivikram (23BCS100), Harshith (23BDS058), Sadan (23BDS048)
**Status as of:** 1 November 2026 (baseline stage complete)

---

## 1. Project summary

This project builds a question-answering system on the **HaluEval QA** dataset to study and reduce hallucinations in LLM answers. It compares four configurations:

| Config | Description |
|---|---|
| A. Baseline | The LLM answers the question directly, with no external evidence. |
| B. RAG | Top-K evidence is retrieved (embeddings + FAISS) and given to the LLM. |
| C. RAG + grounded prompt | The prompt instructs the LLM to answer only from the evidence, or abstain. |
| D. RAG + verification + mitigation | Claims in the answer are checked against evidence with an NLI model; unsupported answers are regenerated. |

The systems will be compared on hallucination rate, answer accuracy, faithfulness, retrieval performance (Recall@K, MRR) and response latency. The project does not claim to eliminate hallucinations; it measures how much each stage improves reliability.

## 2. Work completed so far

1. **Environment.** Python virtual environment on a MacBook Air (no GPU) with the required libraries installed.
2. **Dataset.** HaluEval QA loaded and inspected (fields: `knowledge`, `question`, `right_answer`, `hallucinated_answer`). Cleaned the data by dropping empty fields and duplicate questions.
3. **Fixed splits.** Shuffled once with seed 42 and saved three permanent splits:
   - `dev`: 100 questions (development and debugging)
   - `val`: 300 questions (threshold tuning)
   - `test`: 500 questions (final reported results)

   The full cleaned dataset is also saved, because the retrieval index will later be built from the `knowledge` field of many samples rather than only the test rows.
4. **LLM access.** Built `src/llm.py`, a reusable function that calls the LLM API with:
   - a disk cache, so the same prompt is never paid for twice
   - throttling based on requests per minute and tokens per minute
   - retry handling for rate-limit and server errors
   - logging of latency, model version and token usage for every call
5. **Metrics.** Built `src/metrics.py` with Exact Match (EM), token-level F1 and a "contains" match (the gold answer appears inside the prediction). The contains metric is included because LLMs often answer in full sentences, which makes strict EM look pessimistic.
6. **Baseline (Config A).** Built `src/run_baseline.py` and ran the baseline LLM on the dev split.

## 3. LLM used

**Model:** `openai/gpt-oss-20b`, served through the free tier of the Groq API, with reasoning effort set to low. The same model will be used for every configuration, so that only the prompt and the evidence change between experiments. A small model was chosen deliberately: it makes enough mistakes on closed-book questions for retrieval and verification to show a measurable effect.

## 4. Baseline results (dev split, 100 questions, no retrieval)

| Metric | Value |
|---|---|
| Exact Match | 0.250 |
| Contains match | 0.310 |
| Token F1 | 0.301 |
| Average latency per question | 0.73 s |
| Average tokens per question | 290 (total 28,985) |

These numbers come from the first full baseline run on the dev split. They are development-stage results, not final results; the final comparison will be run on the test split. The baseline answers only about a quarter to a third of the questions correctly without evidence, which leaves clear room for retrieval to help.

## 5. Problems faced and how they were solved

| # | Problem | Cause | Solution |
|---|---|---|---|
| 1 | The first LLM choice (Gemini 3.8 Flash via Google AI Studio) could not run the baseline. | The free tier allows only 5 requests per minute and 20 requests per day, which is far below what 100+ questions per experiment need. | Checked the usage dashboard, stopped the run, and switched to a provider with a larger free quota (Groq). |
| 2 | Repeated HTTP 503 errors and 90-second read timeouts during the first baseline attempt. | The Gemini model was overloaded; the retry loop made things worse by spending the small daily quota on failed attempts. | Shortened timeouts, added smarter retry rules, and made the code stop with a clear message when a daily limit is reached instead of retrying. |
| 3 | Running a model locally (Ollama) was not an option. | The laptop has 8 GB of RAM and froze even with the smallest model. | Used a hosted API instead of a local model. |
| 4 | Groq returned a 404 "model not found" error for `llama-3.1-8b-instant`. | That model is not available on this account; the account lists only a few chat models. | Read the list of available models, and chose `openai/gpt-oss-20b`. |
| 5 | Some answers came back empty. | `gpt-oss-20b` is a reasoning model; its internal thinking sometimes used up the whole token limit before producing an answer. | Set reasoning effort to low, and made the code automatically retry with a larger token limit when an answer is empty. Empty answers are never saved to the cache. |
| 6 | Free-tier limits constrain the size of the experiments. | The free plan allows about 200K tokens per day, and RAG prompts (with retrieved passages) are much longer than baseline prompts. | Cache every call, keep retrieved chunks short, run the heavy experiments on a subset of the test split first (200 questions), and spread runs over several days. |

## 6. Observations and limitations so far

- The baseline numbers are on 100 development questions only; they will change on the larger test split and should not be reported as final.
- EM (0.250) is lower than the contains match (0.310) and F1 is similar to EM, which suggests some correct answers are phrased differently from the gold answer. Manual inspection of incorrect baseline answers is still needed to see which errors are real mistakes and which are formatting differences.
- So far only answer correctness has been measured. Hallucination rate and faithfulness require the evidence-verification stage, which is not built yet.
- The evidence corpus comes from the dataset's own `knowledge` field, so retrieval will be easier than in a real-world open-domain setting. This will be stated as a limitation in the final report.
- The free-tier token limit means experiments must be planned carefully and results cached.

## 7. Next steps

1. **Manual error review** of about 20 incorrect baseline answers on the dev split.
2. **Evidence corpus and chunking:** clean the `knowledge` field and split it into short chunks.
3. **Embeddings and FAISS:** embed chunks with a sentence-transformers model, build the index, and implement Top-K retrieval; check retrieved passages by hand and measure Recall@K and MRR.
4. **RAG generation (Config B) and grounded prompt (Config C)** on the dev split.
5. **Hallucination detection:** claim extraction, NLI-based verification, answer-level decision, and threshold selection on the validation split.
6. **Mitigation loop (Config D):** retrieve more evidence or refine the prompt, and regenerate when a claim is unsupported.
7. **Experiments E1 to E6** on a fixed subset of the test split, followed by tables, plots and error analysis.
8. **Streamlit demo, final report and presentation.**