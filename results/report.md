# LLM Hallucination Detection & Mitigation: Progress Report

**Course:** Large Language Models, IIIT Dharwad
**Team members:** Thrivikram (23BCS100), Kuladeep (23BDS021), Harshith (23BDS058), Sadan (23BDS048), Chaitra (23BDS016)
**Status as of:** 2 November 2026. Baseline, RAG generation, error analysis and re-scoring are complete; hallucination detection and mitigation are not built yet.

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
9. **Error analysis.** Built `src/error_analysis.py`, which sorts every dev question into an outcome category for B and C and tracks how questions move between configurations. A manual review of the printed examples showed that many "wrong" answers were scoring artifacts (Section 5.3).
10. **Re-scoring.** Added a lenient metric to `src/metrics.py` and built `src/rescore.py`, which re-scores the saved predictions (no new API calls) and lists short-form answers for manual verification.

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

**Answer correctness is reported at three levels**, because the original strict metric turned out to undercount correct answers (Section 5.3):

- **Strict "contains":** the normalized gold answer appears inside the normalized prediction. This is the original metric and its numbers are unchanged.
- **Lenient "contains":** strict match, or every key word of the gold answer appears in the prediction after cleaning. The cleaning treats Unicode dashes and curly quotes as ordinary ones, maps number words to digits ("eight" to "8"), and ignores the words a, an, the, and, of.
- **Manually verified:** lenient match plus a manual decision on answers that are a shorter form of the gold answer (for example "Bangladesh" for "People's Republic of Bangladesh") and on one plural-form case. The verdicts are listed in Appendix A. Incomplete answers (for example a year where a full date was asked) are not counted as correct.

Other metrics:

- **Exact Match (EM), token F1:** standard answer-matching metrics (strict normalization, unchanged).
- **Abstained:** the answer says "unknown", "insufficient evidence" or similar.
- **Wrong and answered:** the model gave an answer (did not abstain) and it is not correct under the stated scoring level. This is used as a simple **hallucination-rate proxy** for now. Claim-level verification (not built yet) will add an evidence-based measure.
- **Evidence had answer:** the gold answer text appears in the retrieved evidence. This is a ceiling on what a faithful answer can achieve. It cannot work for yes/no questions, where "yes" or "no" never appears in the passage (4 dev questions).
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

The correct passage is ranked first for 83% of questions and is in the top 3 for 90%. Top-3 was therefore chosen as the starting evidence size. In the 10 questions where retrieval failed, the questions were typically vague and contained no distinctive name (for example "Which Magnolia actor was also a Navy combat veteran?").

### 5.2 Answer quality, hallucination proxy and cost

| Config | Strict contains | Lenient contains | Manually verified | Abstained | Wrong and answered (strict / lenient / verified) |
|---|---|---|---|---|---|
| A. Baseline | 0.31 | 0.31 | 0.33 | 0.01 | 0.68 / 0.68 / 0.66 |
| B. RAG (top-3) | 0.71 | 0.77 | 0.83 | 0.02 | 0.27 / 0.21 / 0.15 |
| C. RAG + grounded (top-3) | 0.72 | 0.76 | 0.79 | 0.17 | 0.11 / 0.07 / 0.04 |

| Config | EM (strict) | Token F1 (strict) | Avg tokens per question | Avg API latency |
|---|---|---|---|---|
| A. Baseline | 0.25 | 0.301 | 290 | 0.73 s |
| B. RAG (top-3) | 0.66 | 0.731 | 433 | 0.77 s |
| C. RAG + grounded (top-3) | 0.63 | 0.717 | 434 | 0.69 s |

Average retrieval time was about 0.01 to 0.02 s per question. In B and C, the retrieved evidence contained the gold answer text for 86% of questions, and a chunk from the question's own source row was retrieved for 90%. Strict and lenient scores are reproducible from code (`src/rescore.py`); the manually verified scores depend on the author's verdicts in Appendix A.

### 5.3 Error analysis

**The strict metric undercounted correct answers.** Of the 11 answers that Config C got "wrong despite having the evidence" under the strict metric, 7 were formatting or short-form artifacts (for example "Nigel Martin‑Smith" written with a non-breaking hyphen, "8" for "eight", "Barbara Carroll Forrest" for "Barbara Forrest", "Bangladesh" for "People's Republic of Bangladesh"). Two were incomplete answers and only 2 were truly wrong. For Config B, 12 of 18 such cases were artifacts. After correction, the real misreadings of present evidence are about 3 in B and 2 in C.

**Where Config B's 15 remaining wrong answers come from (manually verified scoring):**

| Cause | Count |
|---|---|
| Retrieval failed and the model guessed (including "No such player.") | 8 |
| Evidence was present but misread (for example "¡Viva Zapata!" for a question about an album; "Yes." to a statement-style question) | 3 |
| Yes/no question answered incorrectly | 1 |
| Incomplete answer (year instead of full date; one of two occupations) | 2 |
| Malformed question (not counted as correct) | 1 |

Roughly half of B's wrong answers therefore come from retrieval failures, which is important for designing the mitigation loop: better retrieval can fix about half, and verification must catch the rest.

**Retrieval failures and abstention.** In 10 of 100 dev questions the question's own source passage was not in the top 3. In these, Config B guessed wrongly 8 times, abstained once and was correct once (from the model's own knowledge, not from the evidence). **Config C abstained on all 10.** Overall, C abstained on 17 questions: 10 retrieval failures, 1 yes/no question, and 6 where the retrieved passage only implied the answer. In these 6, HaluEval's passage uses a pronoun or starts mid-sentence ("He is the chairman...", "It is the sister company of Wellcome"), so it never names the entity the question asks about. Abstaining there is defensible for a strictly grounded model.

**Config B vs Config C.** Relative to B, Config C turned 11 of B's 15 wrong answers into abstentions and 1 into a correct answer; 3 stayed wrong and 1 new wrong answer appeared (the model answered "Yes." to "Battletoads/Double Dragon is a game that can be played on this company's Game Boy?"). C gave up 5 answers that B had correct: 4 by abstaining (3 where the passage only implied the answer, and 1 where B had answered from its own memory because retrieval failed) and the "Yes." case. Net effect: correct answers 83 to 79 per 100, wrong answers 15 to 4, abstentions 2 to 17.

**Other patterns.**
- Two questions are phrased as statements ending in "?" (for example "July 31, 1912 this Nobel Prize winner in Economic Sciences founded EdChoice?") and the model sometimes answers "Yes." to them.
- One B answer (David Puttnam) was correct although the retrieved evidence did not contain it. This shows that a correct answer is not necessarily a faithful one, which is the motivation for evidence-based verification.

### 5.4 What the results show

- **Retrieval helps a lot.** Adding evidence raised the share of correct answers from about 0.33 (baseline) to 0.83 (B) and 0.79 (C), and cut confidently wrong answers from about 66% of questions to 15% (B) and 4% (C). This holds at every scoring level, and the gap is far beyond what chance variation on 100 questions could explain.
- **The grounded prompt trades a little accuracy for far fewer fabricated answers.** Config C gave 4 points fewer correct answers than B but removed most of B's wrong answers, replacing them with explicit abstentions, including on every retrieval failure. This is the behaviour we wanted. The correctness gap between B and C is only a few questions, so it should not be over-interpreted.
- **Cost of RAG.** Prompts are about 50% longer (290 to about 433 tokens per question), while API latency is about the same. Retrieval itself adds only milliseconds.

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
| 9 | The error analysis showed that the "contains" metric still marked many correct RAG answers as wrong, which inflated the hallucination proxy. | The text normalizer did not handle Unicode dash characters (non-breaking hyphens), number words ("eight" vs "8"), extra middle names, or shorter forms of a name. | Kept the strict numbers unchanged for traceability, added a lenient metric that handles these cases, and manually verified the remaining short-form answers (Appendix A). All three levels are reported. |

## 7. Observations and limitations

- **Small, single-run results.** Everything above is from 100 dev questions at temperature 0, one run. With 100 questions, a proportion near 0.8 has a margin of roughly plus or minus 8 to 9 percentage points, so small gaps (such as B vs C on correctness) should not be treated as real differences. Only the large gaps (A vs B and C, and the drop in wrong answers) are clear.
- **Scoring depends on gold-string matching.** Even the lenient and manually verified scores compare against a single gold string, which penalizes valid alternative phrasings. The manual verdicts in Appendix A were made by one person (the author) on a small number of answers.
- **Easier-than-real retrieval.** The evidence corpus is built from the dataset's own `knowledge` field, so retrieval is easier than open-domain search. The retrieval numbers should be read with that in mind.
- **Evidence passages often lack the entity name.** HaluEval passages frequently replace the entity name with a pronoun or start mid-sentence, so even correctly retrieved evidence can be only implicit. This makes strict grounding harder and explains several abstentions.
- **Dataset quirks.** Some questions are phrased as statements, one is malformed, and yes/no gold answers cannot be found literally in the evidence, so the "evidence had answer" flag is unreliable for them.
- **Reported latency is API call time only.** Wall-clock time per question was several seconds because the code deliberately pauses between calls to respect rate limits, and sometimes retries.
- **Prompts were not tuned.** One prompt per configuration was used. The validation and test splits are untouched so far.
- **Hallucination rate is still a proxy.** "Wrong and answered" compares against gold answers. Evidence-based claim verification (next stage) will measure faithfulness directly.

## 8. Next steps

1. **Hallucination detection (the claim step uses a small amount of API quota; verification runs locally):**
   - *Claim extraction.* Answers are very short, so each question-answer pair is rewritten as one declarative sentence by the LLM. This is done for HaluEval's `right_answer` and `hallucinated_answer` (labeled examples for measuring the detector, rewritten through the same step so the detector cannot rely on style differences) and for the non-abstained B and C answers.
   - *Verification.* A small NLI model checks whether the retrieved evidence entails, contradicts or does not support each claim, producing a hallucination score and an answer-level supported/hallucinated decision.
   - *Evaluation.* Accuracy, precision, recall and F1 on the labeled pairs, using both retrieved evidence (realistic) and the question's own knowledge text (upper bound). The threshold is selected on a validation subset only.
2. **Mitigation loop (Config D):** when a claim is unsupported, retrieve more evidence or refine the prompt, regenerate, and verify again (at most one or two retries). The error analysis suggests about half of the wrong answers come from retrieval failures, so retrieving more evidence is a promising first strategy.
3. **Controlled experiments E1 to E6** on a fixed subset of the test split: baseline vs RAG, Top-K of 1, 3 and 5, chunk sizes (small sizes such as 25, 40 and 60 words, since passages are short), prompt variants, verification, and mitigation. All reported with strict, lenient and manually verified scoring where applicable.
4. **Tables, plots and final error analysis.**
5. **Streamlit demo, final report and presentation.**

## Appendix A: Manual verdicts on short-form answers (dev split)

These answers are shorter or slightly different forms of the gold answer. Each was judged by hand and is reflected in the "manually verified" scores.

| Question (shortened) | Gold answer | Model answer | Verdict |
|---|---|---|---|
| Who was born first, Sharon den Adel or Ezra Koenig? | Sharon Janny den Adel | Sharon den Adel (A, B, C) | Correct (same person, common name) |
| Azfar Hussain ... official language of which country? | People's Republic of Bangladesh | Bangladesh (A, B, C) | Correct (short form of the country name) |
| What the locals consider ... suburb of a town in the Division of Mayo | township of Stirling | Stirling (B, C) | Correct |
| Miss A member in While You Were Sleeping | Bae Suzy | Suzy (B) | Correct (stage name) |
| Darrelle Revis's team two years after he was drafted | The New York Jets | Jets (B) | Correct |
| What type of game are Total Overdose and Chili Con Carnage? | third-person shooter | "...third-person shooters." (B) | Correct (plural only) |
| When was the 2007 movie released? | January 12, 2007 | 2007 (A, B, C) | Incomplete, not counted |
| What job do both Selpin and Cavani have? | film director and screenwriter | Film director (A, B, C) | Incomplete, not counted |
| Which UK equestrian weekly is Mark Phillips a columnist in? | Horse & Hound | The Horse (A) | Wrong, a different magazine |
| ...and which South Korean actor, born on ? (malformed question) | July 5, 1984 | Yeon Woo-jin (B) | Not counted (the question is malformed; C answered with name and date and is counted correct) |