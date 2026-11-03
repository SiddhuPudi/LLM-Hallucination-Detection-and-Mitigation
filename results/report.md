# LLM Hallucination Detection & Mitigation: Progress Report

**Course:** Large Language Models, IIIT Dharwad
**Team members:** Thrivikram (23BCS100), Kuladeep (23BDS021), Harshith (23BDS058), Sadan (23BDS048), Chaitra (23BDS016)
**Status as of:** 3 November 2026. Baseline, RAG generation, error analysis, re-scoring, claim extraction, NLI verification and the first hallucination detector (evaluated on the dev split) are complete. Threshold tuning on validation data and the mitigation loop are not built yet.

---

## 1. Project summary

This project builds a question-answering system on the **HaluEval QA** dataset to study and reduce hallucinations in LLM answers. It compares four configurations:

| Config | Description | Status |
| --- | --- | --- |
| A. Baseline | The LLM answers the question directly, with no external evidence. | Done (dev) |
| B. RAG | The top-3 evidence chunks are retrieved (embeddings + vector search) and given to the LLM. | Done (dev) |
| C. RAG + grounded prompt | The prompt tells the LLM to answer only from the evidence, or abstain. | Done (dev) |
| D. RAG + verification + mitigation | Claims in the answer are checked against the evidence; unsupported answers are regenerated. | Detector built and evaluated on dev; threshold tuning and the mitigation loop not built yet |

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
11. **Claim extraction.** Built `src/make_claims.py`, which rewrites each question and answer as one declarative sentence (Section 4.4). Ran it on the dev split: 381 claims.
12. **NLI verification.** Built `src/nli.py` and `src/verify.py`, which score every claim against the evidence with an NLI model (Section 4.5).
13. **Detector evaluation and analysis.** Built `src/detect_eval.py`, `src/detector_analysis.py` and `src/detector_variants.py`, which evaluate the detector on HaluEval's labeled pairs and on the pipeline's own answers, examine its mistakes, and compare score variants (Sections 5.5 to 5.7).

## 3. Models used

- **Answer-generating LLM:** `openai/gpt-oss-20b`, served through the free tier of the Groq API, with reasoning effort set to low. The same model is used for every configuration (baseline, RAG, claim rewriting), so only the prompt and the evidence change between experiments. A small model was chosen deliberately: it makes enough mistakes on closed-book questions for retrieval and verification to show a measurable effect.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`, run locally on CPU.
- **NLI verifier:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, run locally on CPU (download about 370 MB). It outputs probabilities for entailment, neutral and contradiction.

## 4. Method summary

### 4.1 Evidence corpus and retrieval

- **Corpus:** the `knowledge` field of all 10,000 cleaned rows (not only the dev, val and test rows), so the correct passage has to compete with many wrong ones.
- **Chunking:** word-based chunks of up to 120 words with a small overlap. The result was 10,040 chunks with an average of 55 words, so nearly every row stayed as a single chunk. The embedding model truncates its input at about 256 word pieces (roughly 150 words), so chunks longer than that would be cut; this is relevant for the planned chunk-size experiment, which will use small sizes.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized). Embedding all 10,040 chunks took about 20 seconds on the MacBook Air.
- **Search:** exact cosine similarity over all chunk vectors (numpy). FAISS is supported as an optional switch, but see problem 7 in Section 6.
- **Retrieval depth:** top-3 chunks per question.

### 4.2 Prompts

- **Config B (plain RAG):** the retrieved evidence is shown to the model with the instruction to use it to answer, giving only a short answer.
- **Config C (grounded + abstain):** the model is told to use only the evidence, not to use outside knowledge, and to reply exactly "Insufficient evidence" if the evidence does not contain the answer.

The exact prompts are in Appendix B.

### 4.3 Metrics

**Answer correctness is reported at three levels**, because the original strict metric turned out to undercount correct answers (Section 5.3):

- **Strict "contains":** the normalized gold answer appears inside the normalized prediction. This is the original metric and its numbers are unchanged.
- **Lenient "contains":** strict match, or every key word of the gold answer appears in the prediction after cleaning. The cleaning treats Unicode dashes and curly quotes as ordinary ones, maps number words to digits ("eight" to "8"), and ignores the words a, an, the, and, of.
- **Manually verified:** lenient match plus a manual decision on answers that are a shorter form of the gold answer (for example "Bangladesh" for "People's Republic of Bangladesh") and on one plural-form case. The verdicts are listed in Appendix A. Incomplete answers (for example a year where a full date was asked) are not counted as correct.

Other metrics:

- **Exact Match (EM), token F1:** standard answer-matching metrics (strict normalization, unchanged).
- **Abstained:** the answer says "unknown", "insufficient evidence" or similar.
- **Wrong and answered:** the model gave an answer (did not abstain) and it is not correct under the stated scoring level. This is a simple **hallucination-rate proxy** based on the gold answers; the claim-level detector (Section 4.5) measures faithfulness to the evidence directly.
- **Evidence had answer:** the gold answer text appears in the retrieved evidence. This is a ceiling on what a faithful answer can achieve. It cannot work for yes/no questions, where "yes" or "no" never appears in the passage (4 dev questions).
- **Retrieval:** Recall@K (a chunk from the question's own source row is in the top K), AnswerInChunk@K (the gold answer text is in the top K chunks) and MRR.
- **Cost:** average tokens per question and average API latency per question.
- **Detector metrics:** AUC (how well the hallucination score separates the two classes, independent of any threshold), and accuracy, precision, recall and F1 at a stated threshold.

### 4.4 Claim extraction

The generated answers are very short (often one to three words), so splitting them into sentences would do nothing. Instead, the LLM rewrites each question and answer as **one short declarative sentence** that states the answer as a fact, without adding information (prompt in Appendix B). For example, the question "Who won the singles title at a 1973 tournament held at a Los Angeles arena designed by Charles Luckman?" with the answer "Margaret Court" becomes a sentence stating that Margaret Court won that title.

The claim step is applied to four kinds of answers on the dev split:

- HaluEval's `right_answer` and `hallucinated_answer` (labeled examples for measuring the detector, 100 each). Both go through the same rewriting step so that the detector cannot rely on style differences: HaluEval's hallucinated answers are usually full sentences while the right answers are short phrases.
- The non-abstained answers of Config B (98) and Config C (83).

This produced 381 claims. Inspection showed that the claims are faithful and add no new facts, and that claims for genuine yes/no questions are correct. One limitation: for questions phrased as statements, a "Yes." answer becomes a claim that simply restates the question, which can look supported even when the answer does not respond to the question.

### 4.5 Verification and detector

**Premise and hypothesis.** The NLI model reads the evidence (premise) and the claim (hypothesis) and outputs the probabilities of entailment, neutral and contradiction. Two kinds of evidence are used:

- **Retrieved evidence:** the top-3 passages that Configs B and C actually used (the realistic setting).
- **Oracle evidence:** the question's own `knowledge` text (an upper bound that separates retrieval errors from verifier errors).

**NLI score.** Hallucination score = 1 minus the probability of entailment.

**Lexical support.** Added after an error analysis showed that NLI often fails when the passage uses a pronoun instead of the entity name (Section 5.6). The lexical support of a claim is the share of its new content words (words not already in the question, excluding a small list of function and negation words) that appear in the evidence. If the claim contains no new words, the NLI entailment is used instead.

**Combination.** Two combinations of the NLI entailment and the lexical support were compared: the larger of the two ("max") and their average ("mean"). The hallucination score is 1 minus the combined support, and a claim is flagged as unsupported when the score reaches a threshold. The threshold of 0.5 used so far is a placeholder; **the final threshold will be selected on validation data only.**

### 4.6 Detector evaluation protocol

1. **HaluEval labeled pairs:** the claim from each `right_answer` is a supported example and the claim from each `hallucinated_answer` is a hallucinated example (200 claims on dev).
2. **Pipeline answers:** can the detector flag the wrong answers of Config B (98 answered, 21 wrong) and Config C (83 answered, 7 wrong)? "Wrong" means incorrect under the lenient metric.

Each evaluation is run with retrieved and with oracle evidence.

## 5. Results (dev split, 100 questions)

All numbers below come from our own runs on the 100-question dev split. They are development-stage results, not final results.

### 5.1 Retrieval (120-word chunks, 10,040-chunk index)

| Metric | Value |
| --- | --- |
| Recall@1 | 0.830 |
| Recall@3 | 0.900 |
| Recall@5 | 0.920 |
| Recall@10 | 0.940 |
| AnswerInChunk@1 / @3 / @5 / @10 | 0.810 / 0.860 / 0.900 / 0.920 |
| MRR@10 | 0.871 |

The correct passage is ranked first for 83% of questions and is in the top 3 for 90%. Top-3 was therefore chosen as the starting evidence size. In the 10 questions where retrieval failed, the questions were typically vague and contained no distinctive name (for example "Which Magnolia actor was also a Navy combat veteran?").

### 5.2 Answer quality, hallucination proxy and cost

| Config | Strict contains | Lenient contains | Manually verified | Abstained | Wrong and answered (strict / lenient / verified) |
| --- | --- | --- | --- | --- | --- |
| A. Baseline | 0.31 | 0.31 | 0.33 | 0.01 | 0.68 / 0.68 / 0.66 |
| B. RAG (top-3) | 0.71 | 0.77 | 0.83 | 0.02 | 0.27 / 0.21 / 0.15 |
| C. RAG + grounded (top-3) | 0.72 | 0.76 | 0.79 | 0.17 | 0.11 / 0.07 / 0.04 |

| Config | EM (strict) | Token F1 (strict) | Avg tokens per question | Avg API latency |
| --- | --- | --- | --- | --- |
| A. Baseline | 0.25 | 0.301 | 290 | 0.73 s |
| B. RAG (top-3) | 0.66 | 0.731 | 433 | 0.77 s |
| C. RAG + grounded (top-3) | 0.63 | 0.717 | 434 | 0.69 s |

Average retrieval time was about 0.01 to 0.02 s per question. In B and C, the retrieved evidence contained the gold answer text for 86% of questions, and a chunk from the question's own source row was retrieved for 90%. Strict and lenient scores are reproducible from code (`src/rescore.py`); the manually verified scores depend on the author's verdicts in Appendix A.

### 5.3 Error analysis of the answers

**The strict metric undercounted correct answers.** Of the 11 answers that Config C got "wrong despite having the evidence" under the strict metric, 7 were formatting or short-form artifacts (for example "Nigel Martin‑Smith" written with a non-breaking hyphen, "8" for "eight", "Barbara Carroll Forrest" for "Barbara Forrest", "Bangladesh" for "People's Republic of Bangladesh"). Two were incomplete answers and only 2 were truly wrong. For Config B, 12 of 18 such cases were artifacts. After correction, the real misreadings of present evidence are about 3 in B and 2 in C.

**Where Config B's 15 remaining wrong answers come from (manually verified scoring):**

| Cause | Count |
| --- | --- |
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

### 5.4 What the answer results show

- **Retrieval helps a lot.** Adding evidence raised the share of correct answers from about 0.33 (baseline) to 0.83 (B) and 0.79 (C), and cut confidently wrong answers from about 66% of questions to 15% (B) and 4% (C). This holds at every scoring level, and the gap is far beyond what chance variation on 100 questions could explain.
- **The grounded prompt trades a little accuracy for far fewer fabricated answers.** Config C gave 4 points fewer correct answers than B but removed most of B's wrong answers, replacing them with explicit abstentions, including on every retrieval failure. This is the behaviour we wanted. The correctness gap between B and C is only a few questions, so it should not be over-interpreted.
- **Cost of RAG.** Prompts are about 50% longer (290 to about 433 tokens per question), while API latency is about the same. Retrieval itself adds only milliseconds.

### 5.5 Hallucination detector results

Scoring 381 claims with the NLI model took 328 seconds with retrieved evidence and 161 seconds with oracle evidence on the MacBook Air CPU. The mean probability of entailment was 0.635 (retrieved) and 0.747 (oracle) for right-answer claims, against 0.277 and 0.282 for hallucinated-answer claims.

**HaluEval labeled pairs (200 claims, threshold 0.5 as a placeholder).**

| Score | AUC retrieved | AUC oracle | Retrieved: precision / recall / F1 |
| --- | --- | --- | --- |
| NLI only (1 - entailment) | 0.732 | 0.792 | 0.664 / 0.730 / 0.695 |
| Lexical support only | 0.724 | 0.755 | 0.713 / 0.670 / 0.691 |
| max(NLI, lexical) | 0.759 | 0.807 | 0.771 / 0.540 / 0.635 |
| mean(NLI, lexical) | **0.771** | **0.830** | 0.742 / 0.690 / 0.715 |

Confusion matrices with retrieved evidence (rows: right / hallucinated claims; columns: kept / flagged):

| Score | Right claims kept | Right claims flagged | Hallucinated caught | Hallucinated missed |
| --- | --- | --- | --- | --- |
| NLI only | 63 | 37 | 73 | 27 |
| max(NLI, lexical) | 84 | 16 | 54 | 46 |
| mean(NLI, lexical) | 76 | 24 | 69 | 31 |

Using the NLI contradiction probability, or contradiction minus entailment, instead of 1 minus entailment barely changed the NLI-only AUC (retrieved: 0.693 and 0.732 vs 0.732; oracle: 0.795 and 0.811 vs 0.792), so the score definition is not the bottleneck.

**The detector on the pipeline's own answers (label: wrong under the lenient metric; retrieved evidence).**

| Answers | Score | AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Config B (98 answered, 21 wrong) | NLI only | 0.678 | 0.333 | 0.571 | 0.421 |
| | Lexical only | 0.724 | 0.444 | 0.571 | 0.500 |
| | max | 0.745 | 0.625 | 0.476 | 0.541 |
| | mean | 0.745 | 0.480 | 0.571 | 0.522 |
| Config C (83 answered, 7 wrong) | NLI only | 0.582 | 0.125 | 0.429 | 0.194 |
| | Lexical only | 0.746 | 0.278 | 0.714 | 0.400 |
| | max | 0.733 | 0.375 | 0.429 | 0.400 |
| | mean | 0.731 | 0.286 | 0.571 | 0.381 |

With oracle evidence the AUC values for Config B were 0.698 (NLI), 0.756 (lexical), 0.787 (max) and 0.750 (mean); for Config C they were 0.549, 0.646, 0.680 and 0.527, but Config C has only 7 wrong answers, so these values are very noisy.

### 5.6 Error analysis of the detector

**False alarms (right claims flagged as hallucinated).** With NLI only and retrieved evidence, 37 of 100 right claims were flagged. All 10 right answers whose source passage was not retrieved were flagged, which is correct behaviour because the evidence really does not support them. Only 27 of the 90 right claims whose source passage was retrieved (30%) were flagged; with oracle evidence 26 of 100 were flagged. Of the 8 worst false alarms:

- 6 involve passages that use a pronoun or omit the entity name ("It was founded in 1996 by Nobel laureate Milton Friedman" for EdChoice, "its long-time leader Roberto Formigoni" for Network Italy, "endorsed the diet" for the Werewolf Diet, "The film stars ... Catherine O'Hara", "As with the TF101, it was piloted by..." for the TF102). The claim names the entity, the passage does not, and the NLI model cannot connect them.
- 1 claim copies question details that the passage does not contain (the release date and label of Joyner Lucas's mixtape).
- 1 requires a comparison (which of two people contributed to more films).

The lexical support score was introduced to address the missing-entity cases: it checks whether the answer's own words occur in the evidence. It reduced false alarms from 37 to 24 (mean) or 16 (max) at the cost of more missed hallucinations (31 or 46 instead of 27).

**Misses (hallucinated claims judged supported).** The 8 hallucinated claims with the highest support are mostly not detector errors:

- 5 are actually true, so HaluEval's "hallucinated" label is wrong: the 337th Flight Test Squadron was based in California, Newark is what Street Fight and Cory Booker share, Milton Friedman is American, Neofelis lives in Southeast Asia, and Beth Gibbons and Han Seung-yeon are both performers.
- 2 are true statements that do not answer the question (Adam Clayton Powell is a documentary, but the question asked which documentary is about Finnish rock groups; Claire Keane did contribute to Disney films, but the question asked who contributed more).
- 1 is a true but generic statement (Robin Shou acted in Fatal Termination).

Only 2 questions had literally identical right and hallucinated claims, so the label noise is larger than that count suggests. These 8 are the most extreme cases, not a sample, so the overall noise rate is not yet known; a manual audit of all 27 missed claims would measure it.

### 5.7 What the detector results show

- **The detector is a useful but imperfect signal.** On the labeled pairs its AUC is about 0.73 to 0.77 with retrieved evidence, clearly better than chance (0.5) but not reliable enough to accept or reject answers automatically at the placeholder threshold.
- **Combining NLI with the lexical check helps.** The average of the two ("mean") has the best AUC on the pairs (0.771), ties for the best on Config B (0.745) and is close to the best on Config C. "max" is more cautious: higher precision, lower recall. Which suits the mitigation loop better depends on the cost of each mistake (a false alarm costs a regeneration and risks replacing a correct answer; a miss leaves a wrong answer in place) and will be decided on validation data.
- **Retrieval is not the main limit of the detector.** Oracle evidence improves the pair AUC only from 0.73 to 0.79, so most of the error comes from the verifier, the claim format and the dataset, not from retrieval.
- **The detector measures support by the evidence, not agreement with the gold answer.** The remaining wrong answers of Config C are largely incomplete answers (a year instead of a date, one of two occupations) whose claims are supported by the evidence, so they should not be flagged by a faithfulness check.
- **The accuracy on the pairs is probably understated** because of HaluEval label noise, but this cannot be quantified until the audit is done.
- **A first working detector** is mean(NLI, lexical) with retrieved evidence, with max kept as a more precise alternative; the final choice and threshold are selected on validation data.

## 6. Problems faced and how they were solved

| # | Problem | Cause | Solution |
| --- | --- | --- | --- |
| 1 | The first LLM choice (Gemini 3.8 Flash via Google AI Studio) could not run the baseline. | The free tier allows only 5 requests per minute and 20 requests per day, far below what 100+ questions per experiment need. | Checked the usage dashboard, stopped the run, and switched to a provider with a larger free quota (Groq). |
| 2 | Repeated HTTP 503 errors and 90-second read timeouts during the first baseline attempt. | The Gemini model was overloaded; the retry loop made things worse by spending the small daily quota on failed attempts. | Shortened timeouts, added smarter retry rules, and made the code stop with a clear message when a daily limit is reached. |
| 3 | Running a model locally (Ollama) was not an option. | The laptop has 8 GB of RAM and froze even with the smallest model. | Used a hosted API instead of a local model. |
| 4 | Groq returned a 404 "model not found" error for `llama-3.1-8b-instant`. | That model is not available on this account; the account lists only a few chat models. | Read the account's model list and chose `openai/gpt-oss-20b`. |
| 5 | Some answers came back empty. | `gpt-oss-20b` is a reasoning model; its internal thinking sometimes used up the whole token limit before producing an answer. | Set reasoning effort to low, and made the code retry automatically with a larger token limit. Empty answers are never saved to the cache. The retry worked every time it triggered in the baseline and RAG runs. |
| 6 | Free-tier limits constrain the size of the experiments. | The free plan allows about 200K tokens per day. Measured cost: about 290 tokens per baseline question, about 433 per RAG question and about 240 per rewritten claim. | Cache every call, keep retrieved chunks short, smoke-test on 10 questions before each full run, and plan the heavy experiments (on a 200-question subset of test first) across several days. |
| 7 | The FAISS vector search crashed Python with a segmentation fault. | A known kind of conflict between FAISS and PyTorch on Apple Silicon, made more likely by the very new Python 3.14 and FAISS 1.15.1. Building embeddings worked; only the parallel search crashed. | Switched to exact cosine search with numpy, which gives the same results as a FAISS flat index at this size (10K vectors). FAISS is kept as an optional switch and can be retried in a Python 3.12 environment. |
| 8 | The baseline metrics looked stricter than the model really was. | The model sometimes gives a correct answer in different words (for example "Bangladesh" vs "People's Republic of Bangladesh"). | Added the "contains" metric and manually reviewed 20 baseline errors: 16 were real wrong answers (mostly wrong named entities), 2 were correct but phrased differently, 1 was a partial date and 1 was an abstention. |
| 9 | The error analysis showed that the "contains" metric still marked many correct RAG answers as wrong, which inflated the hallucination proxy. | The text normalizer did not handle Unicode dash characters (non-breaking hyphens), number words ("eight" vs "8"), extra middle names, or shorter forms of a name. | Kept the strict numbers unchanged for traceability, added a lenient metric that handles these cases, and manually verified the remaining short-form answers (Appendix A). All three levels are reported. |
| 10 | The NLI detector flagged many correct claims (37 of 100 right claims). | HaluEval passages often use a pronoun instead of the entity name, so the NLI model cannot link the claim to the passage; some claims also copy question details that the passage lacks. | Added a lexical support score (do the answer's words appear in the evidence?) and combined it with NLI, which cut the false alarms to 24 (mean) or 16 (max). The combination was designed on dev, so it must be confirmed on validation data. |
| 11 | The detector appeared to miss many hallucinated claims. | Many of HaluEval's "hallucinated" answers are actually true or merely fail to answer the question, so they are not hallucinations in the evidence-based sense. | Inspected the highest-support misses, documented the label noise as a limitation, and planned a manual audit of all missed claims. |
| 12 | Minor tooling issues. | zsh treated a trailing `#` comment in a pasted command as an argument; Python 3.14 produced harmless FutureWarnings from PyTorch, and the Hugging Face hub warned about unauthenticated downloads. | Removed the comments from the commands; the warnings were ignored because they do not affect results. |

## 7. Observations and limitations

- **Small, single-run results.** Everything above is from 100 dev questions at temperature 0, one run. With 100 questions, a proportion near 0.8 has a margin of roughly plus or minus 8 to 9 percentage points, so small gaps (such as B vs C on correctness) should not be treated as real differences. Only the large gaps (A vs B and C, and the drop in wrong answers) are clear.
- **Detector results are preliminary.** The lexical support score was designed after looking at dev errors, so the dev numbers are optimistic. With 200 balanced claims an AUC has a standard error of roughly 0.03, so the differences between the score variants (0.02 to 0.04) are within about one standard error. Config C has only 7 wrong answers, so its detector results are very noisy. The 0.5 threshold is a placeholder. All of this must be re-examined on validation data.
- **Scoring depends on gold-string matching.** Even the lenient and manually verified scores compare against a single gold string, which penalizes valid alternative phrasings. The manual verdicts in Appendix A were made by one person (the author) on a small number of answers.
- **HaluEval label noise.** Some "hallucinated" answers are actually true or only fail to answer the question, so detector accuracy on the labeled pairs is understated by an unknown amount. Some questions are also phrased as statements, one is malformed, and yes/no gold answers cannot be found literally in the evidence, so the "evidence had answer" flag is unreliable for them.
- **Claim rewriting can lose the question's constraint.** A claim can be true and supported without answering the question (for example for comparison questions or for a "Yes." answer to a statement-style question), so the detector measures support by evidence, not correctness.
- **Evidence passages often lack the entity name.** HaluEval passages frequently replace the entity name with a pronoun or start mid-sentence, so even correctly retrieved evidence can be only implicit. This makes strict grounding harder, explains several abstentions and is the main cause of the NLI false alarms.
- **Easier-than-real retrieval.** The evidence corpus is built from the dataset's own `knowledge` field, so retrieval is easier than open-domain search. The retrieval numbers should be read with that in mind.
- **Reported latency is API call time only.** Wall-clock time per question was several seconds because the code deliberately pauses between calls to respect rate limits, and sometimes retries.
- **Prompts were not tuned.** One prompt per configuration was used. The validation and test splits are untouched so far.

## 8. Reproducibility and compute cost

- **Reproducibility.** Fixed splits (seed 42), temperature 0 for all LLM calls, every LLM call cached on disk (keyed by model, reasoning setting, temperature, token limit and prompt), and the model version logged with each answer. All results are produced by scripts in `src/` from saved files, and the scoring, detector analysis and variant comparison re-use saved predictions and claim scores without new API calls.
- **API token budget used so far (dev stages).** Baseline about 29K tokens (100 questions), Config B about 43K, Config C about 43K, claim rewriting about 83K new tokens (381 claims, about 10 minutes), for a total of about 199K tokens across two days of the free tier.
- **Local compute (MacBook Air, CPU only).** Embedding the 10,040 chunks about 20 seconds; retrieval about 0.01 to 0.02 seconds per question; NLI verification of 381 claims 328 seconds (retrieved evidence) and 161 seconds (oracle evidence).

## 9. Next steps

1. **Validation runs and threshold selection.** Run Config B and Config C on a 100-question subset of the validation split, rewrite their answers and the HaluEval pairs into claims, score them with the NLI model and the lexical check, and choose between "mean" and "max" and the threshold using validation data only. This costs roughly 200K tokens in total (about 43K for each configuration plus about 90K for the claims), so it is spread over two days.
2. **Mitigation loop (Config D).** When the detector flags an answer, retrieve more evidence or refine the prompt, regenerate, and verify again (at most one or two retries). The error analysis suggests that about half of the wrong answers come from retrieval failures, so retrieving more evidence is a promising first strategy; the detector's false alarms must also be kept low so that correct answers are not replaced.
3. **Label-noise audit (optional).** Check by hand the 27 hallucinated claims that the first detector missed, to estimate how many are not really hallucinations, and report the detector's accuracy on the cleaned pairs.
4. **Controlled experiments E1 to E6** on a fixed subset of the test split: baseline vs RAG, Top-K of 1, 3 and 5, chunk sizes (small sizes such as 25, 40 and 60 words, since passages are short), prompt variants, verification, and mitigation. All reported with strict, lenient and manually verified scoring where applicable.
5. **Tables, plots and final error analysis.**
6. **Streamlit demo, final report and presentation.**

## Appendix A: Manual verdicts on short-form answers (dev split)

These answers are shorter or slightly different forms of the gold answer. Each was judged by hand and is reflected in the "manually verified" scores.

| Question (shortened) | Gold answer | Model answer | Verdict |
| --- | --- | --- | --- |
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

## Appendix B: Prompts used

**Baseline (Config A):**

```
Answer the question in as few words as possible. Give only the answer, no explanation.

Question: {question}
Answer:
```

**Plain RAG (Config B):**

```
Use the evidence below to answer the question. Give only the answer in as few words as possible, no explanation.

Evidence:
{evidence}

Question: {question}
Answer:
```

**Grounded RAG with abstention (Config C):**

```
Answer the question using ONLY the evidence below. Do not use outside knowledge. If the evidence does not contain the answer, reply exactly: Insufficient evidence.
Otherwise give only the answer in as few words as possible, no explanation.

Evidence:
{evidence}

Question: {question}
Answer:
```

**Claim rewriting:**

```
Rewrite the question and its answer as ONE short, self-contained declarative sentence that states the answer as a fact. Do not add any information that is not in the question or the answer. If the answer is Yes or No, state the fact accordingly. Output only the sentence.

Question: {question}
Answer: {answer}
Sentence:
```

In the RAG prompts, `{evidence}` is the three retrieved passages numbered [1], [2] and [3].
