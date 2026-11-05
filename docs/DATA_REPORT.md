# Data report (565-essay spreadsheet)

Everything here was measured by running `python -m ielts_ft.data_prep` on `IELTS_Task2_565_cleaned_minimal.xlsx`
(`data_report.json` is written next to the output). Re-run it after changing the data.

## What is in the file

| Column | Content | How it is used |
|---|---|---|
| `essay_prompt`, `essay` | Task 2 question and student answer | Model input |
| `essay_score`, `Rater1_*` | Numeric human scores (overall, TR, CC, LR, GRA) | **Labels** (`essay_score` = overall) |
| `Rater2_TR/CC/LR/GRA` | Text: "Band N / Positives / Negatives" with quotes | Strengths, weaknesses, evidence |
| `Feedback` | Paragraph-level suggestions and "Overall improvements" lines | Only the *advice lines*; the `Original:`/`Suggested:` rewrites are deliberately ignored |

## Findings that shaped the pipeline

1. **Scores and explanations agree.** The band in the `Rater2_*` text equals the numeric `Rater1_*` band in 99.7% of
   2,260 criterion cases. The 2 essays whose explanation band differs from the numeric label are dropped (a Band-7 explanation under a Band-5
   label would teach the model contradictions), as are 2 essays whose human overall band is a full band away from the
   criteria average.
2. **Quotations are often not exact.** Of 5,747 quoted passages found in the explanations, 5,199 (90.5%) are verbatim in
   the essay, 300 (5.2%) were within a typo of the essay and were *snapped* to the real wording, and 248 (4.3%) were
   corrected/paraphrased versions - those are removed. (In the `Example:` lines specifically only 44% were verbatim.)
   Without this step the model would learn to "quote" text that is not there, which the objectives forbid.
3. **The `Feedback` rewrites are not usable.** They give the student finished sentences ("Suggested: ...") and some contain
   invented facts and stray non-English characters. That contradicts the goal of guiding the learner, so they are not
   trained on.
4. **No ready-made improvement suggestions exist.** Per-criterion suggestions are built from (a) hint templates tied to the
   weakness type and a quote that was cited as a problem, (b) the reviewer's own "Overall improvements" line and an
   "Improvement point" line. 5,032 suggestions in total, 69% distinct, the most frequent one makes up 5.6% of them.
   This is the weakest part of the data. `ielts_ft/enrich.py` + prompt E upgrade it with a stronger model.
5. **The overall score is not the IELTS average of the four criteria.** In 173 of 497 kept essays (35%) the human overall
   band is 0.5 lower than "mean of four criteria, rounded the IELTS way". The data is kept as given
   (`--overall-policy gold`). `--overall-policy rule` recomputes it if you prefer a strictly derivable overall. **Decision
   for you.**
6. **Narrow score range.** Overall bands: 2.0 (1), 4.0 (1), 4.5 (5), 5.0 (39), 5.5 (126), 6.0 (228), 6.5 (128), 7.0 (31),
   7.5 (5); nothing above 7.5. A model trained on this will be a good judge between 5 and 7 and unreliable outside it.
   The report prints an "always predict the median" baseline so you can see whether the model adds anything.
7. **80 essays (14%) are under 250 words.** They get `underlength_warning: true`, a Task Response weakness and priority 1.
   Word count and the flag are overwritten by code after generation, because LLMs miscount.
8. **Only 387 distinct raw prompt strings for 565 essays**; after normalising punctuation 325, and after merging
   near-duplicates (TF-IDF cosine >= 0.6) **276 prompt groups**. One prompt has 22 essays. Splitting by essay would have leaked
   questions across train and test.

## Result of the cleaning

| | Count |
|---|---|
| Rows in file / after removing exact duplicates | 565 / 564 |
| Records kept (all four criteria have strengths, weaknesses and verified evidence) | **497** |
| Dropped | 67 (4 score inconsistencies; 63 where some criterion had no verifiable strength, weakness or evidence) |

| Split | Essays | Prompt groups | Mean overall | Under 250 words |
|---|---|---|---|---|
| Train | 395 (79.5%) | 202 | 6.02 | 55 |
| Validation | 53 (10.7%) | 25 | 5.93 | 7 |
| Test | 49 (9.9%) | 25 | 5.92 | 9 |

The split is made on all 564 rows *before* dropping, so removing a record can never move a prompt between splits; the code
asserts that no prompt group appears in two splits.

Token lengths with the Qwen3.5 tokenizer: mean 1,785, 95th percentile 1,987, max 2,181 per example (705k tokens per epoch);
`max_seq_len` 2,560 is therefore enough and nothing is truncated.

## Honest limitations

* 395 training essays is small. Expect strong format learning, moderate scoring skill.
* The "reference" explanations look LLM-written (uniform phrasing), and the labels come from one rater column. The model
  learns to imitate that reference, not an official examiner. Validate on a handful of essays scored by a certified examiner.
* The test set has 49 essays: bootstrap 95% intervals are printed and are wide. Do not over-interpret differences smaller
  than about 0.15 bands of MAE.
* Suggestions are templated/assembled (see 4). Do not judge the *style* of feedback from the rule-based data alone.
