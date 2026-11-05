# How the system works - a guide for non-experts

You do not need to know programming or machine learning to follow this guide. Technical words are explained the first
time they appear and collected in the glossary at the end.

---

## 1. The big picture

**Goal.** A learner writes an IELTS Writing Task 2 essay. The system reads the question and the essay and answers with a
structured assessment, so that a *later* part of the system (the "coach") can guide the learner step by step instead of
simply giving corrected text.

This repository builds **Stage 1: the assessor** - the part that *understands the essay and diagnoses it*. It does **not**
chat with the learner yet; the coach is built with prompts (see `PROMPT_ENGINEERING.md`).

```
  Exam question ─┐
                 ├─►  ASSESSOR (fine-tuned AI model)  ─►  Assessment (JSON)  ─►  Coach (Stage 2) ─► Learner
  Student essay ─┘        trained in this repository         bands, strengths,        step by step,
                                                              weaknesses, quotes,      one hint at a time
                                                              hints, priorities
```

**What the assessor returns** (always the same fields, in the same order):

| Part | Plain meaning |
|---|---|
| `overall_band` | Estimated total score (0-9, in steps of 0.5) |
| `word_count`, `underlength_warning` | Words written; warning if under 250 |
| 4 x criterion block | For Task Response, Coherence & Cohesion, Lexical Resource, Grammatical Range & Accuracy: band, strengths, weaknesses, **evidence (exact quotes from the essay)**, improvement hints |
| `priority_areas` | The most important problems, in the order to work on them (max. 8) |
| `overall_feedback` | A short summary |

**Why JSON?** JSON is a strict, machine-readable format. Because the fields never change, other programs (the coach,
a score dashboard) can rely on them.

**Why hints instead of rewrites?** Learners improve most when *they* fix the problem. The assessor is therefore trained to
write hints ("check the verb in *this* sentence") and never to hand over a corrected version.

---

## 2. What is "fine-tuning"?

A large language model already knows English and a little about IELTS, but it does not reliably produce *our* format,
use *our* scoring scale, or stay honest about quotes. **Fine-tuning** means showing it several hundred examples of the
perfect answer until it imitates them.

Think of a bright new teacher who has read every textbook. Fine-tuning is their first month shadowing an experienced
examiner: same knowledge, but now calibrated to the house rules.

### What is LoRA?

Retraining all billions of numbers inside a model needs enormous memory. **LoRA** (Low-Rank Adaptation) freezes the
model and adds small "adapter" layers - typically under 1% of the size - and trains only those. Result: much cheaper, and
the finished product is a small file (the *adapter*) that is plugged into the original model.

The notebook offers several modern LoRA refinements; the defaults are sensible, and you can ignore the rest:

| Option | In one sentence |
|---|---|
| **rank (r)** | How big the adapter is. Bigger learns more but can over-fit; 16 is a good start. |
| **alpha** | How loud the adapter speaks. Rule of thumb: 2 x rank. |
| **rsLoRA** | A stabilising trick so higher ranks work well. On by default. |
| **DoRA** | A newer variant that separates "direction" from "size" of the weights. A bit slower, sometimes more accurate. |
| **PiSSA / OLoRA / Gaussian init** | Different ways to *start* the adapter. Try only after the default works. |
| **LoRA+** | Lets one half of the adapter learn faster. Optional speed-up. |
| **NEFTune** | Adds a little noise during training to reduce memorising. Optional. |
| **Quantisation (QLoRA)** | Stores the model with fewer bits to save memory. Less accurate; not recommended for the Qwen3.5 family. |

---

## 3. The data and how it is prepared

You supplied 565 essays. Raw data is never fed straight into training. The program (`ielts_ft/data_prep.py`):

1. **Reads the scores** from the numeric columns (these are the "truth" for bands).
2. **Reads the explanations** and keeps only sentences that are consistent with those scores.
3. **Checks every quotation** against the essay. Quotes that are not in the essay are fixed (if only a typo differs) or
   removed. This is how we make sure the model learns *never to invent text*.
4. **Removes giveaway rewrites** from the old feedback (they would teach the model to hand over answers).
5. **Builds the JSON answer** for each essay in the exact format.
6. **Splits 80% / 10% / 10% into Train / Validation / Test - by exam question**, not by essay. If ten students answered
   the same question, all ten go into the same part. Otherwise the model could "recognise" a question it has seen and
   look smarter than it really is. Very similar questions are kept together too.

Three sets, three jobs:

| Set | Job | Analogy |
|---|---|---|
| **Train** (80%) | The model learns from it | Practice exercises |
| **Validation** (10%) | You check progress and choose settings | Mock exam |
| **Test** (10%) | Used once at the very end | The real exam |

The test set is locked: the evaluation tool refuses to open it unless you add `--final`.

The full findings are in `DATA_REPORT.md`. The key limitation: almost all essays are Band 5-7, so the model will not be
reliable for very weak or very strong essays.

---

## 4. Step-by-step: running it on Google Colab

Colab is Google's free/paid online computer with a GPU (a graphics chip that makes training fast).

1. Open `notebooks/IELTS_Qwen_LoRA_Colab.ipynb` in Colab (File -> Open notebook -> GitHub tab, or upload it).
2. **Runtime -> Change runtime type -> GPU.** Choose the best GPU your account offers.
3. Run the cells **from top to bottom** (click the play button, or Shift+Enter).
   * *Cell 2* downloads the code and installs libraries (a few minutes).
   * *Cell 3* asks you to upload the Excel file.
   * *Cell 4* shows the control panel: pick the model and options.
4. **First run with "Smoke test" ticked.** It trains for only a few steps (minutes). If it finishes without errors, the
   whole chain works.
5. Untick *Smoke test* and re-run the settings cells and the training cell for the real run.
6. *Section 6* tests the model on the validation set and compares it with the same model **without** fine-tuning.
7. *Section 7* lets you paste any essay and see the assessment.
8. *Section 8* saves the adapter to your Google Drive. Do this before closing Colab - Colab erases its disk.
9. *Section 9* (final test) - once, at the very end.

### Which model fits which GPU?

| GPU you got on Colab | Choose | Why |
|---|---|---|
| T4 (16 GB), free tier | Qwen3.5-0.8B or 2B | Larger ones do not fit; T4 uses 16-bit "fp16" |
| L4 (24 GB) | Qwen3.5-4B | |
| A100 40 GB | Qwen3.5-9B | |
| A100 80 GB / H100 / 96 GB | Qwen3.8-27B (or 3.6/3.5-27B) | Needs ~55-70 GB |

The settings cell prints an estimate of memory and time for the model you picked. Estimates are rough (within about
two times); the first real run tells you the truth.

**About "Qwen3.8":** the smaller models in the notebook list are the Qwen3.5 family (0.8B, 2B, 4B, 9B) because, at the time
of writing, the Qwen3.8 release on Hugging Face is the 27B model only. They share the same architecture, so the same code
works. Type any other Hugging Face model id in the "custom" box to try it.

---

## 5. Reading the results

After validation you get a table (`eval_report.md`). What to look at:

| Number | Good looks like | What it tells you |
|---|---|---|
| Valid JSON / follows schema | ~100% | The model reliably uses the format |
| **Fabricated-quote rate** | close to 0% | The model does not invent text. Most important honesty check |
| MAE (average distance in bands) | below the "always-median" MAE shown next to it | Whether it beats a lazy guess |
| Within 0.5 band | as high as possible (e.g. > 75%) | Practical accuracy |
| QWK (agreement score, 0-1) | > 0.5 is usable, > 0.7 good | Agreement with the reference, beyond chance |
| Unique improvement suggestions | high | Feedback is not repetitive |

With only ~50 validation essays, results move by a few points just by chance; the report shows 95% intervals.

### If something looks wrong

| You see | Likely reason | Try |
|---|---|---|
| "CUDA out of memory" | Model too big for the GPU | Smaller model, lower *Max tokens*, batch size 1, Runtime -> Restart |
| Loss becomes `nan` | Learning rate too high / fp16 overflow | Lower the learning rate to 5e-5, use bf16 if the GPU supports it |
| Valid JSON is low | Too little training | More epochs; check you did not tick the smoke test |
| Validation loss goes up while training loss goes down | Over-fitting (memorising) | Fewer epochs, higher LoRA dropout, lower rank, NEFTune 5 |
| All scores are 6.0 | Model learnt the average only | More epochs, bigger model; look at per-criterion MAE vs. baseline |
| Training is very slow | Fast kernels not installed | Run the optional `flash-linear-attention` / `causal-conv1d` install cell |
| "git clone failed" | Private repository | Add a `GITHUB_TOKEN` secret in Colab or upload the project zip |

---

## 6. The OpenAI route in one paragraph

The same data can fine-tune an OpenAI model (`docs/OPENAI_FINETUNING.md`). You would pay roughly **$6-10** for the
training of a "mini" model, plus fractions of a cent per assessed essay. **But** OpenAI is closing fine-tuning: new
organisations are already blocked, and every fine-tuned model of these generations stops working on **14 January 2027**.
Treat that route as an optional comparison, not as the foundation.

---

## 7. Safe-use rules

* The assessor gives an **estimate**, not an official IELTS score. Say so to learners.
* Do not upload essays that you do not have permission to use; the OpenAI route sends them to a third party.
* Always keep the Test set unseen until the very end.
* Re-check the model if you change the prompt, the data or the base model.
* Keep a human examiner in the loop for high-stakes decisions.

---

## Glossary

| Term | Meaning |
|---|---|
| **Adapter** | The small file produced by LoRA; plugged into the base model |
| **Band** | IELTS score from 0 to 9 in steps of 0.5 |
| **Base model** | The pre-trained model you start from (e.g. Qwen3.5-4B) |
| **Epoch** | One full pass through the training data |
| **Evidence** | An exact quote from the essay that supports a statement |
| **Fine-tuning** | Further training of a model on your own examples |
| **GPU / VRAM** | The chip / memory used for training; VRAM is the limiting resource |
| **Hallucination** | The model stating something that is not true (e.g. quoting words the student never wrote) |
| **JSON** | A strict text format with named fields |
| **Learning rate** | How big each learning step is. Too high: unstable. Too low: slow |
| **Loss** | A number measuring how wrong the model is. It should fall during training |
| **MAE / QWK** | Average error in bands / agreement score (see section 5) |
| **Over-fitting** | Memorising the training examples instead of learning the skill |
| **Prompt** | The instruction text given to a model |
| **System prompt** | The standing instructions that define the model's role |
| **Token** | A word piece; models read and bill in tokens (~0.75 words each) |
| **Quantisation** | Storing numbers with fewer bits to save memory |
