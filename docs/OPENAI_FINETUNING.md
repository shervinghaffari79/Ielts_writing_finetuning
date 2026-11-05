# Fine-tuning the same assessor with OpenAI - instructions, cost and time

## 0. Read this first: the service is being shut down

Checked on **2026-11-05** against OpenAI's [deprecations page](https://developers.openai.com/api/docs/deprecations)
and [pricing page](https://developers.openai.com/api/docs/pricing):

| Date | What happens |
|---|---|
| 2026-05-07 | Organisations that **never** ran fine-tuning can no longer create fine-tuning jobs |
| 2026-07-02 | Creating jobs requires that the organisation **ran inference on a fine-tuned model in the last 60 days** |
| **2027-01-06** | **No** organisation can create new fine-tuning jobs |
| **2027-01-14** | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini` **and every fine-tune of them** are shut down (suggested replacement: `gpt-6-luna`, which cannot be fine-tuned) |

What this means for you:

* If your OpenAI organisation has never fine-tuned before, **you cannot do this at all** - the API will refuse.
* If it can, the resulting model would live for **about two months** at most. It is useful as a quick comparison
  ("how does the closed-source route compare?"), **not** as the foundation of a product.
* The pages above were read through an automated fetch; confirm the dates on the live pages before you plan around them.

**Recommended alternatives that follow the same approach**

1. **Open-source LoRA (this repository's main track).** Same data, same JSON, same evaluation - and you own the weights.
2. **Prompt-only with a current OpenAI model**: use the long prompt (`system_prompt_full()`), JSON-schema structured
   output and 3-5 validation-set examples as few-shot. Evaluate it with the identical script
   (`python -m ielts_ft.openai_ft predict --model <model> --long-system-prompt ...`). This is the honest baseline
   against which both fine-tuned routes should be judged.

Everything below applies **only if your organisation is still eligible**.

## 1. What is identical to the open-source route

| Item | Same? |
|---|---|
| Training file content (system / user / assistant messages) | Yes - `ielts_ft.openai_ft prepare` just removes the extra metadata keys |
| Loss only on the assistant answer | Yes - OpenAI supervised fine-tuning trains on assistant messages |
| Group-by-prompt 80/10/10 split, sealed test set | Yes |
| Evaluation (JSON validity, band errors, quote fabrication...) | Yes - `predict` writes the same `eval_report.md` |

What differs: you cannot choose LoRA rank, target layers, DoRA, etc. You only pick the base model and three
hyper-parameters (`n_epochs`, `batch_size`, `learning_rate_multiplier`; "auto" is the default).

## 2. Step by step

```bash
pip install openai tiktoken
export OPENAI_API_KEY=sk-...                      # in Colab: Secrets panel, then os.environ["OPENAI_API_KEY"] = ...
```

1. **Prepare the data** (after `python -m ielts_ft.data_prep`):
   ```bash
   python -m ielts_ft.openai_ft prepare --train data/processed/train.jsonl --val data/processed/validation.jsonl --out data/openai
   ```
   It checks every training line is valid JSON in the assistant message and has at least the 10 examples OpenAI requires.
2. **Estimate the cost first** (free, local):
   ```bash
   python -m ielts_ft.openai_ft estimate --train data/openai/train.jsonl --model gpt-4.1-mini --epochs 3
   ```
3. **Start the job and wait** (the script uploads both files and polls every 30 s):
   ```bash
   python -m ielts_ft.openai_ft train --train data/openai/train.jsonl --val data/openai/validation.jsonl \
          --model gpt-4.1-mini-2025-04-14 --epochs 3
   ```
   Output ends with the model id, e.g. `ft:gpt-4.1-mini-2025-04-14:your-org::abc123`.
   Check later with `... status --job ftjob-...`.
4. **Evaluate on validation** (never on test while tuning):
   ```bash
   python -m ielts_ft.openai_ft predict --data data/processed/validation.jsonl --model ft:gpt-4.1-mini-2025-04-14:your-org::abc123 --out outputs/openai_val
   ```
5. **Compare with the un-tuned model** using the long prompt: same command with the base model id and
   `--long-system-prompt`.
6. **Final test** once, with `--final` and the test file.
7. **Look at the training curve** in the OpenAI dashboard: if validation loss rises while training loss falls, use fewer
   epochs (2) next time.

Tips: keep temperature 0 for scoring; use `response_format={"type": "json_object"}` (already set in `predict`); fine-tuned
models cost more per token at inference than their base models (table below).

## 3. Cost estimate for this dataset

Measured with the real training file (395 examples, o200k_base token counts; the training file has **693,439 tokens**,
mean **496 prompt tokens** and **1,263 answer tokens** per essay). Training is billed as
*tokens in the training file x epochs*. Prices are USD per 1M tokens from OpenAI's pricing page (2026-11-05):

| Base model | Train $/1M | Fine-tuned input / output $/1M | **Training cost, 3 epochs** (2.08M billed tokens) | Cost per assessed essay | 1,000 essays |
|---|---|---|---|---|---|
| gpt-4o-mini-2024-07-18 | 3.00 | 0.30 / 1.20 | **$6.24** | $0.0017 | $1.66 |
| gpt-4.1-mini-2025-04-14 | 5.00 | 0.80 / 3.20 | **$10.40** | $0.0044 | $4.44 |
| gpt-4.1-2025-04-14 | 25.00 | 3.00 / 12.00 | **$52.01** | $0.0166 | $16.64 |
| gpt-4o-2024-08-06 | 25.00 | 3.75 / 15.00 | **$52.01** | $0.0208 | $20.80 |

Scale linearly with epochs (2 epochs = 2/3 of the cost). Add a small amount for validation evaluation calls and for
your own test runs: evaluating 50 essays on gpt-4.1-mini costs about $0.22.
Re-run `python -m ielts_ft.openai_ft estimate ...` after you change the data: the numbers are computed, not typed in.

## 4. Time estimate

OpenAI publishes no throughput numbers, so these are **assumptions you should replace after your first job**:

* Training once running: roughly 250k tokens/min for the "mini" models, ~90k tokens/min for the large ones
  -> **about 8 min (mini) / 23 min (4.1 / 4o)** of pure training for 2.08M billed tokens.
* Queue and validation: between a few minutes and a couple of hours, depending on load.
* Realistic total: **15-130 minutes** for the mini models, **30-150 minutes** for the large ones.
* Your own work: ~30 min to prepare and launch, ~15 min to evaluate.

For comparison, open-source LoRA estimates on Colab GPUs are in the notebook (section 4) and `python -m ielts_ft.resources`.

## 5. Risks and good practice

* **Do not send data you are not allowed to share.** Essays are uploaded to OpenAI; check consent/privacy rules.
  (Data sent to a third-party service may be retained according to its policy.)
* **Model retirement**: pin the exact snapshot; plan the migration the day you start.
* **Small data**: 395 examples is on the low side. Expect the model to learn the *format* well and the *scoring* only
  moderately; judge it by QWK/MAE against the "always predict the median" baseline printed in the report.
* **Don't tune on test.** The CLI refuses to read a file named `test*` without `--final`.
