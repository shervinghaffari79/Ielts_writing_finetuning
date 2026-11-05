# IELTS Writing Task 2 assessor - LoRA fine-tuning (open source) + OpenAI route

Stage 1 of an LLM system that guides English learners through IELTS Writing Task 2: a model that reads a **task prompt +
student essay** and returns a **strict JSON assessment** - four criterion bands, overall band, strengths, weaknesses,
*exact-quote evidence*, improvement hints, priority areas - ready for a later step-by-step coaching stage.

| You want to... | Go to |
|---|---|
| Run it on Colab (model/LoRA settings via widgets) | [`notebooks/IELTS_Qwen_LoRA_Colab.ipynb`](notebooks/IELTS_Qwen_LoRA_Colab.ipynb) |
| Understand it without a technical background | [`docs/GUIDE_FOR_BEGINNERS.md`](docs/GUIDE_FOR_BEGINNERS.md) |
| Design the next stage (coach, checker, guardrail **system prompts**) | [`docs/PROMPT_ENGINEERING.md`](docs/PROMPT_ENGINEERING.md) |
| Fine-tune with OpenAI + cost/time | [`docs/OPENAI_FINETUNING.md`](docs/OPENAI_FINETUNING.md) (**read the shutdown warning**) |
| See what is in the data and what was cleaned | [`docs/DATA_REPORT.md`](docs/DATA_REPORT.md) |

## Quick start (command line / Colab terminal)

```bash
pip install -U "transformers>=5" "peft>=0.17" accelerate datasets pandas openpyxl scikit-learn rapidfuzz pyyaml tiktoken
python -m ielts_ft.data_prep --xlsx data/IELTS_Task2_565_cleaned_minimal.xlsx --out data/processed --tokenizer Qwen/Qwen3.5-0.8B
python -m ielts_ft.train_lora --config configs/default.yaml --set model_name=Qwen/Qwen3.5-4B lora_r=16
python -m ielts_ft.evaluate --data data/processed/validation.jsonl --base Qwen/Qwen3.5-4B --adapter outputs/ielts-lora/adapter
python -m ielts_ft.evaluate --data data/processed/validation.jsonl --base Qwen/Qwen3.5-4B --baseline   # un-tuned comparison
python -m ielts_ft.infer --base Qwen/Qwen3.5-4B --adapter outputs/ielts-lora/adapter --prompt q.txt --essay e.txt
python -m ielts_ft.resources            # VRAM / time table per model and GPU
python -m pytest -q                     # 27 tests, no GPU needed
```

Ready-made configs: `configs/default.yaml`, `qwen38_27b_a100.yaml`, `qwen35_4b_t4.yaml` (T4-friendly), `smoke_cpu.yaml`.

## Layout

```
ielts_ft/
  schema.py      output contract, validator, quote check, JSON extraction, deterministic word-count fix
  parsing.py     spreadsheet text -> structured, quote-verified targets (+ hint bank, priorities, summary)
  data_prep.py   xlsx -> train/validation/test JSONL, grouped-by-prompt split, data_report.json
  prompts.py     short (training) and long (baseline) system prompts, message builder
  train_lora.py  LoRA training: rsLoRA, DoRA, PiSSA/OLoRA init, LoRA+, NEFTune, 8/4-bit, early stopping
  infer.py       Assessor class (generate + validate + retry); render.py learner-friendly view / step view
  metrics.py, evaluate.py   JSON validity, quote fabrication, MAE/QWK with bootstrap CI, repetition; sealed test split
  openai_ft.py   estimate cost/time, prepare, train, status, predict  (same eval as open-source track)
  enrich.py      optional: teacher-LLM upgrade of improvement_suggestions
  resources.py   VRAM/time estimator
```

## Design decisions worth knowing

* **Hybrid-architecture aware LoRA.** Qwen3.5/3.6/3.8 use Gated-DeltaNet layers for 3 of every 4 blocks. Their projections
  are named `in_proj_qkv`, `in_proj_z`, `out_proj`, not `q/k/v/o`. A "standard" q/k/v/o LoRA would adapt only 6 of 24 layers
  (0.8B model). The default preset `qwen35_all` covers all 24 (150 modules, verified).
* **DoRA guard.** In `Qwen3.5-0.8B`, `linear_attn.in_proj_qkv` of layer 0 has all-zero rows; DoRA divides by row norms and
  produced `NaN`. Such layers are skipped automatically. OLoRA + rsLoRA changed the model's output at step 0 (loss 4.4 -> 8.0),
  so rsLoRA is switched off for OLoRA. A first-batch loss check aborts before any GPU time is wasted on NaN.
* **bf16 LoRA, not QLoRA by default** (Unsloth advises against 4-bit for Qwen3.5-family; 4-bit remains available).
* **Thinking off.** The chat template is rendered with `enable_thinking=False` for training and inference: the JSON is the answer.
* **Loss only on the JSON answer**, never on the prompt/essay; examples longer than `max_seq_len` are skipped, never truncated.
* **Code, not the LLM, counts words** (`postprocess`), because the model cannot.
* **Test set is sealed**: `evaluate.py`/`openai_ft predict` refuse `test*` files without `--final`.
* **Field order is the one from the instruction document**. An alternative `reasoning_first` order (bands after analysis) is
  implemented in `schema.ordered` for an experiment, but not enabled.

## What was verified in this session - and what was not

Verified here (CPU only, no GPU or Colab connection was available in this environment):

* Data pipeline on the real spreadsheet: 497/564 records kept, every quote checked, no prompt group shared between splits.
* **Real training smoke test**: `Qwen/Qwen3.5-0.8B` + LoRA (r=8, 150 modules over 24 layers) for 3 optimiser steps on 6 examples
  on CPU - loss 1.38 -> 1.26 -> 1.07, validation loss 1.02, adapter saved (~80 s/step on 4 CPU cores without the fast kernels).
* Adapter reload + generation + JSON extraction + scoring path; DoRA, PiSSA, OLoRA, Gaussian, LoRA+ variants initialise and
  back-propagate on the real model; gold-as-prediction scores perfectly (MAE 0, 0 fabricated quotes).
* 27 unit tests (schema, parsing, split leakage, metrics, OpenAI estimator with a fake client, tokenisation masking).
* Notebook: generated by `scripts/build_notebook.py`; every code cell parses. It was **not executed on Colab**.

Not done: GPU runs, any real training, the 27B model, real OpenAI API calls (no key; also see the shutdown notice), teacher
enrichment against a real API. The first GPU run should therefore be the notebook's *smoke test*.

## Open questions for you

1. **Overall band**: keep the human `essay_score` (default; 35% of essays are 0.5 below the IELTS average of the four criteria) or
   recompute it from the criteria (`--overall-policy rule`)?
2. **OpenAI**: does your organisation still have fine-tuning access? If not, use the prompt-only baseline instead.
3. Can you get a few essays scored by a certified examiner to validate the reference scores?

Licence of the data: unspecified - the spreadsheet is not committed (`data/` is git-ignored).
