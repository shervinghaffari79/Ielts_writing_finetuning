"""Generates notebooks/IELTS_Qwen_LoRA_Colab.ipynb  (run: python scripts/build_notebook.py)."""
import json
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# IELTS Writing Task 2 assessor - LoRA fine-tuning on Colab

**What this notebook does (in plain words):** it teaches an open-source language model (e.g. Qwen) to read an
IELTS Task 2 question + a student essay and answer with a *structured assessment*: a band score for each of the four
IELTS criteria, strengths, weaknesses, **exact quotes from the essay as evidence**, hints for improvement, and priorities.

You do **not** need to understand the code. Run the cells from top to bottom. The cells marked **SETTINGS** have
controls you can change. Everything else just runs.

**Before you start:** `Runtime -> Change runtime type -> GPU`.
* Free T4 (16 GB): pick `Qwen3.5-2B` or `0.8B`.
* L4 (24 GB) / A100-40GB: `Qwen3.5-4B` or `9B`.
* A100-80GB / H100 / 96 GB GPUs (Colab Pro+): `Qwen3.8-27B`.

Tick **Smoke test** in the settings first - it trains for a few steps just to prove everything is wired correctly.
""")

md("## 1. Check the GPU")
code("""
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout or "NO GPU - change the runtime type!")
print("torch", torch.__version__, "| bf16 supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
""")

md("""
## 2. Get the code and install the libraries
If the GitHub repository is private, add a Colab *Secret* named `GITHUB_TOKEN` (key icon on the left) with a token that
can read the repo. Alternatively, upload the project zip when asked.
""")
code("""
import os, sys, subprocess, pathlib
REPO = "shervinghaffari79/Ielts_writing_finetuning"
BRANCH = "claude/beautiful-maxwell-1132mn"   # change to "main" once the pull request is merged
WORK = pathlib.Path("/content/Ielts_writing_finetuning")

if not WORK.exists():
    try:
        from google.colab import userdata
        tok = userdata.get("GITHUB_TOKEN")
    except Exception:
        tok = None
    url = f"https://{tok + '@' if tok else ''}github.com/{REPO}.git"
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, str(WORK)], capture_output=True, text=True)
    if r.returncode != 0:
        print("git clone failed (private repo? wrong branch?). Upload the zip of the project instead:")
        from google.colab import files
        up = files.upload()
        zipname = next(iter(up))
        subprocess.run(["unzip", "-q", "-o", zipname, "-d", "/content/_zip"], check=True)
        inner = [p for p in pathlib.Path("/content/_zip").iterdir() if p.is_dir()]
        (inner[0] if inner else pathlib.Path("/content/_zip")).rename(WORK)
os.chdir(WORK); sys.path.insert(0, str(WORK))
print("working directory:", os.getcwd())
""")
code("""
# Qwen3.5/3.6/3.8 need transformers v5 (older versions cannot load them).
!pip -q install -U "transformers>=5" "peft>=0.17" accelerate datasets pandas openpyxl scikit-learn rapidfuzz pyyaml tiktoken ipywidgets bitsandbytes
# Optional speed-ups for the Gated-DeltaNet layers of Qwen3.5+ (can take a few minutes; training still works without them, just slower)
!pip -q install -U flash-linear-attention || echo "fla not installed - using slower PyTorch fallback"
!pip -q install causal-conv1d --no-build-isolation || echo "causal-conv1d not installed - using slower PyTorch fallback"
""")

md("""
## 3. Your data
Upload `IELTS_Task2_565_cleaned_minimal.xlsx` (or point `XLSX` to a file in Google Drive). The notebook then:
1. checks every quotation in the reference feedback against the essay (only exact quotes survive),
2. converts everything to the fixed JSON format,
3. splits **80/10/10 by exam question** (all essays for the same or very similar question go to the same split).
""")
code("""
import os, json, pathlib
XLSX = "/content/IELTS_Task2_565_cleaned_minimal.xlsx"
if not os.path.exists(XLSX):
    from google.colab import files
    up = files.upload()                    # choose the .xlsx
    name = next(iter(up)); os.rename(name, XLSX)
from ielts_ft.data_prep import prepare
report = prepare(XLSX, "data/processed", tokenizer_name=None)
print(json.dumps({k: report[k] for k in ("rows_after_dedupe", "records_kept", "records_dropped", "quotes_in_source", "splits")}, indent=1))
""")

md("""
## 4. SETTINGS - choose the model and the LoRA options
*Hover over a control for a hint.* Defaults are a safe starting point.

| Setting | Plain-language meaning |
|---|---|
| Model | The base model that will be adapted. Bigger = better but needs a bigger GPU. |
| LoRA rank *r* | How much new "capacity" the adapter has. 8-16 is usually enough; 32-64 for a big model/data. |
| LoRA alpha | Strength of the adapter. Common rule: alpha = 2 x rank. |
| rsLoRA | A stabilised scaling that makes higher ranks train better. Keep on. |
| DoRA | A newer LoRA variant, often a bit more accurate, ~20-30% slower. |
| Init | How the adapter starts. `pissa` can converge faster, `default` is the safe choice. |
| LoRA+ ratio | Lets one half of the adapter learn faster (try 8-16). 0 = off. |
| NEFTune | Adds a little noise while training; can reduce over-fitting on small data. |
| Epochs / learning rate | How many passes over the data / how big each learning step is. |
| Smoke test | Trains for ~5 steps on 16 examples. Use it first to check everything works. |
""")
code("""
import ipywidgets as w
from IPython.display import display
MODELS = ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B",
          "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-0.8B", "(custom - type below)"]
S = lambda **k: dict(style={"description_width": "170px"}, layout=w.Layout(width="520px"), **k)
W = dict(
    model=w.Dropdown(options=MODELS, value="Qwen/Qwen3.5-4B", description="Model", **S()),
    custom=w.Text(value="", description="Custom HF model id", **S()),
    precision=w.Dropdown(options=["auto", "bf16", "fp16", "fp32"], value="auto", description="Precision", **S()),
    quant=w.Dropdown(options=["none", "8bit", "4bit"], value="none", description="Quantisation (QLoRA)", **S()),
    r=w.IntSlider(value=16, min=4, max=128, step=4, description="LoRA rank r", **S()),
    alpha=w.IntSlider(value=32, min=4, max=256, step=4, description="LoRA alpha", **S()),
    dropout=w.FloatSlider(value=0.05, min=0, max=0.3, step=0.01, description="LoRA dropout", **S()),
    targets=w.Dropdown(options=["qwen35_all", "attention_only", "mlp_only", "all_linear"], value="qwen35_all", description="Layers to adapt", **S()),
    rslora=w.Checkbox(value=True, description="rsLoRA", **S()),
    dora=w.Checkbox(value=False, description="DoRA", **S()),
    init=w.Dropdown(options=["default", "gaussian", "pissa", "pissa_niter_4", "olora"], value="default", description="Adapter init", **S()),
    lpp=w.FloatText(value=0.0, description="LoRA+ ratio (0=off)", **S()),
    lr=w.FloatText(value=1e-4, description="Learning rate", **S()),
    epochs=w.FloatText(value=3, description="Epochs", **S()),
    bs=w.IntText(value=1, description="Batch size / GPU", **S()),
    ga=w.IntText(value=16, description="Gradient accumulation", **S()),
    seq=w.IntText(value=2560, description="Max tokens / example", **S()),
    neft=w.FloatText(value=0.0, description="NEFTune alpha (0=off)", **S()),
    smoke=w.Checkbox(value=True, description="Smoke test (5 steps, 16 examples)", **S()),
)
display(w.VBox(list(W.values())))
""")
code("""
from ielts_ft.train_lora import TrainConfig
from ielts_ft import resources
def make_config():
    name = W["custom"].value.strip() if W["model"].value.startswith("(custom") else W["model"].value
    cfg = TrainConfig(model_name=name, precision=W["precision"].value, quantization=W["quant"].value,
        lora_r=W["r"].value, lora_alpha=W["alpha"].value, lora_dropout=W["dropout"].value,
        target_preset=W["targets"].value, use_rslora=W["rslora"].value, use_dora=W["dora"].value,
        init_lora_weights=W["init"].value, loraplus_lr_ratio=W["lpp"].value, learning_rate=W["lr"].value,
        num_train_epochs=W["epochs"].value, per_device_train_batch_size=W["bs"].value,
        gradient_accumulation_steps=W["ga"].value, max_seq_len=W["seq"].value, neftune_noise_alpha=W["neft"].value,
        output_dir="/content/outputs/" + name.split("/")[-1] + ("-smoke" if W["smoke"].value else ""))
    if W["smoke"].value:
        cfg.max_steps, cfg.max_train_samples, cfg.max_eval_samples, cfg.early_stopping_patience = 5, 16, 4, 0
        cfg.gradient_accumulation_steps = 2
    cfg.validate(); return cfg
cfg = make_config()
print("Chosen model:", cfg.model_name)
if cfg.model_name in resources.MODELS:
    tok_total = json.load(open("data/processed/data_report.json")).get("train_token_lengths", {}).get("total", 715_000)
    for row in resources.table(cfg.model_name, cfg.max_seq_len, tok_total, cfg.num_train_epochs, cfg.quantization, cfg.lora_r, fast_kernels=False):
        print(f"  {row['gpu']:18s} needs ~{row['vram_needed_gb']:5.1f} GB  {'fits' if row['fits'] else 'DOES NOT FIT':12s} est. {row['hours']:.1f} h")
""")

md("""
## 5. Train
With *Smoke test* ticked this takes a few minutes and only proves the pipeline works. Untick it (and re-run the two cells
above) for a real run. The loss should go **down**; if it jumps to `nan`, lower the learning rate.
""")
code("""
from ielts_ft.train_lora import train
cfg = make_config()
result = train(cfg)
result
""")

md("""
## 6. Check quality on the VALIDATION set
Compares the fine-tuned model with the *un-tuned* model that is given the long rubric as instructions. The
**test set is not touched here**.
""")
code("""
from ielts_ft.evaluate import evaluate, read_jsonl
from ielts_ft.infer import Assessor
from ielts_ft.metrics import format_report_md
N = 8 if W["smoke"].value else 0         # 0 = whole validation set
val = read_jsonl("data/processed/validation.jsonl")[: N or None]
ft = Assessor(cfg.model_name, f"{cfg.output_dir}/adapter")
raws = ft.generate([(r["meta"]["prompt"], r["meta"]["essay"]) for r in val], batch_size=2)
rep = evaluate(val, raws, f"{cfg.output_dir}/eval_val", title="Fine-tuned model - validation")
print(format_report_md(rep))
""")
code("""
# OPTIONAL: baseline = same model WITHOUT fine-tuning, given the long rubric prompt. Shows what fine-tuning adds.
del ft; import gc, torch; gc.collect(); torch.cuda.empty_cache()
base = Assessor(cfg.model_name, None, baseline=True)
raws_b = base.generate([(r["meta"]["prompt"], r["meta"]["essay"]) for r in val], batch_size=2)
print(format_report_md(evaluate(val, raws_b, f"{cfg.output_dir}/eval_val_baseline", title="Un-tuned baseline - validation")))
del base; gc.collect(); torch.cuda.empty_cache()
""")

md("## 7. Try it on your own essay")
code("""
from ielts_ft.render import render_report, render_step
ft = Assessor(cfg.model_name, f"{cfg.output_dir}/adapter")
question = w.Textarea(value="Some people think that university education should be free. To what extent do you agree or disagree?", description="Question", layout=w.Layout(width="95%", height="80px"))
essay_box = w.Textarea(value="Paste the student's essay here...", description="Essay", layout=w.Layout(width="95%", height="260px"))
display(question, essay_box)
""")
code("""
res = ft.assess(question.value, essay_box.value)
print("validator:", res.errors or "all checks passed")
if res.assessment:
    print(render_report(res.assessment))
    print("\\n--- step-by-step view (what a learner would see first) ---")
    print(render_step(res.assessment, 1))
else:
    print(res.raw)
""")

md("""
## 8. Save the adapter
The adapter is small (tens to hundreds of MB). Save it to Google Drive (and/or the Hugging Face Hub).
""")
code("""
from google.colab import drive
drive.mount("/content/drive")
import shutil
dst = "/content/drive/MyDrive/ielts_adapters/" + pathlib.Path(cfg.output_dir).name
shutil.copytree(cfg.output_dir, dst, dirs_exist_ok=True); print("saved to", dst)
# Hugging Face Hub (optional): from huggingface_hub import login; login(); 
#   from peft import PeftModel; ft.model.push_to_hub("your-name/ielts-assessor-lora")
""")

md("""
## 9. FINAL test (do this ONCE, after you have finished choosing the model and settings)
The test set must stay unseen while you tune. Set `I_AM_DONE_TUNING = True` only for your final run.
""")
code("""
I_AM_DONE_TUNING = False
if I_AM_DONE_TUNING:
    test = read_jsonl("data/processed/test.jsonl")
    raws_t = ft.generate([(r["meta"]["prompt"], r["meta"]["essay"]) for r in test], batch_size=2)
    print(format_report_md(evaluate(test, raws_t, f"{cfg.output_dir}/eval_test", title="FINAL test")))
else:
    print("Test set left untouched.")
""")

md("""
## 10. OpenAI route: cost & time estimate (no API key needed)
**Read `docs/OPENAI_FINETUNING.md` first: OpenAI is closing its fine-tuning service.**
""")
code("""
from ielts_ft.openai_ft import estimate, read_jsonl as rj, WARNING
print(WARNING)
rows = rj("data/processed/train.jsonl")
for m in ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"]:
    e = estimate(rows, m, epochs=3)
    print(f"{m:14s} training cost ${e['training_cost_usd']:>6}  ({e['billed_training_tokens']:,} tokens)  "
          f"per essay ${e['cost_per_essay_usd']}  time ~{e['time_minutes_incl_queue_range'][0]}-{e['time_minutes_incl_queue_range'][1]} min")
""")

nb["cells"] = C
nb["metadata"] = {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "IELTS_Qwen_LoRA_Colab.ipynb"
nbf.write(nb, out)
print("wrote", out)
