"""LoRA fine-tuning of an open-source chat model (Hugging Face transformers + PEFT).

    python -m ielts_ft.train_lora --config configs/default.yaml --set model_name=Qwen/Qwen3.5-4B lora_r=32

Design choices (all of them are switchable in the config):
  * Plain `transformers.Trainer` + own tokenisation  -> few moving parts, stable across library versions.
  * Loss only on the assistant answer (the JSON), never on the prompt/essay ("completion-only").
  * Qwen3.5/3.6/3.8 are HYBRID models: 3 of every 4 layers are Gated-DeltaNet (`linear_attn.*`) layers whose
    projections are NOT called q/k/v/o. Targeting only q/k/v/o would silently adapt 25% of the layers, so the
    default target preset `qwen35_all` also covers `in_proj_qkv`, `in_proj_z`, `out_proj`.
  * State-of-the-art LoRA variants available: rsLoRA, DoRA, PiSSA/OLoRA/Gaussian init, LoRA+ (different LR for
    A and B), NEFTune noise, optional 8/4-bit quantisation (QLoRA).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------- configuration
TARGET_PRESETS = {
    "qwen35_all": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                   "gate_proj", "up_proj", "down_proj"],
    "attention_only": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"],
    "mlp_only": ["gate_proj", "up_proj", "down_proj"],
    "all_linear": None,        # every nn.Linear except lm_head / vision tower
}


@dataclass
class TrainConfig:
    # --- model & data
    model_name: str = "Qwen/Qwen3.5-4B"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/validation.jsonl"
    output_dir: str = "outputs/ielts-lora"
    max_seq_len: int = 3072
    enable_thinking: bool = False         # training/inference without a <think> block (faster, JSON only)
    trust_remote_code: bool = False
    # --- precision / memory
    precision: str = "auto"               # auto | bf16 | fp16 | fp32
    quantization: str = "none"            # none | 8bit | 4bit  (4bit = QLoRA; NOT recommended for Qwen3.5 family)
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    # --- LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_preset: str = "qwen35_all"     # see TARGET_PRESETS, or "custom"
    target_modules: list = field(default_factory=list)   # used when target_preset == "custom"
    use_rslora: bool = True               # rank-stabilised scaling alpha/sqrt(r)  (helps with r >= 16)
    use_dora: bool = False                # weight-decomposed LoRA (slower, sometimes better at low rank)
    init_lora_weights: str = "default"    # default | gaussian | pissa | pissa_niter_4 | olora
    loraplus_lr_ratio: float = 0.0        # 0 = off; 8-16 typical (LR of the B matrices = ratio * LR)
    # --- optimisation
    learning_rate: float = 1e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"            # adamw_torch | adamw_8bit | paged_adamw_8bit (needs bitsandbytes)
    max_grad_norm: float = 1.0
    neftune_noise_alpha: float = 0.0      # 0 = off; 5 is the common value
    early_stopping_patience: int = 2      # evaluations without improvement; 0 = off
    seed: int = 42
    # --- bookkeeping
    logging_steps: int = 5
    max_steps: int = -1                   # >0 overrides epochs (smoke tests)
    max_train_samples: int = 0            # >0 truncates the training set (smoke tests)
    max_eval_samples: int = 0
    save_merged: bool = False
    report_to: str = "none"

    def validate(self):
        if self.quantization == "4bit" and "Qwen3" in self.model_name:
            print("[warn] 4-bit QLoRA is not recommended for the Qwen3.5-family (larger quantisation error). "
                  "Prefer bf16 LoRA with a smaller model if memory is short.")
        if self.use_dora and self.quantization != "none":
            print("[warn] DoRA + quantisation is slow; consider turning one of them off.")
        if self.target_preset != "custom" and self.target_preset not in TARGET_PRESETS:
            raise ValueError(f"unknown target_preset {self.target_preset}")
        if self.lora_alpha <= 0 or self.lora_r <= 0:
            raise ValueError("lora_r and lora_alpha must be > 0")
        if self.init_lora_weights == "olora" and self.use_rslora:
            # measured: OLoRA + rsLoRA changes the model output at step 0 (loss 4.4 -> 8.0 on Qwen3.5-0.8B)
            print("[warn] OLoRA is not compatible with rsLoRA scaling; turning rsLoRA off.")
            self.use_rslora = False


def load_config(path: str | None, overrides: list[str] | None = None) -> TrainConfig:
    data: dict[str, Any] = {}
    if path:
        import yaml
        data = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        k, v = item.split("=", 1)
        data[k.strip()] = yaml_value(v)
    valid = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(data) - valid
    if unknown:
        raise SystemExit(f"Unknown config keys: {sorted(unknown)}")
    cfg = TrainConfig(**data)
    cfg.validate()
    return cfg


def yaml_value(v: str):
    import yaml
    return yaml.safe_load(v)


# ----------------------------------------------------------------------------- data
def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def render_prompt(tok, messages: list[dict], enable_thinking: bool = False) -> str:
    """Chat-template text of everything BEFORE the assistant answer (system + user + assistant header)."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)


def tokenize_example(tok, messages: list[dict], max_len: int, enable_thinking: bool = False):
    """-> (input_ids, labels) with labels = -100 on the prompt, or None if longer than max_len."""
    prompt = render_prompt(tok, messages[:-1], enable_thinking)
    answer = messages[-1]["content"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(answer, add_special_tokens=False).input_ids
    if len(p_ids) + len(a_ids) > max_len:
        return None
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids


class TokenizedDataset:
    def __init__(self, rows: list[dict], tok, max_len: int, enable_thinking: bool):
        self.items, self.skipped = [], 0
        for r in rows:
            t = tokenize_example(tok, r["messages"], max_len, enable_thinking)
            if t is None:
                self.skipped += 1
            else:
                self.items.append({"input_ids": t[0], "labels": t[1]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ----------------------------------------------------------------------------- model
def pick_dtype(precision: str):
    """CPU -> fp32. GPU: bf16 where supported (A100/L4/H100), else fp16 (T4)."""
    import torch
    if not torch.cuda.is_available() or precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_base(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"      # for generation; the training collator pads on the right itself
    dtype = pick_dtype(cfg.precision)
    kw: dict[str, Any] = dict(dtype=dtype, trust_remote_code=cfg.trust_remote_code)
    if cfg.attn_implementation:
        kw["attn_implementation"] = cfg.attn_implementation
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    if cfg.quantization in ("4bit", "8bit"):
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization == "4bit", load_in_8bit=cfg.quantization == "8bit",
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kw)
    return model, tok, dtype


def find_target_modules(model, cfg: TrainConfig) -> list[str]:
    """Full names of the nn.Linear layers that get LoRA adapters (vision tower and lm_head excluded)."""
    import torch
    suffixes = cfg.target_modules if cfg.target_preset == "custom" else TARGET_PRESETS[cfg.target_preset]
    names, skipped = [], []
    for n, m in model.named_modules():
        is_linear = isinstance(m, torch.nn.Linear) or m.__class__.__name__ in ("Linear4bit", "Linear8bitLt")
        if not is_linear or "lm_head" in n or re.search(r"visual|vision|embed", n):
            continue
        if suffixes is None or n.split(".")[-1] in suffixes:
            if cfg.use_dora and hasattr(m, "weight") and m.weight.dtype != torch.uint8 \
                    and bool((m.weight.detach().float().norm(dim=1) == 0).any()):
                # DoRA divides by each row's norm; Qwen3.5's linear_attn.in_proj_qkv has all-zero rows -> NaN
                skipped.append(n)
                continue
            names.append(n)
    if skipped:
        print(f"[lora] DoRA: skipped {len(skipped)} layers that contain all-zero weight rows (e.g. {skipped[0]})")
    if not names:
        raise SystemExit("No LoRA target layers found - check target_preset / target_modules")
    return names


def attach_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model
    targets = find_target_modules(model, cfg)
    init = {"default": True}.get(cfg.init_lora_weights, cfg.init_lora_weights)
    pcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=targets, use_rslora=cfg.use_rslora, use_dora=cfg.use_dora,
                      init_lora_weights=init, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, pcfg)
    layers = {m.group(1) for n in targets if (m := re.search(r"layers\.(\d+)\.", n))}
    print(f"[lora] {len(targets)} adapted modules across {len(layers)} transformer layers")
    model.print_trainable_parameters()
    return model


# ----------------------------------------------------------------------------- train
def train(cfg: TrainConfig) -> dict:
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    set_seed(cfg.seed)
    random.seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, tok, dtype = load_base(cfg)
    if cfg.quantization != "none":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = attach_lora(model, cfg)

    train_rows, val_rows = read_jsonl(cfg.train_file), read_jsonl(cfg.val_file)
    if cfg.max_train_samples:
        train_rows = train_rows[:cfg.max_train_samples]
    if cfg.max_eval_samples:
        val_rows = val_rows[:cfg.max_eval_samples]
    ds_tr = TokenizedDataset(train_rows, tok, cfg.max_seq_len, cfg.enable_thinking)
    ds_va = TokenizedDataset(val_rows, tok, cfg.max_seq_len, cfg.enable_thinking)
    print(f"[data] train={len(ds_tr)} (skipped {ds_tr.skipped} too long)  validation={len(ds_va)}")
    if not len(ds_tr):
        raise SystemExit("No training examples fit into max_seq_len")

    eff_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(ds_tr) / eff_batch))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else int(math.ceil(steps_per_epoch * cfg.num_train_epochs))
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    use_bf16 = dtype == torch.bfloat16 and torch.cuda.is_available()
    use_fp16 = dtype == torch.float16 and torch.cuda.is_available()
    do_eval = len(ds_va) > 0

    args_kw = dict(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps, learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs, max_steps=cfg.max_steps, lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=warmup, weight_decay=cfg.weight_decay, max_grad_norm=cfg.max_grad_norm,
        bf16=use_bf16, fp16=use_fp16, use_cpu=not torch.cuda.is_available(), optim=cfg.optim,
        logging_steps=cfg.logging_steps, report_to=cfg.report_to, seed=cfg.seed,
        remove_unused_columns=False, save_total_limit=2, dataloader_num_workers=0,
        eval_strategy="epoch" if do_eval and cfg.max_steps <= 0 else ("steps" if do_eval else "no"),
        save_strategy="epoch" if do_eval and cfg.max_steps <= 0 else ("steps" if do_eval else "no"),
        load_best_model_at_end=bool(do_eval and cfg.max_steps <= 0),
        metric_for_best_model="eval_loss", greater_is_better=False, save_only_model=False,
    )
    if cfg.max_steps > 0 and do_eval:
        args_kw["eval_steps"] = args_kw["save_steps"] = max(1, cfg.max_steps)
    if cfg.neftune_noise_alpha > 0:
        args_kw["neftune_noise_alpha"] = cfg.neftune_noise_alpha
    args = TrainingArguments(**args_kw)

    optimizers = (None, None)
    if cfg.loraplus_lr_ratio > 0:      # LoRA+: B matrices learn faster than A matrices
        from peft.optimizers import create_loraplus_optimizer
        opt = create_loraplus_optimizer(model=model, optimizer_cls=torch.optim.AdamW, lr=cfg.learning_rate,
                                        loraplus_lr_ratio=cfg.loraplus_lr_ratio, weight_decay=cfg.weight_decay)
        optimizers = (opt, None)
    callbacks = []
    if cfg.early_stopping_patience and do_eval and cfg.max_steps <= 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va if do_eval else None,
                      data_collator=Collator(tok.pad_token_id), callbacks=callbacks, optimizers=optimizers)
    # sanity check BEFORE spending GPU hours: the very first loss must be a finite number
    model.eval()
    with torch.no_grad():
        b = Collator(tok.pad_token_id)([ds_tr[0]])
        b = {k: v.to(model.device) for k, v in b.items()}
        init_loss = float(model(**b).loss)
    model.train()
    print(f"[check] initial loss on one training example: {init_loss:.3f}")
    if not math.isfinite(init_loss):
        raise SystemExit("Initial loss is NaN/inf. Typical causes: DoRA or PiSSA on a layer with zero rows, fp16 "
                         "overflow (try precision=bf16 or fp32), or a wrong chat template. Turn the fancy option off.")
    print(f"[train] effective batch={eff_batch} steps/epoch={steps_per_epoch} total_steps={total_steps} "
          f"warmup={warmup} dtype={dtype}")
    result = trainer.train()

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (out / "train_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    summary = {"train_loss": result.metrics.get("train_loss"), "steps": trainer.state.global_step,
               "train_runtime_s": result.metrics.get("train_runtime"),
               "log_history": trainer.state.log_history}
    if do_eval:
        summary["eval"] = trainer.evaluate()
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if cfg.save_merged and cfg.quantization == "none":
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(out / "merged")
        tok.save_pretrained(out / "merged")
    print(f"[done] adapter saved to {adapter_dir}")
    del trainer, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # free GPU memory so evaluation can load the model in the same Colab session
    return {"adapter_dir": str(adapter_dir), **{k: v for k, v in summary.items() if k != "log_history"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    a = ap.parse_args()
    train(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
