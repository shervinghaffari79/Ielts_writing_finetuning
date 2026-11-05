"""Back-of-the-envelope VRAM / time estimates for the Colab track.

These are ESTIMATES (typically within ~1.5x). The numbers for weights follow Unsloth's published bf16-LoRA
requirements for the Qwen3.5 family (0.8B:3GB, 2B:5GB, 4B:10GB, 9B:22GB, 27B:56GB) plus logits memory, which
is large here because the vocabulary has ~248k tokens.
"""
from __future__ import annotations

import math

# model -> (parameters in billions, hidden size, layers, vocab)
MODELS = {
    "Qwen/Qwen3.5-0.8B": (0.8, 1024, 24, 248_320), "Qwen/Qwen3.5-2B": (2.0, 2048, 24, 248_320),
    "Qwen/Qwen3.5-4B": (4.0, 2560, 32, 248_320), "Qwen/Qwen3.5-9B": (9.0, 4096, 32, 248_320),
    "Qwen/Qwen3.5-27B": (27.0, 5120, 64, 248_320), "Qwen/Qwen3.6-27B": (27.0, 5120, 64, 248_320),
    "Qwen/Qwen3.8-27B": (27.0, 5120, 64, 248_320),
}
# GPU -> (memory GB, peak dense bf16/fp16 TFLOPs, assumed model-FLOPs utilisation for HF Trainer + LoRA)
GPUS = {
    "T4 (16GB, fp16)": (16, 65, 0.20), "L4 (24GB)": (24, 121, 0.25), "A100 40GB": (40, 312, 0.30),
    "A100 80GB": (80, 312, 0.30), "H100 80GB": (80, 989, 0.20),
}
BYTES = {"none": 2.0, "8bit": 1.1, "4bit": 0.6}


def estimate_vram_gb(params_b: float, hidden: int, layers: int, vocab: int, seq: int, batch: int = 1,
                     quant: str = "none", lora_r: int = 16) -> float:
    weights = params_b * BYTES[quant]
    lora_and_optim = params_b * 0.012 * (lora_r / 16) * 4        # adapters + Adam states in fp32 (small)
    logits = batch * seq * vocab * 4 * 2 / 1e9                    # fp32 logits + their gradient
    ckpt = layers * batch * seq * hidden * 2 / 1e9                # one saved activation per layer (checkpointing)
    peak_layer = batch * seq * hidden * 2 * 40 / 1e9              # transient inside one layer
    return weights + lora_and_optim + logits + ckpt + peak_layer + 1.5


def estimate_hours(params_b: float, total_tokens_per_epoch: int, epochs: float, gpu: str, fast_kernels: bool) -> float:
    mem, tflops, mfu = GPUS[gpu]
    flops = 8 * params_b * 1e9 * total_tokens_per_epoch * epochs     # fwd + bwd + recompute for checkpointing
    slow = 1.0 if fast_kernels else 1.6                               # torch fallback for Gated-DeltaNet layers
    return flops / (tflops * 1e12 * mfu) / 3600 * slow


def table(model: str, seq: int, tokens_per_epoch: int, epochs: float, quant: str = "none", lora_r: int = 16,
          fast_kernels: bool = True) -> list[dict]:
    p, h, L, v = MODELS[model]
    need = estimate_vram_gb(p, h, L, v, seq, 1, quant, lora_r)
    rows = []
    for g, (mem, *_rest) in GPUS.items():
        rows.append({"gpu": g, "fits": need <= mem * 0.92, "vram_needed_gb": round(need, 1),
                     "hours": round(estimate_hours(p, tokens_per_epoch, epochs, g, fast_kernels), 2)})
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-per-epoch", type=int, default=715_000, help="from data_report.json (train_token_lengths.total)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--seq", type=int, default=2560)
    ap.add_argument("--quant", default="none")
    a = ap.parse_args()
    for m in MODELS:
        print(f"\n{m}")
        for r in table(m, a.seq, a.tokens_per_epoch, a.epochs, a.quant):
            print(f"  {r['gpu']:18s} need ~{r['vram_needed_gb']:5.1f} GB  {'fits' if r['fits'] else 'NO  '}  ~{r['hours']:.2f} h")


if __name__ == "__main__":
    main()
