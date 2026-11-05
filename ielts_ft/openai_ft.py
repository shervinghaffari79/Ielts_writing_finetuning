"""OpenAI supervised fine-tuning track: same data, same prompts, same evaluation as the open-source track.

    python -m ielts_ft.openai_ft estimate --train data/processed/train.jsonl --model gpt-4.1-mini --epochs 3
    python -m ielts_ft.openai_ft prepare  --train data/processed/train.jsonl --val data/processed/validation.jsonl --out data/openai
    python -m ielts_ft.openai_ft train    --train data/openai/train.jsonl --val data/openai/validation.jsonl --model gpt-4.1-mini-2025-04-14
    python -m ielts_ft.openai_ft status   --job ftjob-...
    python -m ielts_ft.openai_ft predict  --data data/processed/validation.jsonl --model ft:gpt-4.1-mini-2025-04-14:org::id --out outputs/openai_val

!! AVAILABILITY WARNING (checked 2026-11-05 on OpenAI's deprecations page) !!
   * Since 2026-05-07 organisations that never fine-tuned cannot create fine-tuning jobs.
   * Since 2026-07-02 job creation needs fine-tuned-model inference within the past 60 days.
   * 2027-01-06: no new fine-tuning jobs for anyone.  2027-01-14: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini
     AND all their fine-tunes are shut down.
   Read docs/OPENAI_FINETUNING.md before spending money.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# USD per 1M tokens, standard tier. Source: developers.openai.com/api/docs/pricing (fetched 2026-11-05).
# ALWAYS re-check before spending: these change.
PRICES = {
    "gpt-4.1-mini-2025-04-14": {"train": 5.00, "in": 0.80, "cached_in": 0.20, "out": 3.20},
    "gpt-4.1-2025-04-14": {"train": 25.00, "in": 3.00, "cached_in": 0.75, "out": 12.00},
    "gpt-4o-mini-2024-07-18": {"train": 3.00, "in": 0.30, "cached_in": 0.15, "out": 1.20},
    "gpt-4o-2024-08-06": {"train": 25.00, "in": 3.75, "cached_in": 1.875, "out": 15.00},
}
ALIASES = {"gpt-4.1-mini": "gpt-4.1-mini-2025-04-14", "gpt-4.1": "gpt-4.1-2025-04-14",
           "gpt-4o-mini": "gpt-4o-mini-2024-07-18", "gpt-4o": "gpt-4o-2024-08-06"}
WARNING = ("WARNING: OpenAI is winding down fine-tuning. New organisations are blocked; existing ones can create "
           "jobs only until 2027-01-06, and these base models and their fine-tunes shut down 2027-01-14. "
           "See docs/OPENAI_FINETUNING.md.")

# Rough wall-clock speed. OpenAI does not publish throughput; these are order-of-magnitude assumptions
# (training tokens per minute once the job is running) that you should replace after your first job.
ASSUMED_TOKENS_PER_MIN = {"gpt-4.1-mini-2025-04-14": 250_000, "gpt-4o-mini-2024-07-18": 300_000,
                          "gpt-4.1-2025-04-14": 90_000, "gpt-4o-2024-08-06": 90_000}
ASSUMED_QUEUE_MIN = (5, 120)


def resolve_model(m: str) -> str:
    return ALIASES.get(m, m)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def count_tokens(messages: list[dict]) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")          # gpt-4o / gpt-4.1 family
    return sum(3 + len(enc.encode(m["content"])) for m in messages) + 3


def to_openai_rows(rows: list[dict]) -> list[dict]:
    """OpenAI wants only {"messages": [...]} per line (extra keys can make the upload fail)."""
    return [{"messages": r["messages"]} for r in rows]


def validate_rows(rows: list[dict]) -> list[str]:
    errs = []
    if len(rows) < 10:
        errs.append("OpenAI requires at least 10 training examples")
    for i, r in enumerate(rows):
        roles = [m["role"] for m in r["messages"]]
        if roles[-1] != "assistant":
            errs.append(f"row {i}: last message must be the assistant answer")
        try:
            json.loads(r["messages"][-1]["content"])
        except Exception:
            errs.append(f"row {i}: assistant content is not valid JSON")
        if count_tokens(r["messages"]) > 65_536:
            errs.append(f"row {i}: more than 65k tokens")
    return errs[:20]


def estimate(train_rows: list[dict], model: str, epochs: int = 3, n_infer: int = 1000,
             val_rows: list[dict] | None = None) -> dict:
    m = resolve_model(model)
    p = PRICES[m]
    per_example = [count_tokens(r["messages"]) for r in train_rows]
    train_tokens = sum(per_example)
    billed = train_tokens * epochs
    prompt_tok = [count_tokens(r["messages"][:-1]) for r in train_rows]
    out_tok = [count_tokens(r["messages"][-1:]) for r in train_rows]
    mean_in, mean_out = sum(prompt_tok) / len(prompt_tok), sum(out_tok) / len(out_tok)
    speed = ASSUMED_TOKENS_PER_MIN.get(m, 100_000)
    run_min = billed / speed
    est = {
        "model": m, "examples": len(train_rows), "epochs": epochs,
        "tokens_in_training_file": train_tokens, "billed_training_tokens": billed,
        "training_cost_usd": round(billed / 1e6 * p["train"], 2),
        "mean_prompt_tokens": round(mean_in), "mean_answer_tokens": round(mean_out),
        "cost_per_essay_usd": round(mean_in / 1e6 * p["in"] + mean_out / 1e6 * p["out"], 5),
        f"cost_{n_infer}_essays_usd": round(n_infer * (mean_in / 1e6 * p["in"] + mean_out / 1e6 * p["out"]), 2),
        "time_minutes_training_assumed": round(run_min, 1),
        "time_minutes_incl_queue_range": (round(run_min + ASSUMED_QUEUE_MIN[0]), round(run_min + ASSUMED_QUEUE_MIN[1])),
        "price_table_usd_per_1M": p,
    }
    if val_rows:
        # validation passes are not billed as training tokens separately in the docs, but add a safety margin
        est["validation_tokens"] = sum(count_tokens(r["messages"]) for r in val_rows)
    return est


def prepare(train: str, val: str, out: str) -> dict:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, src in (("train", train), ("validation", val)):
        rows = to_openai_rows(read_jsonl(src))
        errs = validate_rows(rows) if name == "train" else []
        if errs:
            raise SystemExit("Validation failed:\n - " + "\n - ".join(errs))
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        res[name] = len(rows)
    return res


def _client():
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first (in Colab: Secrets panel).")
    return OpenAI()


def start_training(client, train_path: str, val_path: str | None, model: str, epochs: int | str = "auto",
                   batch_size: int | str = "auto", lr_multiplier: float | str = "auto", seed: int = 42,
                   suffix: str = "ielts-t2") -> str:
    tf = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
    vf = client.files.create(file=open(val_path, "rb"), purpose="fine-tune") if val_path else None
    job = client.fine_tuning.jobs.create(
        training_file=tf.id, validation_file=vf.id if vf else None, model=resolve_model(model), seed=seed,
        suffix=suffix,
        method={"type": "supervised", "supervised": {"hyperparameters": {
            "n_epochs": epochs, "batch_size": batch_size, "learning_rate_multiplier": lr_multiplier}}})
    return job.id


def wait(client, job_id: str, poll: int = 30, verbose: bool = True):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        if verbose:
            print(time.strftime("%H:%M:%S"), job.status, getattr(job, "fine_tuned_model", None) or "")
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(poll)


def predict(client, rows: list[dict], model: str, out_dir: str, system: str | None = None) -> list[str]:
    """Run an OpenAI model (fine-tuned or not) over a split. Deterministic (temperature 0)."""
    raws = []
    for r in rows:
        msgs = [dict(m) for m in r["messages"][:-1]]
        if system:
            msgs[0]["content"] = system
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=2500,
                                              response_format={"type": "json_object"})
        raws.append(resp.choices[0].message.content)
    from .evaluate import evaluate
    evaluate(rows, raws, out_dir, title=f"Evaluation of {model}")
    return raws


def main():
    print(WARNING, "\n")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--train", required=True)
    e.add_argument("--model", default="gpt-4.1-mini")
    e.add_argument("--epochs", type=int, default=3)
    e.add_argument("--n-infer", type=int, default=1000)
    pr = sub.add_parser("prepare")
    pr.add_argument("--train", required=True)
    pr.add_argument("--val", required=True)
    pr.add_argument("--out", default="data/openai")
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val")
    t.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    t.add_argument("--epochs", default="3")
    t.add_argument("--batch-size", default="auto")
    t.add_argument("--lr-multiplier", default="auto")
    t.add_argument("--no-wait", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--job", required=True)
    pd_ = sub.add_parser("predict")
    pd_.add_argument("--data", required=True)
    pd_.add_argument("--model", required=True)
    pd_.add_argument("--out", default="outputs/openai_eval")
    pd_.add_argument("--long-system-prompt", action="store_true", help="use the full rubric prompt (for non-tuned models)")
    pd_.add_argument("--final", action="store_true")
    a = ap.parse_args()

    if a.cmd == "estimate":
        print(json.dumps(estimate(read_jsonl(a.train), a.model, a.epochs, a.n_infer), indent=2))
    elif a.cmd == "prepare":
        print(prepare(a.train, a.val, a.out))
    elif a.cmd == "train":
        cl = _client()
        num = lambda x: x if x == "auto" else (float(x) if "." in str(x) else int(x))
        jid = start_training(cl, a.train, a.val, a.model, num(a.epochs), num(a.batch_size), num(a.lr_multiplier))
        print("job:", jid)
        if not a.no_wait:
            job = wait(cl, jid)
            print("result:", job.status, job.fine_tuned_model)
    elif a.cmd == "status":
        cl = _client()
        job = cl.fine_tuning.jobs.retrieve(a.job)
        print(job.status, job.fine_tuned_model)
        for ev in list(cl.fine_tuning.jobs.list_events(a.job, limit=10)):
            print(" ", ev.message)
    elif a.cmd == "predict":
        if "test" in Path(a.data).name and not a.final:
            raise SystemExit("Pass --final to evaluate on the held-out test split.")
        from .prompts import system_prompt_full
        predict(_client(), read_jsonl(a.data), a.model, a.out, system_prompt_full() if a.long_system_prompt else None)


if __name__ == "__main__":
    main()
