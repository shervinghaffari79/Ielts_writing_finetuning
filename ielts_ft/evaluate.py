"""Generate predictions for a dataset split and score them.

    python -m ielts_ft.evaluate --data data/processed/validation.jsonl --base Qwen/Qwen3.5-4B --adapter outputs/ielts-lora/adapter
    python -m ielts_ft.evaluate --data data/processed/validation.jsonl --base Qwen/Qwen3.5-4B --baseline      # no fine-tuning
    python -m ielts_ft.evaluate --data data/processed/test.jsonl --predictions preds.jsonl --final

The TEST file is protected: you must pass --final to touch it, to keep it untouched during tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import format_report_md, score_predictions
from .schema import extract_json, postprocess


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def evaluate(rows: list[dict], preds_raw: list[str], out_dir: str | None = None, title: str = "Evaluation") -> dict:
    preds = []
    for raw, r in zip(preds_raw, rows):
        obj = extract_json(raw)
        try:
            preds.append(postprocess(obj, r["meta"]["essay"]) if obj else None)
        except (KeyError, TypeError):
            preds.append(None)
    rep = score_predictions(preds, rows)
    if out_dir:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        (o / "eval_report.json").write_text(json.dumps(rep, indent=2, default=str))
        (o / "eval_report.md").write_text(format_report_md(rep, title))
        with open(o / "predictions.jsonl", "w", encoding="utf-8") as f:
            for raw, r in zip(preds_raw, rows):
                f.write(json.dumps({"id": r["id"], "raw": raw}, ensure_ascii=False) + "\n")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--baseline", action="store_true", help="un-tuned model with the long rubric prompt")
    ap.add_argument("--predictions", help="score an existing predictions.jsonl instead of generating")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--final", action="store_true", help="required to evaluate the held-out TEST split")
    a = ap.parse_args()
    if "test" in Path(a.data).name and not a.final:
        raise SystemExit("Refusing to touch the test split without --final (keep it unseen until model selection is done).")
    rows = read_jsonl(a.data)
    if a.limit:
        rows = rows[:a.limit]
    if a.predictions:
        by_id = {p["id"]: p["raw"] for p in read_jsonl(a.predictions)}
        raws = [by_id.get(r["id"], "") for r in rows]
    else:
        from .infer import Assessor
        ass = Assessor(a.base, a.adapter, baseline=a.baseline)
        raws = ass.generate([(r["meta"]["prompt"], r["meta"]["essay"]) for r in rows], batch_size=a.batch_size)
    rep = evaluate(rows, raws, a.out, title=f"Evaluation of {a.adapter or a.base} on {Path(a.data).name}")
    print(format_report_md(rep))


if __name__ == "__main__":
    main()
