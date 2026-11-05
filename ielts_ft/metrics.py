"""Evaluation metrics: does the model score like the reference AND obey the output rules?"""
from __future__ import annotations

import math
import random
from collections import Counter

from .schema import CRITERIA, SHORT_OF, overall_from_criteria, quote_in_essay, validate_assessment

KEYS = ("overall",) + CRITERIA
GOLD_KEY = {"overall": "overall", **{c: c for c in CRITERIA}}


def _qwk(a: list[float], b: list[float]) -> float:
    from sklearn.metrics import cohen_kappa_score
    ai, bi = [int(round(x * 2)) for x in a], [int(round(x * 2)) for x in b]
    if len(set(ai + bi)) < 2:
        return float("nan")
    return float(cohen_kappa_score(ai, bi, weights="quadratic", labels=list(range(0, 19))))


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def _band_stats(pred: list[float], gold: list[float]) -> dict:
    n = len(pred)
    err = [p - g for p, g in zip(pred, gold)]
    return {"n": n, "exact": sum(e == 0 for e in err) / n, "within_0.5": sum(abs(e) <= 0.5 for e in err) / n,
            "within_1": sum(abs(e) <= 1 for e in err) / n, "mae": sum(abs(e) for e in err) / n,
            "bias": sum(err) / n, "qwk": _qwk(pred, gold), "pearson": _pearson(pred, gold)}


def _bootstrap(pred, gold, fn, n_boot=400, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(pred)))
    vals = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        v = fn([pred[i] for i in s], [gold[i] for i in s])
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if len(vals) > 20 else (float("nan"),) * 2


def score_predictions(preds: list[dict | None], rows: list[dict]) -> dict:
    """`rows` are dataset records (with meta.gold / meta.essay); preds[i] is the parsed assessment or None."""
    n = len(rows)
    rep: dict = {"n_essays": n}
    parsed = [p is not None for p in preds]
    rep["json_parse_rate"] = sum(parsed) / n
    errs = [validate_assessment(p, r["meta"]["essay"]) if p else ["unparsed"] for p, r in zip(preds, rows)]
    structural = [[e for e in es if "far from criteria mean" not in e and "not found in essay" not in e]
                  for es in errs]
    rep["schema_valid_rate"] = sum(not s for s in structural) / n
    rep["fully_valid_rate"] = sum(not e for e in errs) / n

    # --- quote fidelity ("must not invent a sentence/error/example")
    tot_q = bad_q = 0
    clean_resp = 0
    for p, r in zip(preds, rows):
        if not p:
            continue
        bad_here = 0
        quotes = [q for c in CRITERIA for q in p["criteria"].get(c, {}).get("evidence", []) if isinstance(q, str)]
        quotes += [x["evidence"] for x in p.get("priority_areas", []) if isinstance(x, dict) and x.get("evidence")]
        for q in quotes:
            tot_q += 1
            if not quote_in_essay(q, r["meta"]["essay"]):
                bad_q += 1
                bad_here += 1
        clean_resp += bad_here == 0
    rep["quotes_total"] = tot_q
    rep["quote_fabrication_rate"] = bad_q / tot_q if tot_q else float("nan")
    rep["responses_without_fabricated_quotes"] = clean_resp / n

    # --- scores (only on parsed outputs; unparsed counted separately above)
    ok = [(p, r) for p, r in zip(preds, rows) if p and all(c in p.get("criteria", {}) for c in CRITERIA)]
    rep["bands"] = {}
    if ok:
        for k in KEYS:
            pr = [float(p["overall_band"] if k == "overall" else p["criteria"][k]["band"]) for p, _ in ok]
            gd = [float(r["meta"]["gold"][k]) for _, r in ok]
            st = _band_stats(pr, gd)
            if len(pr) >= 20:
                st["mae_ci95"] = _bootstrap(pr, gd, lambda a, b: sum(abs(x - y) for x, y in zip(a, b)) / len(a))
                st["qwk_ci95"] = _bootstrap(pr, gd, _qwk)
            const = sorted(gd)[len(gd) // 2]
            st["baseline_mae_always_median"] = sum(abs(const - g) for g in gd) / len(gd)
            rep["bands"][k] = st
        rep["overall_consistent_with_criteria"] = sum(
            abs(p["overall_band"] - overall_from_criteria([p["criteria"][c]["band"] for c in CRITERIA])) <= 0.5
            for p, _ in ok) / len(ok)
        # --- repetition ("feedback should not be general or repetitive")
        sug = [s for p, _ in ok for c in CRITERIA for s in p["criteria"][c].get("improvement_suggestions", [])]
        cnt = Counter(sug)
        rep["suggestion_uniqueness"] = len(cnt) / len(sug) if sug else float("nan")
        rep["most_repeated_suggestion_share"] = (cnt.most_common(1)[0][1] / len(sug)) if sug else float("nan")
        rep["priority_count_mean"] = sum(len(p["priority_areas"]) for p, _ in ok) / len(ok)
        short = [(p, r) for p, r in ok if r["meta"]["word_count"] < 250]
        if short:
            rep["underlength_cases"] = len(short)
            rep["underlength_flagged"] = sum(p["underlength_warning"] for p, _ in short) / len(short)
    rep["common_errors"] = dict(Counter(e.split(":")[0][:50] for es in errs for e in es).most_common(8))
    return rep


def format_report_md(rep: dict, title: str = "Evaluation") -> str:
    f = lambda x: "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
    L = [f"# {title}", "", f"Essays: {rep['n_essays']}", "",
         "| Check | Value |", "|---|---|",
         f"| Valid JSON | {f(rep['json_parse_rate'])} |", f"| Follows the schema | {f(rep['schema_valid_rate'])} |",
         f"| Fully valid (schema + quotes + word count) | {f(rep['fully_valid_rate'])} |",
         f"| Fabricated-quote rate (lower is better) | {f(rep['quote_fabrication_rate'])} |",
         f"| Responses with no fabricated quote | {f(rep['responses_without_fabricated_quotes'])} |",
         f"| Overall within 0.5 of the criteria average | {f(rep.get('overall_consistent_with_criteria'))} |",
         f"| Unique improvement suggestions | {f(rep.get('suggestion_uniqueness'))} |", "",
         "| Criterion | Exact | Within 0.5 | MAE | Always-median MAE | Bias | QWK |", "|---|---|---|---|---|---|---|"]
    for k, s in rep.get("bands", {}).items():
        L.append(f"| {k} | {f(s['exact'])} | {f(s['within_0.5'])} | {f(s['mae'])} | "
                 f"{f(s['baseline_mae_always_median'])} | {f(s['bias'])} | {f(s['qwk'])} |")
    L += ["", "MAE = average distance in bands. QWK = agreement with the reference (1 = perfect, 0 = chance). "
          "'Always-median' is what a model gets by ignoring the essay: a useful model must beat it."]
    return "\n".join(L)
