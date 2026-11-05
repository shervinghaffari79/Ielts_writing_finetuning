"""Spreadsheet -> train / validation / test JSONL (chat format) + data report.

    python -m ielts_ft.data_prep --xlsx data/IELTS_Task2_565_cleaned_minimal.xlsx --out data/processed

Key points (from the Stage 1 instructions):
  * Split 80/10/10 BY PROMPT GROUP, never by essay. Identical or very similar prompts are merged
    into one group first, so no prompt (or near-copy of it) appears in two splits.
  * The gold numeric scores (Rater1_* / essay_score) are the labels; the text explanation comes from
    Rater2_* and is kept only when it agrees with those scores and every quotation is verifiable.
  * The model input is only the prompt + essay. Scores never appear in the input.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .parsing import (build_overall_feedback, build_priorities, build_suggestions, parse_criterion_text,
                      parse_feedback_column, resolve_quote)
from .prompts import build_messages
from .schema import (CRITERIA, CRITERIA_SHORT, MIN_WORDS, ORDER_SPEC, canonical_json, count_words,
                     overall_from_criteria, validate_assessment)

REQUIRED_COLUMNS = ["essay_id", "essay_prompt", "essay", "essay_score", "Rater1_TR", "Rater1_CC",
                    "Rater1_LR", "Rater1_GRA", "Rater2_TR", "Rater2_CC", "Rater2_LR", "Rater2_GRA", "Feedback"]


# ----------------------------------------------------------------------------- build one target
def build_target(row: pd.Series, overall_policy: str = "gold", require_full: bool = True):
    """Return (assessment_dict | None, info). `info` explains drops and carries quote statistics."""
    essay = str(row["essay"]).strip()
    wc = count_words(essay)
    info = {"quotes": 0, "verbatim": 0, "snapped": 0, "dropped": 0, "reason": None}
    bands, parsed = {}, {}
    fb = parse_feedback_column(str(row["Feedback"]))
    for short, key in CRITERIA_SHORT.items():
        gold = float(row[f"Rater1_{short}"])
        pc = parse_criterion_text(str(row[f"Rater2_{short}"]), essay)
        for k in ("quotes", "verbatim", "snapped", "dropped"):
            info[k] += pc.stats[k]
        if pc.band is None:
            info["reason"] = f"{short}: no band in explanation"
            return None, info
        if abs(pc.band - gold) > 1e-9:
            info["reason"] = f"{short}: explanation band {pc.band} != gold {gold}"
            return None, info
        bands[key], parsed[key] = gold, pc
    for key, pc in parsed.items():
        if not pc.evidence:   # fallback: essay sentences the reviewer flagged for this criterion (verified)
            for o in fb["originals"][key]:
                real = resolve_quote(o, essay)
                if real and real not in pc.evidence:
                    pc.evidence.append(real)
                if len(pc.evidence) >= 2:
                    break
            info["evidence_fallback"] = info.get("evidence_fallback", 0) + bool(pc.evidence)
        missing = [n for n in ("strengths", "weaknesses", "evidence") if not getattr(pc, n)]
        if missing and require_full:
            info["reason"] = f"{key}: nothing usable for {','.join(missing)}"
            return None, info
    underlength = wc < MIN_WORDS
    strengths = {c: list(parsed[c].strengths) for c in CRITERIA}
    weaknesses = {c: list(parsed[c].weaknesses) for c in CRITERIA}
    evidence = {c: list(parsed[c].evidence) for c in CRITERIA}
    if underlength:
        weaknesses["task_response"].insert(
            0, f"The essay has only {wc} words, below the 250-word minimum, which limits how fully the task "
               "can be answered and developed.")
        weaknesses["task_response"] = weaknesses["task_response"][:4]
    criteria = {}
    for c in CRITERIA:
        criteria[c] = {
            "band": bands[c],
            "strengths": strengths[c],
            "weaknesses": weaknesses[c],
            "evidence": evidence[c],
            "improvement_suggestions": build_suggestions(
                c, weaknesses[c], parsed[c].weak_quotes, fb["overall_line"].get(c), fb["points"][c], underlength),
        }
    rule_overall = overall_from_criteria(list(bands.values()))
    overall = float(row["essay_score"]) if overall_policy == "gold" else rule_overall
    pri = build_priorities(bands, weaknesses, evidence, wc)
    out = {
        "overall_band": overall,
        "word_count": wc,
        "underlength_warning": underlength,
        "criteria": criteria,
        "priority_areas": pri,
        "overall_feedback": build_overall_feedback(overall, bands, strengths, weaknesses, pri, wc,
                                                   int(row["essay_id"])),
    }
    errs = validate_assessment(out, essay)
    if errs:
        info["reason"] = "validation: " + "; ".join(errs[:3])
        return None, info
    info["rule_overall"] = rule_overall
    return out, info


# ----------------------------------------------------------------------------- prompt grouping
def _norm_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", p.lower())).strip()


def cluster_prompts(prompts: list[str], threshold: float = 0.6, embed_model: str | None = None) -> list[int]:
    """Group identical / near-identical prompts. Returns a group id for every item in `prompts`.

    Default: TF-IDF (word 1-2 grams) cosine similarity >= threshold, connected components.
    Optional: sentence-transformers embeddings (semantic similarity) with `embed_model`.
    """
    uniq = sorted({_norm_prompt(p) for p in prompts})
    idx = {p: i for i, p in enumerate(uniq)}
    parent = list(range(len(uniq)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if embed_model:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(embed_model).encode(uniq, normalize_embeddings=True)
        sim = emb @ emb.T
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit_transform(uniq)
        sim = (X @ X.T).toarray()
    n = len(uniq)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                parent[find(j)] = find(i)
    roots = {}
    return [roots.setdefault(find(idx[_norm_prompt(p)]), len(roots)) for p in prompts]


def group_split(groups: list[int], fractions=(0.8, 0.1, 0.1), seed: int = 42, scores: list[float] | None = None):
    """Assign whole groups to train/validation/test so the essay counts approach `fractions`."""
    rng = random.Random(seed)
    size = Counter(groups)
    gl = list(size)
    rng.shuffle(gl)
    gl.sort(key=lambda g: -size[g])             # big groups first, ties in random order
    total = sum(size.values())
    target = [f * total for f in fractions]
    filled = [0, 0, 0]
    assign = {}
    for g in gl:
        k = max(range(3), key=lambda i: (target[i] - filled[i]) / target[i])
        assign[g] = k
        filled[k] += size[g]
    return [("train", "validation", "test")[assign[g]] for g in groups]


# ----------------------------------------------------------------------------- main pipeline
def prepare(xlsx: str, out_dir: str, *, seed: int = 42, threshold: float = 0.6, embed_model: str | None = None,
            overall_policy: str = "gold", order: str = ORDER_SPEC, word_count_hint: bool = False,
            require_full: bool = True, tokenizer_name: str | None = None) -> dict:
    df = pd.read_excel(xlsx)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Spreadsheet is missing columns: {missing}")
    n_raw = len(df)
    df = df.dropna(subset=["essay", "essay_prompt"]).copy()
    for c in ("essay", "essay_prompt"):
        df[c] = df[c].astype(str).str.strip()
    df = df.drop_duplicates(subset=["essay_prompt", "essay"]).reset_index(drop=True)
    report: dict = {"rows_raw": n_raw, "rows_after_dedupe": len(df)}

    # 1. targets
    targets, infos, drops = [], [], Counter()
    for _, row in df.iterrows():
        t, info = build_target(row, overall_policy, require_full)
        targets.append(t)
        infos.append(info)
        if t is None:
            drops[info["reason"].split(":")[0] + ":" + info["reason"].split(":")[1][:40]
                  if info["reason"] and ":" in info["reason"] else str(info["reason"])] += 1
    df["ok"] = [t is not None for t in targets]
    q = {k: sum(i[k] for i in infos) for k in ("quotes", "verbatim", "snapped", "dropped")}
    report["quotes_in_source"] = q
    report["records_kept"] = int(df.ok.sum())
    report["records_dropped"] = int((~df.ok).sum())
    report["drop_reasons"] = dict(drops.most_common(10))
    report["gold_overall_differs_from_rule_in"] = int(sum(
        1 for t, i in zip(targets, infos) if t and abs(t["overall_band"] - i["rule_overall"]) > 1e-9))

    # 2. grouped split (on ALL rows so that dropping records cannot move a prompt between splits)
    df["group"] = cluster_prompts(df.essay_prompt.tolist(), threshold, embed_model)
    df["split"] = group_split(df.group.tolist(), seed=seed)
    report["unique_prompts"] = int(df.essay_prompt.map(_norm_prompt).nunique())
    report["prompt_groups"] = int(df.group.nunique())

    # 3. write
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = {"train": [], "validation": [], "test": []}
    for (_, row), t in zip(df.iterrows(), targets):
        if t is None:
            continue
        gold = {"overall": float(row["essay_score"]),
                **{k: float(row[f"Rater1_{s}"]) for s, k in CRITERIA_SHORT.items()}}
        msgs = build_messages(row["essay_prompt"], row["essay"], canonical_json(t, order),
                              word_count_hint=word_count_hint)
        recs[row["split"]].append({
            "id": int(row["essay_id"]), "group": int(row["group"]), "messages": msgs,
            "meta": {"prompt": row["essay_prompt"], "essay": row["essay"], "gold": gold,
                     "word_count": t["word_count"], "reference": t}})
    for name, items in recs.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4. leakage check + stats
    sets = {k: {r["group"] for r in v} for k, v in recs.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                or sets["validation"] & sets["test"]), "prompt group leaked across splits"
    report["splits"] = {}
    for name, items in recs.items():
        ov = Counter(r["meta"]["gold"]["overall"] for r in items)
        report["splits"][name] = {
            "essays": len(items), "prompt_groups": len(sets[name]),
            "mean_overall": round(sum(r["meta"]["gold"]["overall"] for r in items) / max(1, len(items)), 3),
            "overall_distribution": {str(k): v for k, v in sorted(ov.items())},
            "underlength": sum(r["meta"]["reference"]["underlength_warning"] for r in items)}
    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        lens = sorted(len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=False)
                          if False else tok(tok.apply_chat_template(r["messages"], tokenize=False)).input_ids)
                      for r in recs["train"])
        if lens:
            report["train_token_lengths"] = {"tokenizer": tokenizer_name, "mean": sum(lens) // len(lens),
                                             "p95": lens[int(0.95 * len(lens))], "max": lens[-1],
                                             "total": sum(lens)}
    (out / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--similarity", type=float, default=0.6, help="prompt similarity threshold for grouping")
    ap.add_argument("--embed-model", default=None, help="e.g. sentence-transformers/all-MiniLM-L6-v2 (semantic grouping)")
    ap.add_argument("--overall-policy", choices=["gold", "rule"], default="gold",
                    help="gold = human overall score; rule = recompute from the 4 criteria (IELTS rounding)")
    ap.add_argument("--word-count-hint", action="store_true", help="add a computed word count line to the input")
    ap.add_argument("--allow-incomplete", action="store_true", help="keep records even if a criterion lacks evidence")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer name to report token lengths")
    a = ap.parse_args()
    rep = prepare(a.xlsx, a.out, seed=a.seed, threshold=a.similarity, embed_model=a.embed_model,
                  overall_policy=a.overall_policy, word_count_hint=a.word_count_hint,
                  require_full=not a.allow_incomplete, tokenizer_name=a.tokenizer)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
