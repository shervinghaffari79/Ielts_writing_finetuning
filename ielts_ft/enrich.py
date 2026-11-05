"""OPTIONAL: upgrade the rule-based `improvement_suggestions` with a stronger "teacher" LLM.

The offline pipeline builds suggestions from templates + the reviewer's own advice lines. They are specific and
guiding, but they repeat across essays (the most common one appears in ~6% of criteria). If you have API access
to a strong model, run this once over the TRAIN and VALIDATION files (never the test references you evaluate
against, to keep the comparison clean) and fine-tune on the result.

    export OPENAI_API_KEY=...
    python -m ielts_ft.enrich --in data/processed/train.jsonl --out data/processed/train_enriched.jsonl --model <teacher-model>

Guardrails: the teacher sees the essay, the bands and the verified weaknesses/evidence; it must return hints (not
rewrites). Responses that copy long stretches of the essay, are too long, or are malformed are rejected and the
original suggestions are kept. The prompt is TEACHER_PROMPT in docs/PROMPT_ENGINEERING.md (section E).
"""
from __future__ import annotations

import argparse
import json
import re

from .schema import CRITERIA, CRITERION_TITLES, canonical_json, normalize_text

TEACHER_SYSTEM = (
    "You are an experienced IELTS Writing teacher writing improvement hints for ONE student essay. For each criterion "
    "write 2-3 hints. A hint is a concrete action the student can do themselves: where to look (quote a few words "
    "from the essay), what to check or ask themselves, and what kind of change to try. Never rewrite the student's "
    "sentences, never give the corrected version, never invent errors that are not in the weaknesses provided. Each "
    "hint is one or two sentences, at most 40 words, in simple English. Return JSON only: "
    '{"task_response":[...],"coherence_and_cohesion":[...],"lexical_resource":[...],"grammatical_range_and_accuracy":[...]}')


def teacher_input(rec: dict) -> str:
    ref = rec["meta"]["reference"]
    lines = [f"Task prompt: {rec['meta']['prompt']}", f"Essay:\n{rec['meta']['essay']}", ""]
    for c in CRITERIA:
        d = ref["criteria"][c]
        lines += [f"## {CRITERION_TITLES[c]} (band {d['band']})", "weaknesses: " + " | ".join(d["weaknesses"]),
                  "evidence quotes: " + " | ".join(f'"{q}"' for q in d["evidence"])]
    return "\n".join(lines)


def accept(sugg: dict, essay: str) -> bool:
    e = normalize_text(essay)
    for c in CRITERIA:
        v = sugg.get(c)
        if not isinstance(v, list) or not 2 <= len(v) <= 3 or not all(isinstance(s, str) and s.strip() for s in v):
            return False
        for s in v:
            if len(s.split()) > 45:
                return False
            # reject if it reproduces a long stretch of the essay (= rewriting / copying)
            for q in re.findall(r'"([^"]+)"', s):
                if len(q.split()) > 14 and normalize_text(q) in e:
                    return False
    return True


def enrich_record(client, model: str, rec: dict) -> bool:
    resp = client.chat.completions.create(
        model=model, temperature=0.3, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": teacher_input(rec)}])
    try:
        sugg = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return False
    if not accept(sugg, rec["meta"]["essay"]):
        return False
    ref = rec["meta"]["reference"]
    for c in CRITERIA:
        ref["criteria"][c]["improvement_suggestions"] = [s.strip() for s in sugg[c]]
    rec["messages"][-1]["content"] = canonical_json(ref)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="teacher model id")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    ok = sum(enrich_record(client, a.model, r) for r in rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"enriched {ok}/{len(rows)} records -> {a.out} (others keep the rule-based suggestions)")


if __name__ == "__main__":
    main()
