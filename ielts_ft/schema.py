"""Output schema, validation and text utilities shared by every other module.

The structure follows the Stage 1 instruction document exactly. Field NAMES and the
ORDER of keys are fixed so that the output can be consumed by the later
prompt-engineering stage.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

CRITERIA = (
    "task_response",
    "coherence_and_cohesion",
    "lexical_resource",
    "grammatical_range_and_accuracy",
)
CRITERIA_SHORT = {"TR": CRITERIA[0], "CC": CRITERIA[1], "LR": CRITERIA[2], "GRA": CRITERIA[3]}
SHORT_OF = {v: k for k, v in CRITERIA_SHORT.items()}
CRITERION_TITLES = {
    "task_response": "Task Response",
    "coherence_and_cohesion": "Coherence and Cohesion",
    "lexical_resource": "Lexical Resource",
    "grammatical_range_and_accuracy": "Grammatical Range and Accuracy",
}
CRITERION_LISTS = ("strengths", "weaknesses", "evidence", "improvement_suggestions")
MAX_PRIORITIES = 8
MIN_WORDS = 250

# Order in which keys are written. "spec" is the order from the instruction document.
# "reasoning_first" puts the numbers AFTER the analysis (an optional experiment: the model
# then "thinks" before it commits to a band). Either way the parsed object is identical.
ORDER_SPEC = "spec"
ORDER_REASONING_FIRST = "reasoning_first"


# --------------------------------------------------------------------------- text utils
_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', " ": " ",
                            "–": "-", "—": "-"})


def normalize_text(s: str) -> str:
    """Lower-case, unify quotes/dashes, collapse whitespace. Used only for comparisons."""
    return re.sub(r"\s+", " ", s.translate(_QUOTE_MAP)).strip().lower()


def count_words(essay: str) -> int:
    return len(essay.split())


def quote_in_essay(quote: str, essay: str) -> bool:
    """True if `quote` appears in `essay` (case/whitespace/quote-style insensitive).

    A quote may contain an ellipsis ("..." or "…"); every fragment must then occur, in order.
    """
    q = normalize_text(quote).strip(" \"'.,;:!?-")
    if len(q) < 3:
        return False
    e = normalize_text(essay)
    pos = 0
    for frag in re.split(r"\.{3,}|…", q):
        frag = frag.strip(" \"'.,;:!?-")
        if not frag:
            continue
        i = e.find(frag, pos)
        if i < 0:
            return False
        pos = i + len(frag)
    return True


# --------------------------------------------------------------------------- bands
def is_valid_band(x: Any) -> bool:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return 0 <= x <= 9 and abs(x * 2 - round(x * 2)) < 1e-9


def round_ielts(mean: float) -> float:
    """IELTS rounding of an average: .25 -> .5, .75 -> next whole band."""
    f = math.floor(mean)
    r = mean - f
    return f + (0.0 if r < 0.25 else 0.5 if r < 0.75 else 1.0)


def overall_from_criteria(bands: list[float]) -> float:
    return round_ielts(sum(bands) / len(bands))


def fmt_band(b: float) -> str:
    return str(int(b)) if float(b).is_integer() else str(b)


# --------------------------------------------------------------------------- templates
def empty_assessment() -> dict:
    """The template from the instruction document (values to be filled in)."""
    return {
        "overall_band": None,
        "word_count": None,
        "underlength_warning": False,
        "criteria": {c: {"band": None, "strengths": [], "weaknesses": [], "evidence": [],
                         "improvement_suggestions": []} for c in CRITERIA},
        "priority_areas": [{"priority": 1, "criterion": "", "issue": "", "evidence": ""}],
        "overall_feedback": "",
    }


def ordered(obj: dict, order: str = ORDER_SPEC) -> dict:
    """Return a copy of `obj` with keys in the fixed order."""
    crit = {}
    for c in CRITERIA:
        d = obj["criteria"][c]
        if order == ORDER_REASONING_FIRST:
            crit[c] = {k: d[k] for k in CRITERION_LISTS} | {"band": d["band"]}
        else:
            crit[c] = {"band": d["band"]} | {k: d[k] for k in CRITERION_LISTS}
    pri = [{"priority": p["priority"], "criterion": p["criterion"], "issue": p["issue"],
            "evidence": p["evidence"]} for p in obj["priority_areas"]]
    if order == ORDER_REASONING_FIRST:
        return {"word_count": obj["word_count"], "underlength_warning": obj["underlength_warning"],
                "criteria": crit, "priority_areas": pri, "overall_feedback": obj["overall_feedback"],
                "overall_band": obj["overall_band"]}
    return {"overall_band": obj["overall_band"], "word_count": obj["word_count"],
            "underlength_warning": obj["underlength_warning"], "criteria": crit,
            "priority_areas": pri, "overall_feedback": obj["overall_feedback"]}


def canonical_json(obj: dict, order: str = ORDER_SPEC, indent: int | None = None) -> str:
    """Deterministic JSON string (fixed key order, no spaces unless indent is set)."""
    seps = (",", ":") if indent is None else None
    return json.dumps(ordered(obj, order), ensure_ascii=False, indent=indent, separators=seps)


# --------------------------------------------------------------------------- validation
def validate_assessment(obj: Any, essay: str | None = None, strict_overall: bool = False) -> list[str]:
    """Return a list of problems ([] = valid).

    With `essay` given, every evidence quote must occur verbatim in the essay and the
    word-count fields are checked against the real count.
    """
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["root is not an object"]
    for k in ("overall_band", "word_count", "underlength_warning", "criteria", "priority_areas",
              "overall_feedback"):
        if k not in obj:
            errs.append(f"missing key: {k}")
    if errs:
        return errs
    if not is_valid_band(obj["overall_band"]):
        errs.append(f"overall_band invalid: {obj['overall_band']!r}")
    if not isinstance(obj["word_count"], int) or isinstance(obj["word_count"], bool):
        errs.append("word_count must be an integer")
    if not isinstance(obj["underlength_warning"], bool):
        errs.append("underlength_warning must be boolean")
    crit = obj["criteria"]
    if not isinstance(crit, dict) or set(crit) != set(CRITERIA):
        errs.append("criteria must contain exactly the four criterion keys")
        return errs
    bands = []
    for c in CRITERIA:
        d = crit[c]
        if not isinstance(d, dict):
            errs.append(f"{c}: not an object")
            continue
        if not is_valid_band(d.get("band")):
            errs.append(f"{c}.band invalid: {d.get('band')!r}")
        else:
            bands.append(d["band"])
        for k in CRITERION_LISTS:
            v = d.get(k)
            if not isinstance(v, list) or not all(isinstance(s, str) and s.strip() for s in v):
                errs.append(f"{c}.{k} must be a list of non-empty strings")
            elif not v:
                errs.append(f"{c}.{k} is empty")
        if essay is not None and isinstance(d.get("evidence"), list):
            for q in d["evidence"]:
                if isinstance(q, str) and not quote_in_essay(q, essay):
                    errs.append(f"{c}.evidence quote not found in essay: {q[:60]!r}")
    pa = obj["priority_areas"]
    if not isinstance(pa, list) or not pa or len(pa) > MAX_PRIORITIES:
        errs.append(f"priority_areas must have 1..{MAX_PRIORITIES} items")
    else:
        for i, p in enumerate(pa, 1):
            if not isinstance(p, dict) or set(p) != {"priority", "criterion", "issue", "evidence"}:
                errs.append(f"priority_areas[{i}] has wrong keys")
                continue
            if p["priority"] != i:
                errs.append(f"priority_areas[{i}].priority should be {i}")
            if p["criterion"] not in CRITERIA:
                errs.append(f"priority_areas[{i}].criterion invalid: {p['criterion']!r}")
            if not isinstance(p["issue"], str) or not p["issue"].strip():
                errs.append(f"priority_areas[{i}].issue empty")
            if essay is not None and isinstance(p["evidence"], str) and p["evidence"].strip() \
                    and not quote_in_essay(p["evidence"], essay):
                errs.append(f"priority_areas[{i}].evidence not found in essay")
    if not isinstance(obj["overall_feedback"], str) or not obj["overall_feedback"].strip():
        errs.append("overall_feedback empty")
    if essay is not None:
        wc = count_words(essay)
        if obj["word_count"] != wc:
            errs.append(f"word_count {obj['word_count']} != real {wc}")
        if obj["underlength_warning"] != (wc < MIN_WORDS):
            errs.append("underlength_warning inconsistent with word count")
    if len(bands) == 4 and is_valid_band(obj["overall_band"]):
        gap = abs(obj["overall_band"] - overall_from_criteria(bands))
        if gap > (0 if strict_overall else 0.5):
            errs.append(f"overall_band {obj['overall_band']} far from criteria mean ({gap} bands)")
    return errs


# --------------------------------------------------------------------------- parsing model output
def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of raw model text (handles fences, <think>, chatter)."""
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start:i + 1]
                for cand in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                    try:
                        return json.loads(cand)
                    except json.JSONDecodeError:
                        pass
                return None
    return None


def postprocess(obj: dict, essay: str) -> dict:
    """Deterministic fixes applied to model output: real word count / under-length flag.

    A language model counts words badly, a program does not. Nothing else is altered.
    """
    wc = count_words(essay)
    obj = json.loads(json.dumps(obj))
    obj["word_count"] = wc
    obj["underlength_warning"] = wc < MIN_WORDS
    return obj
