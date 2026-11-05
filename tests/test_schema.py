import json

from ielts_ft import schema as S


def good(essay):
    wc = S.count_words(essay)
    crit = {c: {"band": 6.0, "strengths": ["s"], "weaknesses": ["w"], "evidence": ["students learn from each other"],
                "improvement_suggestions": ["do x"]} for c in S.CRITERIA}
    return {"overall_band": 6.0, "word_count": wc, "underlength_warning": wc < 250, "criteria": crit,
            "priority_areas": [{"priority": 1, "criterion": "task_response", "issue": "i", "evidence": ""}],
            "overall_feedback": "ok"}


def test_bands():
    assert S.is_valid_band(6.5) and S.is_valid_band(0) and not S.is_valid_band(6.3) and not S.is_valid_band(9.5)
    assert not S.is_valid_band(True) and not S.is_valid_band("6")
    assert S.round_ielts(6.25) == 6.5 and S.round_ielts(6.75) == 7.0 and S.round_ielts(6.24) == 6.0
    assert S.overall_from_criteria([6, 6, 6, 7]) == 6.5


def test_quote_matching():
    e = "It’s true that   the CITY will be\nmore crowded, and people feel isolated."
    assert S.quote_in_essay("it's true that the city will be more crowded", e)
    assert S.quote_in_essay("the city ... people feel isolated", e)
    assert not S.quote_in_essay("people feel lonely", e)
    assert not S.quote_in_essay("a", e)


def test_validate_and_fabrication(essay):
    a = good(essay)
    assert S.validate_assessment(a, essay) == []
    a["criteria"]["lexical_resource"]["evidence"] = ["this sentence is not in the essay"]
    assert any("not found" in e for e in S.validate_assessment(a, essay))
    b = good(essay)
    b["criteria"]["task_response"]["band"] = 6.3
    assert any("band invalid" in e for e in S.validate_assessment(b, essay))
    c = good(essay)
    c["word_count"] = 5
    assert any("word_count" in e for e in S.validate_assessment(c, essay))


def test_key_order_and_roundtrip(essay):
    a = good(essay)
    spec = json.loads(S.canonical_json(a))
    assert list(spec)[0] == "overall_band" and list(spec["criteria"]["task_response"])[0] == "band"
    rf = json.loads(S.canonical_json(a, S.ORDER_REASONING_FIRST))
    assert list(rf)[-1] == "overall_band" and list(rf["criteria"]["task_response"])[-1] == "band"
    assert rf == json.loads(json.dumps(spec)) or S.ordered(rf) == S.ordered(spec)


def test_extract_json_variants():
    assert S.extract_json('```json\n{"a": 1,}\n```') == {"a": 1}
    assert S.extract_json('<think>hmm {x}</think>\nHere: {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert S.extract_json("no json") is None


def test_postprocess_overrides_word_count(essay):
    a = good(essay)
    a["word_count"], a["underlength_warning"] = 999, True
    out = S.postprocess(a, "word " * 100)
    assert out["word_count"] == 100 and out["underlength_warning"] is True
    out = S.postprocess(a, "word " * 300)
    assert out["word_count"] == 300 and out["underlength_warning"] is False
