import json

import pytest

from ielts_ft import data_prep as D
from ielts_ft import parsing as P
from ielts_ft.schema import quote_in_essay, validate_assessment
from conftest import rater_text


def test_correction_is_cut_and_quotes_verified(essay):
    t = rater_text(6, "mixed classes are better because students learn from each other",
                   "mixed classes are better since pupils learn from one another")
    pc = P.parse_criterion_text(t, essay)
    assert pc.band == 6
    assert pc.weaknesses and all("pupils" not in w for w in pc.weaknesses)     # invented correction removed
    assert all(quote_in_essay(q, essay) for q in pc.evidence) and pc.evidence


def test_unverifiable_only_sentence_dropped(essay):
    t = 'Band 6:\nPositives:\nThe writer says "this never appears anywhere in text".\nNegatives:\nIdeas are thin overall.'
    pc = P.parse_criterion_text(t, essay)
    assert pc.strengths == [] and pc.weaknesses == ["Ideas are thin overall."]


def test_snap_fixes_small_typos(essay):
    got = P.resolve_quote("students learn from eachother", essay)
    assert got and quote_in_essay(got, essay)


def test_feedback_column_ignores_rewrites():
    fb = ('Body Paragraph 1:\nImprovement point: Develop the argument with a specific example.\n'
          'Original: "First of all, weaker students can ask for help."\nSuggested: REWRITE\n'
          'Overall Improvements:\nTask Response: Develop ideas.')
    r = P.parse_feedback_column(fb)
    assert r["overall_line"]["task_response"] == "Develop ideas."
    assert "REWRITE" not in json.dumps(r)


def test_prepare_end_to_end(tmp_path, df):
    xlsx = tmp_path / "d.xlsx"
    df.to_excel(xlsx, index=False)
    rep = D.prepare(str(xlsx), str(tmp_path / "out"), seed=1)
    assert rep["records_kept"] == len(df)
    groups = {}
    for split in ("train", "validation", "test"):
        for line in open(tmp_path / "out" / f"{split}.jsonl"):
            r = json.loads(line)
            groups.setdefault(r["group"], set()).add(split)
            ref = r["meta"]["reference"]
            assert validate_assessment(ref, r["meta"]["essay"]) == []
            assert r["messages"][-1]["role"] == "assistant"
            # scores must NOT leak into the model input
            assert "Band" not in r["messages"][1]["content"]
    assert all(len(v) == 1 for v in groups.values()), "a prompt group is in more than one split"
    assert rep["splits"]["test"]["essays"] > 0 and rep["splits"]["validation"]["essays"] > 0


def test_similar_prompts_share_a_group():
    ps = ["Many museums charge for admission while others are free. Do the advantages of charging outweigh the disadvantages?",
          "Many museums charge for admission while others are free. Do you think the advantages of charging people for admission to museums outweigh the disadvantages?",
          "Some people think zoos should be banned. To what extent do you agree?"]
    g = D.cluster_prompts(ps)
    assert g[0] == g[1] and g[0] != g[2]


def test_score_mismatch_between_raters_drops_record(df):
    row = df.iloc[0].copy()
    row["Rater1_LR"] = 5
    t, info = D.build_target(row)
    assert t is None and "LR" in info["reason"]


def test_underlength_handling(df):
    row = df.iloc[0]          # first essay of each prompt is truncated (<250 words)
    t, _ = D.build_target(row)
    assert t["underlength_warning"] is True
    assert any("250" in w for w in t["criteria"]["task_response"]["weaknesses"])
    assert t["priority_areas"][0]["criterion"] == "task_response"
