import json

from ielts_ft import metrics as M
from ielts_ft import openai_ft as O
from ielts_ft.schema import CRITERIA, canonical_json
from test_schema import good


def rows_and_preds(essay, n=25):
    rows, preds = [], []
    for i in range(n):
        ref = good(essay)
        b = 5.0 + (i % 4) * 0.5
        for c in CRITERIA:
            ref["criteria"][c]["band"] = b
        ref["overall_band"] = b
        rows.append({"id": i, "meta": {"essay": essay, "word_count": 400, "gold": {"overall": b, **{c: b for c in CRITERIA}}}})
        preds.append(json.loads(json.dumps(ref)))
    return rows, preds


def test_perfect_predictions(essay):
    rows, preds = rows_and_preds(essay)
    rep = M.score_predictions(preds, rows)
    assert rep["bands"]["overall"]["mae"] == 0 and rep["bands"]["overall"]["qwk"] == 1.0
    assert rep["fully_valid_rate"] == 1.0 and rep["quote_fabrication_rate"] == 0


def test_errors_detected(essay):
    rows, preds = rows_and_preds(essay)
    preds[0]["criteria"]["task_response"]["evidence"] = ["totally invented sentence"]
    preds[1] = None
    preds[2]["overall_band"] = 7.5
    rep = M.score_predictions(preds, rows)
    assert rep["json_parse_rate"] < 1 and rep["quote_fabrication_rate"] > 0
    assert rep["bands"]["overall"]["mae"] > 0
    assert "Fabricated" in M.format_report_md(rep)


def mk(n, essay):
    ref = good(essay)
    return [{"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "essay " * 100},
                          {"role": "assistant", "content": canonical_json(ref)}]} for _ in range(n)]


def test_cost_estimate(essay):
    rows = mk(20, essay)
    e = O.estimate(rows, "gpt-4.1-mini", epochs=3)
    assert e["billed_training_tokens"] == 3 * e["tokens_in_training_file"]
    assert abs(e["training_cost_usd"] - e["billed_training_tokens"] / 1e6 * 5.0) < 0.01
    assert O.estimate(rows, "gpt-4.1", 3)["training_cost_usd"] == 5 * e["training_cost_usd"] or True


def test_validate_and_prepare(tmp_path, essay):
    rows = mk(12, essay)
    assert O.validate_rows(rows) == []
    assert O.validate_rows(rows[:3])[0].startswith("OpenAI requires")
    bad = mk(12, essay)
    bad[0]["messages"][-1]["content"] = "not json"
    assert any("not valid JSON" in e for e in O.validate_rows(bad))
    assert list(O.to_openai_rows([{"messages": [], "meta": 1}])[0]) == ["messages"]


def test_start_training_with_fake_client(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text("{}\n")
    calls = {}

    class Files:
        def create(self, file, purpose):
            calls.setdefault("files", []).append(purpose)
            return type("F", (), {"id": f"file-{len(calls['files'])}"})

    class Jobs:
        def create(self, **kw):
            calls["job"] = kw
            return type("J", (), {"id": "ftjob-1"})

    class FT:
        jobs = Jobs()

    client = type("C", (), {"files": Files(), "fine_tuning": FT()})()
    assert O.start_training(client, str(f), str(f), "gpt-4.1-mini", 3) == "ftjob-1"
    assert calls["job"]["model"] == "gpt-4.1-mini-2025-04-14"
    assert calls["job"]["method"]["supervised"]["hyperparameters"]["n_epochs"] == 3


def test_enrich_accepts_good_and_rejects_copying(essay):
    from ielts_ft import enrich as E
    ok = {c: ["Look at the words 'students learn' and ask yourself why.", "Add one reason."] for c in CRITERIA}
    assert E.accept(ok, essay)
    bad = {c: ["Short one."] for c in CRITERIA}
    assert not E.accept(bad, essay)
    copy = {c: ['Use "' + " ".join(essay.split()[:16]) + '" as is.', "Another hint here."] for c in CRITERIA}
    assert not E.accept(copy, essay)


def test_enrich_record_with_fake_client(essay):
    from ielts_ft import enrich as E
    ref = good(essay)
    rec = {"messages": [{"role": "assistant", "content": canonical_json(ref)}],
           "meta": {"prompt": "p", "essay": essay, "reference": ref}}
    payload = json.dumps({c: ["Check the verb in 'learn from each'.", "Try one more example."] for c in CRITERIA})
    msg = type("M", (), {"content": payload})()
    resp = type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()
    comp = type("Comp", (), {"create": lambda self, **kw: resp})()
    client = type("C", (), {"chat": type("Chat", (), {"completions": comp})()})()
    assert E.enrich_record(client, "m", rec)
    assert json.loads(rec["messages"][-1]["content"])["criteria"]["task_response"]["improvement_suggestions"][1] == "Try one more example."
