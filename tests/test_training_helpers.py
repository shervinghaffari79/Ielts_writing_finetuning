"""Training-pipeline pieces that need no GPU or download (a fake tokenizer stands in)."""
from ielts_ft import train_lora as T


class FakeTok:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [ord(c) % 250 + 1 for c in text]})()


def test_loss_only_on_answer():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "ANSWER"}]
    ids, labels = T.tokenize_example(FakeTok(), msgs, 1000)
    n_prompt = sum(1 for l in labels if l == -100)
    assert len(ids) == len(labels) and n_prompt == len("<system>s<user>u<assistant>")
    assert labels[n_prompt:] == ids[n_prompt:] and len(ids) - n_prompt == len("ANSWER<eos>")


def test_too_long_is_skipped_not_truncated():
    msgs = [{"role": "user", "content": "u" * 100}, {"role": "assistant", "content": "a" * 100}]
    assert T.tokenize_example(FakeTok(), msgs, 50) is None


def test_collator_pads_and_masks():
    import torch
    b = T.Collator(0)([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, {"input_ids": [4], "labels": [4]}])
    assert b["input_ids"].shape == (2, 3) and b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["labels"][1].tolist() == [4, -100, -100]


def test_config_overrides_and_unknown_keys():
    cfg = T.load_config(None, ["lora_r=32", "use_dora=true", "target_modules=[a,b]"])
    assert cfg.lora_r == 32 and cfg.use_dora is True
    import pytest
    with pytest.raises(SystemExit):
        T.load_config(None, ["not_a_key=1"])


def test_target_modules_cover_hybrid_layers():
    import torch.nn as nn

    class Blk(nn.Module):
        def __init__(self, linear):
            super().__init__()
            if linear:
                self.linear_attn = nn.ModuleDict({"in_proj_qkv": nn.Linear(4, 4), "out_proj": nn.Linear(4, 4)})
            else:
                self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(4, 4), "o_proj": nn.Linear(4, 4)})
            self.mlp = nn.ModuleDict({"gate_proj": nn.Linear(4, 4)})

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleDict({"layers": nn.ModuleList([Blk(True), Blk(True), Blk(True), Blk(False)])})
            self.lm_head = nn.Linear(4, 4)

    names = T.find_target_modules(M(), T.TrainConfig())
    assert any("linear_attn.in_proj_qkv" in n for n in names) and any("self_attn.q_proj" in n for n in names)
    assert not any("lm_head" in n for n in names)
    attn_only_q = T.find_target_modules(M(), T.TrainConfig(target_preset="custom", target_modules=["q_proj"]))
    assert len(attn_only_q) == 1      # shows why q/k/v/o-only would miss 3 of 4 layers


def test_olora_disables_rslora_and_dora_skips_zero_rows():
    cfg = T.TrainConfig(init_lora_weights="olora", use_rslora=True)
    cfg.validate()
    assert cfg.use_rslora is False
    import torch, torch.nn as nn

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(4, 4)
            self.q_proj = nn.Linear(4, 4)
            with torch.no_grad():
                self.in_proj_qkv.weight[1].zero_()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleDict({"layers": nn.ModuleList([Blk()])})

    names = T.find_target_modules(M(), T.TrainConfig(use_dora=True))
    assert names == ["model.layers.0.q_proj"]
    assert len(T.find_target_modules(M(), T.TrainConfig(use_dora=False))) == 2
