"""Run the (fine-tuned or base) model on essays and turn the raw text into a validated assessment."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .prompts import SYSTEM_PROMPT_SHORT, build_messages, system_prompt_full
from .schema import extract_json, postprocess, validate_assessment


@dataclass
class Result:
    assessment: dict | None
    raw: str
    errors: list[str]
    attempts: int = 1


class Assessor:
    """Loads a base model (+ optional LoRA adapter) and assesses essays.

    baseline=True uses the LONG system prompt with the rubric/rules/schema - this is the fair way to
    test an un-tuned model ("how far does prompting alone get us?").
    """

    def __init__(self, base_model: str, adapter: str | None = None, *, baseline: bool = False,
                 precision: str = "auto", quantization: str = "none", enable_thinking: bool = False,
                 attn_implementation: str = "sdpa"):
        from .train_lora import TrainConfig, load_base
        cfg = TrainConfig(model_name=base_model, precision=precision, quantization=quantization,
                          attn_implementation=attn_implementation)
        self.model, self.tok, _ = load_base(cfg)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.system = system_prompt_full() if baseline else SYSTEM_PROMPT_SHORT
        self.enable_thinking = enable_thinking

    def _prompt_text(self, prompt: str, essay: str) -> str:
        msgs = build_messages(prompt, essay, system=self.system)
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=self.enable_thinking)

    def generate(self, items: list[tuple[str, str]], *, max_new_tokens: int = 1800, temperature: float = 0.0,
                 batch_size: int = 4) -> list[str]:
        import torch
        outs: list[str] = []
        for i in range(0, len(items), batch_size):
            texts = [self._prompt_text(p, e) for p, e in items[i:i + batch_size]]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                kw.update(do_sample=False)
            with torch.no_grad():
                gen = self.model.generate(**enc, **kw)
            for row in gen[:, enc.input_ids.shape[1]:]:
                outs.append(self.tok.decode(row, skip_special_tokens=True))
        return outs

    def assess(self, prompt: str, essay: str, *, retries: int = 1, max_new_tokens: int = 1800) -> Result:
        """Greedy first; if the JSON is broken or a quote is not in the essay, resample with a little noise."""
        best: Result | None = None
        for attempt in range(retries + 1):
            raw = self.generate([(prompt, essay)], max_new_tokens=max_new_tokens,
                                temperature=0.0 if attempt == 0 else 0.4, batch_size=1)[0]
            obj = extract_json(raw)
            if obj is None:
                res = Result(None, raw, ["no valid JSON"], attempt + 1)
            else:
                try:
                    obj = postprocess(obj, essay)
                    res = Result(obj, raw, validate_assessment(obj, essay), attempt + 1)
                except (KeyError, TypeError):
                    res = Result(None, raw, ["JSON does not follow the schema"], attempt + 1)
            if best is None or len(res.errors) < len(best.errors):
                best = res
            if not res.errors:
                break
        return best


def read_text_arg(value: str) -> str:
    """Treat `value` as a file path if it exists, otherwise as literal text."""
    import os
    return open(value, encoding="utf-8").read() if os.path.exists(value) else value


def main():
    import argparse
    from .render import render_report
    ap = argparse.ArgumentParser(description="Assess one essay.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt", required=True, help="task prompt text or a .txt file")
    ap.add_argument("--essay", required=True, help="essay text or a .txt file")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of a readable report")
    a = ap.parse_args()
    ass = Assessor(a.base, a.adapter, baseline=a.baseline)
    res = ass.assess(read_text_arg(a.prompt), read_text_arg(a.essay))
    if res.assessment is None:
        print("Model output was not valid JSON:\n", res.raw)
        return
    print(json.dumps(res.assessment, indent=2, ensure_ascii=False) if a.json else render_report(res.assessment))
    if res.errors:
        print("\n[validator warnings]", *res.errors, sep="\n - ")


if __name__ == "__main__":
    main()
