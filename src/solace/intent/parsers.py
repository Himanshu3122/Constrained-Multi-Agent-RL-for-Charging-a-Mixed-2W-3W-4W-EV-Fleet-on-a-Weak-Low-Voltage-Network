"""Request parsers: request text + session context -> raw JSON string.

* ``HFParser`` is the paper's design: an open LLM (Llama-3.1-8B-Instruct with a
  LoRA adapter). It needs the optional ``[llm]`` extra and access to the model
  weights. The paper constrains decoding with a JSON grammar; this
  implementation instead extracts the first JSON object and relies on the
  validator/retry loop (approximation, assumption A15).
* ``RuleBasedParser`` is a deterministic, dependency-free stand-in for tests,
  demos and the API when no LLM is available. It is NOT the paper's method and
  its accuracy says nothing about the LLM layer.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from solace.intent.schema import ChargingConstraint, SessionContext, default_constraint

_DIGITS = str.maketrans("०१२३४५६७८९০১২৩৪৫৬৭৮৯", "01234567890123456789")


class IntentParser(Protocol):
    def parse(self, text: str, ctx: SessionContext, feedback: list[str] | None = None) -> str: ...


class RuleBasedParser:
    """Keyword/regex parser for English, Hindi and Bengali requests (demo only)."""

    _pct = re.compile(r"(\d{1,3})\s*(%|percent|प्रतिशत|শতাংশ)", re.IGNORECASE)
    _en_time = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", re.IGNORECASE)
    _hhmm = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
    _hi_time = re.compile(r"(सुबह|शाम|रात|दोपहर)?\s*(\d{1,2})(?::(\d{2}))?\s*बजे")
    _bn_time = re.compile(r"(সকাল|সন্ধ্যা|রাত|দুপুর)?\s*(\d{1,2})(?::(\d{2}))?\s*টা")
    _kwh = re.compile(r"(\d+(?:\.\d+)?)\s*(kwh|यूनिट|ইউনিট)", re.IGNORECASE)
    _v2g_no = re.compile(
        r"(no|don't|do not|never|without)\s+(v2g|discharg\w*|sell\w*)|"
        r"(डिस्चार्ज|बेच)\S*\s*(मत|नहीं|न)\b|मत\s*(बेच|डिस्चार्ज)|"
        r"(ডিসচার্জ|বিক্রি)\S*\s*(করবেন না|করো না|না)", re.IGNORECASE)
    _v2g_yes = re.compile(r"v2g|discharg|sell|डिस्चार्ज|बेच|ডিসচার্জ|বিক্রি", re.IGNORECASE)
    _cheap = re.compile(r"cheap|lowest cost|low cost|save money|सस्त|কম খরচ|সস্তা", re.IGNORECASE)
    _urgent = re.compile(r"urgent|asap|as soon as possible|quickly|जल्दी|तुरंत|জরুরি|তাড়াতাড়ি", re.IGNORECASE)
    _below_before = re.compile(r"(below|under|less than)\s*$", re.IGNORECASE)
    _below_after = re.compile(r"^\s*(से नीचे|-?এর নিচে|নিচে)")

    @staticmethod
    def _clock(hour: int, minute: int, period: str | None) -> str:
        period = (period or "").lower().replace(".", "")
        if period in ("pm", "शाम", "দুপুর", "दोपहर", "সন্ধ্যা") and hour < 12 or period in ("रात", "রাত") and 6 <= hour < 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        return f"{hour % 24:02d}:{minute:02d}"

    def _deadline(self, text: str) -> str | None:
        for rx, period_group, h_group, m_group in ((self._en_time, 3, 1, 2), (self._hi_time, 1, 2, 3),
                                                   (self._bn_time, 1, 2, 3)):
            m = rx.search(text)
            if m:
                return self._clock(int(m.group(h_group)), int(m.group(m_group) or 0), m.group(period_group))
        m = self._hhmm.search(text)
        return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None

    def parse(self, text: str, ctx: SessionContext, feedback: list[str] | None = None) -> str:
        t = text.translate(_DIGITS)
        base = default_constraint(ctx)
        min_soc, floor = None, None
        for m in self._pct.finditer(t):
            value = min(int(m.group(1)), 100) / 100.0
            if self._below_before.search(t[max(0, m.start() - 15):m.start()]) or self._below_after.search(t[m.end():]):
                floor = value
            elif min_soc is None:
                min_soc = value
        v2g = bool(ctx.v2g_capable and self._v2g_yes.search(t) and not self._v2g_no.search(t))
        kwh = self._kwh.search(t)
        priority = "urgent" if self._urgent.search(t) else "cost" if self._cheap.search(t) else "normal"
        target = min_soc if min_soc is not None else base.min_soc
        out = {
            "deadline": self._deadline(t) or base.deadline,
            "min_soc": target,
            "v2g_allowed": v2g,
            "soc_floor": min(floor if floor is not None else base.soc_floor, target),
            "max_discharge_kwh": float(kwh.group(1)) if (v2g and kwh) else 0.0,
            "priority": priority,
        }
        return json.dumps(out, ensure_ascii=False)


SYSTEM_PROMPT = (
    "You convert EV charging requests (English, Hindi or Bengali) into JSON with exactly these keys: "
    "deadline (HH:MM, 24 h), min_soc (0-1), v2g_allowed (bool), soc_floor (0-1, <= min_soc), "
    "max_discharge_kwh (>= 0, 0 unless v2g_allowed), priority (normal|cost|urgent). "
    "If no discharge limit is stated, set v2g_allowed to false. Output only the JSON object."
)


class HFParser:
    """LLM parser (paper: Llama-3.1-8B-Instruct + LoRA). Requires ``pip install .[llm]``."""

    def __init__(self, model_name: str, adapter_path: str | None = None, max_new_tokens: int = 128):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("HFParser needs the optional dependencies: pip install '.[llm]'") from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def build_messages(text: str, ctx: SessionContext, feedback: list[str] | None = None) -> list[dict[str, str]]:
        user = f"Session context: {ctx.model_dump_json()}\nRequest: {text}"
        if feedback:
            user += "\nYour previous answer was invalid: " + "; ".join(feedback) + ". Return corrected JSON."
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def parse(self, text: str, ctx: SessionContext, feedback: list[str] | None = None) -> str:
        msgs = self.build_messages(text, ctx, feedback)
        ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        with self._torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)


def target_json(con: ChargingConstraint) -> str:
    """Training target for LoRA fine-tuning (see scripts/finetune_intent_lora.py)."""
    return con.model_dump_json()
