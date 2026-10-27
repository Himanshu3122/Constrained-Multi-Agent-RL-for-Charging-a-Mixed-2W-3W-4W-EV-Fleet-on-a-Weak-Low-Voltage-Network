"""Demo trilingual request corpus (NOT the paper's corpus).

The paper's corpus (14,400 requests from 220 seed templates, LLM paraphrases,
annotator audit) is not released. This module generates a small, clearly
labelled demo corpus from a handful of hand-written templates per intent type
(deadline charging, V2G opt-in with limits, cost-priority, urgent top-up) in
English, Hindi and Bengali. It exists to exercise the data format, the LoRA
fine-tuning script and the intent metrics. Because the rule-based parser was
written against the same phrasing, its scores on this corpus are not a
meaningful accuracy estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solace.intent.schema import ChargingConstraint, SessionContext

INTENT_TYPES = ("deadline", "v2g", "cost", "urgent")

# (template, slots) per language and intent. Slots: {soc}, {hour} (12 h clock), {kwh}, {floor}.
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "en": {
        "deadline": ["I need {soc}% by {hour} am, no V2G please.", "Charge to {soc} percent by {hour} am."],
        "v2g": ["Need {soc}% by {hour} am, you can discharge up to {kwh} kWh but not below {floor}%.",
                "Sell up to {kwh} kWh tonight but not below {floor}%, and I need {soc}% by {hour} am."],
        "cost": ["Charge me cheaply to {soc}% by {hour} am.", "Lowest cost is fine, {soc}% by {hour} am."],
        "urgent": ["Urgent: {soc}% by {hour} am please.", "Need {soc}% asap, by {hour} am at the latest."],
    },
    "hi": {
        "deadline": ["सुबह {hour} बजे तक {soc}% चार्ज चाहिए, बिजली मत बेचो।", "सुबह {hour} बजे तक {soc} प्रतिशत कर दो।"],
        "v2g": ["सुबह {hour} बजे तक {soc}% चाहिए, {kwh} यूनिट तक डिस्चार्ज कर सकते हो पर {floor}% से नीचे नहीं।"],
        "cost": ["सस्ते में सुबह {hour} बजे तक {soc}% कर देना।"],
        "urgent": ["जल्दी चाहिए, सुबह {hour} बजे तक {soc}%।"],
    },
    "bn": {
        "deadline": ["সকাল {hour}টার মধ্যে {soc}% চার্জ চাই, বিক্রি করবেন না।"],
        "v2g": ["সকাল {hour}টার মধ্যে {soc}% চাই, {kwh} ইউনিট পর্যন্ত ডিসচার্জ করতে পারেন কিন্তু {floor}%-এর নিচে নয়।"],
        "cost": ["সস্তায় সকাল {hour}টার মধ্যে {soc}% করে দিন।"],
        "urgent": ["জরুরি, সকাল {hour}টার মধ্যে {soc}% চাই।"],
    },
}


def intent_type(con: ChargingConstraint) -> str:
    if con.priority.value == "urgent":
        return "urgent"
    if con.v2g_allowed:
        return "v2g"
    if con.priority.value == "cost":
        return "cost"
    return "deadline"


def generate_demo_corpus(n_per_cell: int, seed: int = 0) -> list[dict]:
    """Return records {text, language, intent, context, constraint} (demo data)."""
    rng = np.random.default_rng(seed)
    records = []
    for lang, by_intent in TEMPLATES.items():
        for intent, templates in by_intent.items():
            for _ in range(n_per_cell):
                soc = int(rng.choice([60, 70, 80, 90]))
                hour = int(rng.integers(6, 10))
                kwh = int(rng.integers(1, 6))
                floor = int(rng.choice([30, 40, 50]))
                text = str(rng.choice(templates)).format(soc=soc, hour=hour, kwh=kwh, floor=floor)
                ctx = SessionContext(vehicle_class="4W", battery_kwh=40.0, soc=0.3, charger_kw=7.4,
                                     plug_in_time="20:00", departure_time="10:00", v2g_capable=True)
                v2g = intent == "v2g"
                con = ChargingConstraint(
                    deadline=f"{hour:02d}:00", min_soc=soc / 100, v2g_allowed=v2g,
                    soc_floor=floor / 100 if v2g else min(0.2, soc / 100),
                    max_discharge_kwh=float(kwh) if v2g else 0.0,
                    priority="urgent" if intent == "urgent" else "cost" if intent == "cost" else "normal")
                records.append({"text": text, "language": lang, "intent": intent, "demo_data": True,
                                "context": ctx.model_dump(), "constraint": con.model_dump(mode="json")})
    return records


def save_jsonl(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
