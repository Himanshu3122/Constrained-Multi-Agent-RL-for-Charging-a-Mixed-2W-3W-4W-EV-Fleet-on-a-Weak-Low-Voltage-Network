"""Intent-layer metrics (paper Table V and Fig. 6).

* field accuracy: share of the six fields that match the gold constraint
  (floats within 1e-6). With all six fields always present, micro-averaged
  field-level precision, recall and F1 coincide with this accuracy; the paper
  reports "field F1" without a formal definition (assumption A16).
* exact match: all six fields match.
* infeasible-passed rate: share of outputs that would be registered although
  the feasibility LP says they are infeasible (0 by construction once the
  solver check is enabled).
* intent-type confusion matrix over {deadline, v2g, cost, urgent}.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from solace.intent.corpus import INTENT_TYPES, intent_type
from solace.intent.schema import ChargingConstraint

FIELDS = ("deadline", "min_soc", "v2g_allowed", "soc_floor", "max_discharge_kwh", "priority")


def field_matches(pred: ChargingConstraint, gold: ChargingConstraint) -> dict[str, bool]:
    out = {}
    for f in FIELDS:
        a, b = getattr(pred, f), getattr(gold, f)
        out[f] = abs(a - b) <= 1e-6 if isinstance(a, float) else a == b
    return out


def intent_metrics(preds: list[ChargingConstraint], golds: list[ChargingConstraint]) -> dict:
    if len(preds) != len(golds) or not preds:
        raise ValueError("need equal, non-empty prediction and gold lists")
    matches = [field_matches(p, g) for p, g in zip(preds, golds, strict=True)]
    per_field = {f: float(np.mean([m[f] for m in matches])) for f in FIELDS}
    field_acc = float(np.mean([v for m in matches for v in m.values()]))
    exact = float(np.mean([all(m.values()) for m in matches]))
    conf = Counter((intent_type(g), intent_type(p)) for p, g in zip(preds, golds, strict=True))
    matrix = [[conf.get((t, q), 0) for q in INTENT_TYPES] for t in INTENT_TYPES]
    type_acc = float(np.trace(np.array(matrix)) / len(preds))
    return {"field_f1": field_acc, "exact_match": exact, "per_field": per_field,
            "intent_type_accuracy": type_acc, "confusion_matrix": {"labels": list(INTENT_TYPES), "matrix": matrix},
            "n": len(preds)}
