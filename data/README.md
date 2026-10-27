# Data

Nothing in this directory is committed except this file. All generated data is ignored by git.

| Path | Created by | Content |
|---|---|---|
| `data/processed/<scenario_set>/` | `scripts/preprocess.py` | `scenarios.npz` (feeder, fleet, base load, PV, tariff) and `sessions.json` (sessions + constraints per day and split) |
| `data/corpus/*.jsonl` | `scripts/intent_corpus.py generate` | **Demo** trilingual requests (not the paper's corpus) |

## External data referenced by the paper (not included)

| Dataset | Used for | Where to get it | How to plug it in |
|---|---|---|---|
| ACN-Data (Caltech/JPL) | 4W arrival, dwell and energy distributions | https://ev.caltech.edu/dataset (registration and terms of use apply) | Export sessions as CSV with `connectionTime`, `disconnectTime`, `kWhDelivered`; set `data.acn_path` |
| Hourly irradiance for West Bengal | PV generation | e.g. a satellite irradiance product of your choice (the paper does not name its source) | CSV with a `ghi` column (W/m²), hourly from 00:00; set `data.irradiance_path` |
| IEEE 34-bus test feeder | Comparison testbed | IEEE PES Distribution Test Feeders | Convert to the JSON schema in `solace/grid/feeder.py::load_feeder_json`; set `feeder.source: json` |
| Indian 2W/3W charging statistics | Generator calibration | Bolt.Earth figures reported by Autocar Professional (Sep 2026) | Encoded as arrival-time parameters in `configs/default.yaml` (assumption A4) |
| Trilingual request corpus (14,400 requests) | Intent layer | Not released with the paper | Use the JSONL format of `solace/intent/corpus.py` |

Without these, every pipeline stage runs on clearly labelled synthetic data. Synthetic results are not comparable to the paper.
