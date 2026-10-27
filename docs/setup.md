# Setup

Tested with Python 3.12 on Windows (CPU). Python 3.10–3.13 should work; Python 3.14+ is not supported because PyTorch wheels may be missing.

## From a clean environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# CPU-only PyTorch (skip the index URL if you want the CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

With [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e ".[dev]"
```

Exact versions used for validation are pinned in `requirements.txt`.

## Optional: LLM intent parser

```bash
pip install -e ".[llm]"          # transformers, peft, accelerate
export HF_TOKEN=...               # Llama 3.1 is a gated model
```

An 8B model needs a GPU (or a lot of RAM) for inference and fine-tuning. Everything else in the repository works without this extra.

## Check the installation

```bash
python -c "import solace; print(solace.__version__)"
python -m pytest            # unit + integration tests (several minutes on CPU)
python scripts/smoke_test.py
```

## Data

No external data is needed for the synthetic pipeline. To use real inputs:

| Input | Config key | Format |
|---|---|---|
| ACN-Data sessions (4W) | `data.acn_path` | CSV export with `connectionTime`, `disconnectTime`, `kWhDelivered` |
| Hourly irradiance | `data.irradiance_path` | CSV with a `ghi` column (W/m²), chronological from 00:00 |
| Custom feeder (e.g. IEEE 34-bus) | `feeder.source: json`, `feeder.json_path` | see `solace/grid/feeder.py::load_feeder_json` |

See [../data/README.md](../data/README.md).
