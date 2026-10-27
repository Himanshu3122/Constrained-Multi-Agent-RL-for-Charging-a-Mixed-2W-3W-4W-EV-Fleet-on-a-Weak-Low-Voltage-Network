# Contributing

Thanks for your interest. This is a research codebase, so changes should keep it faithful to the paper and honest about what it can claim.

## Workflow

1. Create a virtual environment and `pip install -e ".[dev]"` (see `docs/setup.md`).
2. Make the change, with tests:
   - `pytest` (unit + integration)
   - `python scripts/smoke_test.py` for anything touching training, evaluation or the API
   - `ruff check src scripts tests`
3. Open a pull request describing what changed and why.

## Rules for research fidelity

- If you implement something the paper does not specify, add or update an entry in `docs/assumptions.md` and reference its ID in the code.
- Update `docs/paper_implementation_mapping.md` and `IMPLEMENTATION_STATUS.md` when a component's status changes.
- Never commit result tables as paper results. Locally computed numbers must say how they were produced (config, seed, data).
- Do not commit data, checkpoints or secrets (`.gitignore` covers the defaults).

## Style

- Python 3.10+, type hints, docstrings for public functions, `ruff` clean (line length 120).
- Keep research-critical constants in configs, not in code.
