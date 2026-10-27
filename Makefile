PY ?= python
CONFIG ?= configs/smoke.yaml
SEED ?= 0

.PHONY: install lint test smoke preprocess train evaluate api clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src scripts tests

test:
	$(PY) -m pytest

smoke:
	$(PY) scripts/smoke_test.py

preprocess:
	$(PY) scripts/preprocess.py --config $(CONFIG)

train:
	$(PY) scripts/train.py --config $(CONFIG) --seed $(SEED)

evaluate:
	$(PY) scripts/evaluate.py --config $(CONFIG) --checkpoint "checkpoints/$$(basename $(CONFIG) .yaml)/seed{seed}/final.pt"

api:
	$(PY) -m uvicorn --factory solace.api.app:create_app --host 127.0.0.1 --port 8000

clean:
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('.ruff_cache', ignore_errors=True)"
