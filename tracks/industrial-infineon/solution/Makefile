# Use the local venv python if present (local smoke); override on the cluster, e.g. `make smoke PY=python`.
PY ?= $(shell [ -x ./.venv/bin/python ] && echo ./.venv/bin/python || echo python)
SOL := $(CURDIR)

.PHONY: test smoke data train-decoder train-encoder eval submit demo clean

test:
	$(PY) -m pytest -q

data:
	$(PY) -m procseq.build_data --n-per-family 2000 --seed 42

smoke:
	$(PY) -m pytest -q
	$(PY) -m procseq.build_data --smoke
	$(PY) -m procseq.train_decoder --config configs/smoke.yaml
	$(PY) -m procseq.train_encoder --config configs/smoke.yaml
	$(PY) -m procseq.infer --all --config configs/smoke.yaml
	$(PY) -m procseq.run_eval --config configs/smoke.yaml
	@echo "SMOKE OK"

train-decoder:
	accelerate launch -m procseq.train_decoder --config configs/leonardo_decoder.yaml

train-encoder:
	accelerate launch -m procseq.train_encoder --config configs/leonardo_encoder.yaml

eval:
	$(PY) -m procseq.run_eval --config configs/leonardo_decoder.yaml

demo:
	$(PY) -m procseq.demo --config configs/leonardo_decoder.yaml

clean:
	rm -rf artifacts/* runs/*
