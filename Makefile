PY := .venv/bin/python
.PHONY: setup corpus models index test lint type eval results mutation bench all
setup:
	uv venv --python 3.12 .venv && uv pip install -e . --group dev
corpus:
	$(PY) scripts/fetch_corpus.py
models:
	$(PY) scripts/fetch_models.py multilingual-e5-small ms-marco-MiniLM-L-6-v2 bge-small-en-v1.5
index:
	$(PY) -m codicil.cli.build --embedder multilingual-e5-small --out index/multilingual-e5-small
	$(PY) -m codicil.cli.build --embedder bge-small-en-v1.5 --out index/bge-small-en-v1.5
test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check src tests eval scripts && $(PY) -m ruff format --check src tests
type:
	$(PY) -m mypy src
eval:
	$(PY) eval/build_dataset.py
	$(PY) eval/run_retrieval.py --split dev  --out eval/results/retrieval-dev.json  > eval/results/retrieval-dev.txt
	$(PY) eval/run_retrieval.py --split test --configs bm25 dense-e5 hybrid-e5 hybrid-e5-routed hybrid-e5-routed-rerank-en --out eval/results/retrieval-test.json > eval/results/retrieval-test.txt
	$(PY) eval/run_answer.py --split dev  --signals eval/results/signals-dev.json  --tune > eval/results/answer-dev.txt
	$(PY) eval/run_answer.py --split test --signals eval/results/signals-test.json --tau-d 0.84 --tau-c 0 > eval/results/answer-test.txt
results:
	$(PY) scripts/make_results.py
mutation:
	$(PY) scripts/mutation_check.py | tee eval/results/mutation.txt
bench:
	$(PY) scripts/bench_latency.py | tee eval/results/latency.json
all: lint type test
