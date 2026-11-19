PY ?= python3

.PHONY: check test demo export trajectories docker-build docker-check configs

check:            ## единая команда проверки
	$(PY) -m auditgym check

test:
	$(PY) -m pytest -q -p no:cacheprovider tests

demo:
	$(PY) -m auditgym demo --split train --index 8

export:           ## 10 train + 10 dev эпизодов (episodes/ и grader_data/)
	$(PY) -m auditgym export

trajectories:
	$(PY) -m auditgym trajectories --split train

configs:
	$(PY) scripts/make_configs.py

docker-build:
	docker build -t auditgym:1.0.0 .

docker-check: docker-build
	docker run --rm --network none --cpus 2 --memory 4g auditgym:1.0.0
