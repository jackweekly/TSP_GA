PYTHON ?= python
CHECKPOINT ?= checkpoints/island_state.json

.PHONY: run data

run:
	@$(PYTHON) -m tsp_ga.cli run

data:
	@$(PYTHON) -m tsp_ga.cli data
