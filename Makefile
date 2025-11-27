PYTHON ?= python
CHECKPOINT ?= checkpoints/island_state.json

.PHONY: install run data

install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using existing uv"; \
	else \
		echo "uv not found; installing via pip"; \
		$(PYTHON) -m pip install --upgrade uv; \
	fi
	uv pip install -r requirements.txt

run:
	@$(PYTHON) -m tsp_ga.cli run

data:
	@$(PYTHON) -m tsp_ga.cli data
