PYTHON ?= python
CHECKPOINT ?= checkpoints/island_state.json
TSPLIB_DIR ?= data/tsplib
TSPLIB_REPO ?= https://github.com/mastqe/tsplib.git
ARGS ?=

.PHONY: install run data

install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using existing uv"; \
	else \
		echo "uv not found; installing via pip"; \
		$(PYTHON) -m pip install --upgrade uv; \
	fi
	uv pip install -r requirements.txt
	@if ls $(TSPLIB_DIR)/*.tsp >/dev/null 2>&1; then \
		echo "TSPLIB data already present in $(TSPLIB_DIR)"; \
	else \
		echo "Fetching TSPLIB data from $(TSPLIB_REPO)"; \
		tmp_dir=$$(mktemp -d); \
		git clone --depth 1 $(TSPLIB_REPO) $$tmp_dir/tsplib >/dev/null; \
		mkdir -p $(TSPLIB_DIR); \
		cp $$tmp_dir/tsplib/*.tsp $(TSPLIB_DIR)/; \
		if [ -d $$tmp_dir/tsplib/solutions ]; then cp -r $$tmp_dir/tsplib/solutions $(TSPLIB_DIR)/; fi; \
		rm -rf $$tmp_dir; \
		rm -f $(TSPLIB_DIR)/manifest.json; \
	fi

run:
	@$(PYTHON) -m tsp_ga.cli run $(ARGS)

data:
	@$(PYTHON) -m tsp_ga.cli data

stop:
	@echo "Stopping any running TSP_GA processes..."
	@pkill -f "tsp_ga.cli run" 2>/dev/null && echo "Terminated matching python runs" || echo "No matching processes found"
