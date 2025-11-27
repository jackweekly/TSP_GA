# TSP_GA

Evolutionary framework for discovering new travelling salesman problem (TSP) solvers. The goal is to evolve solver programs (heuristics + local search variants) and evaluate them on TSPLIB-style instances of varying sizes/shapes while guarding against data leakage.

## Concepts
- **Genome-as-program**: Solvers are built from composable operators (constructive heuristics, local search moves, meta-heuristics) encoded as a genome. Mutation/crossover operate on these operators, not raw weights.
- **Island model**: Multiple populations run in parallel (GPU nodes), exchanging elite genomes periodically to avoid premature convergence.
- **No leakage**: Instances are partitioned into train/validation/test with hashing to prevent overlap. Validation drives selection; test is held back for final reporting only.
- **Anytime evaluation**: Fitness combines tour cost and runtime; early exits are supported for slow candidates.

## Quick start
```bash
# create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# populate data/tsplib with .tsp/.opt.tour files (see Data section)
make run   # runs/resumes EA with checkpointing at checkpoints/island_state.json
make data  # prints checkpoint insights (best ops per island, avg scores)
make run ARGS="--allow-list berlin52 --max-nodes 100 --max-instances 5 --generations 3 --population 6 --verbose"
```
`make install` also fetches TSPLIB data/solutions from mastqe/tsplib if no `.tsp` files are present. By default `make run` limits loading to instances with ≤1000 nodes (change with `--max-nodes` or `--max-instances`).

For faster/safer startup on large TSPLIB mirrors, defaults now load ≤500-node instances and at most 25 files, with evaluation_samples=1, population=8, islands=1, and generations=3 to emphasize quick single-shot improvements. Use `--verbose` to print a preview of loaded instances and checkpoint paths. You can restrict loading to specific instances with `--allow-list berlin52,att48` (comma-separated, names without `.tsp`). An interactive preset menu appears when running `make run` (arrow keys + Enter); use `ARGS="--no-menu ..."` to bypass and supply flags directly. Continuous runs are supported via `--continuous` with per-generation top-k logs (`--log-top`).

### Stopping runs
Use `make stop` to send SIGTERM to any `tsp_ga.cli run` processes if they are still active in the background.

## Running islands (2×5080 cloud node)
- Configure per-node island counts and migration in `tsp_ga/island.py` (`IslandConfig`).
- Start one Python process per GPU with different seeds; periodically sync elites by writing/reading JSON genomes or via a small message bus (not included yet).
- Keep validation-only instances on each node; copy test-only instances to a separate offline machine for final scoring.

## Data
Place TSPLIB instances in `data/tsplib`. The loader expects pairs of `.tsp` and `.opt.tour` (when available) files. A small manifest cache is built the first time you load data.

To avoid leakage:
- Use `split_by_hash` in `tsp_ga.data` to generate stable train/val/test splits using file hashes.
- Keep the test split offline; do not use it for evolution or hyper-parameter tuning.

## Repository layout
```
tsp_ga/
  data.py          # TSPLIB loading + stable splits
  evaluation.py    # Fitness evaluation with cost/runtime tradeoff
  evolutionary.py  # Evolutionary loop for a single island
  island.py        # Multi-island orchestration & migration
  solvers/         # Primitive solver building blocks and genome encoding
  examples/        # Runnable scripts
```

## Roadmap
- Add richer operator set (Lin-Kernighan variants, edge assembly crossover).
- Plug in GPU-accelerated neighborhood evaluation and batching.
- Logging/telemetry for large-scale runs on 2×5080 cloud nodes.
- Automated reporting on generalization gaps across graph families.

## Contributing
Issues and PRs welcome. Please keep experiments reproducible and guard against data leakage by respecting the provided split utilities.
