import argparse
import json
from pathlib import Path
from typing import Tuple

from tsp_ga.data import load_data, split_by_hash
from tsp_ga.island import IslandConfig, IslandModel


CHECKPOINT_PATH = Path("checkpoints/island_state.json")


def save_checkpoint(model: IslandModel, path: Path = CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_state(), indent=2))


def load_checkpoint(graphs, optima, path: Path = CHECKPOINT_PATH) -> IslandModel:
    state = json.loads(path.read_text())
    return IslandModel.from_state(state, graphs, optima)


def build_model(graphs, optima, resume: bool, cfg: IslandConfig) -> IslandModel:
    if resume and CHECKPOINT_PATH.exists():
        print(f"Resuming from {CHECKPOINT_PATH}")
        return load_checkpoint(graphs, optima, CHECKPOINT_PATH)
    print("Starting new model")
    return IslandModel(cfg, graphs, optima)


def run(args) -> None:
    data_root = Path(args.data_root)
    instances = load_data(
        data_root,
        max_nodes=args.max_nodes,
        max_instances=args.max_instances,
    )
    dims = [len(inst.graph) for inst in instances]
    with_opt = sum(1 for inst in instances if inst.optimum is not None)
    if not instances:
        raise RuntimeError(
            f"No TSPLIB instances found in {data_root}. "
            "Place .tsp (and optional .opt.tour) files there before running."
        )
    print(
        f"Loaded {len(instances)} instances "
        f"(nodes: min={min(dims)} max={max(dims)}, optima={with_opt}/{len(instances)})"
    )
    if args.verbose:
        preview = ", ".join(
            f"{inst.name}({len(inst.graph)})" for inst in instances[: min(10, len(instances))]
        )
        print(f"Preview: {preview}")
    splits = split_by_hash(instances)
    train = splits["train"] or instances
    graphs = [inst.graph for inst in train]
    optima = [inst.optimum for inst in train]

    cfg = IslandConfig(
        population_size=args.population,
        islands=args.islands,
        migration_interval=args.migration_interval,
        migrants=args.migrants,
        evaluation_samples=args.samples,
        max_runtime=args.max_runtime,
        runtime_weight=args.runtime_weight,
        random_seed=args.seed,
    )
    model = build_model(graphs, optima, resume=not args.fresh, cfg=cfg)

    for g in range(args.generations):
        model.step()
        best, score = model.best()
        if best is None:
            print(f"gen {model.generation}: no valid genome scored yet (score={score})")
        else:
            print(f"gen {model.generation}: best ops={best.ops} score={score:.2f}")
        if args.verbose:
            print(f"  checkpoint -> {CHECKPOINT_PATH}")
        save_checkpoint(model)


def island_insights(model: IslandModel) -> Tuple[str, str]:
    details = []
    for idx, island in enumerate(model.islands):
        scored = [(island.evaluate_genome(g), g) for g in island.population]
        scored.sort(key=lambda x: x[0])
        best_score, best_genome = scored[0]
        avg_score = sum(s for s, _ in scored) / len(scored)
        details.append(
            f"island {idx}: best_score={best_score:.2f} best_ops={best_genome.ops} avg_score={avg_score:.2f}"
        )
    best, global_score = model.best()
    header = f"generation={model.generation}, global_best_score={global_score:.2f}, best_ops={best.ops}"
    return header, "\n".join(details)


def data(args) -> None:
    data_root = Path(args.data_root)
    if not CHECKPOINT_PATH.exists():
        print("No checkpoint found; run `make run` first.")
        return
    instances = load_data(data_root)
    splits = split_by_hash(instances)
    eval_set = splits["val"] or splits["train"] or instances
    graphs = [inst.graph for inst in eval_set]
    optima = [inst.optimum for inst in eval_set]
    model = load_checkpoint(graphs, optima, CHECKPOINT_PATH)
    header, body = island_insights(model)
    print(header)
    print(body)


def main():
    parser = argparse.ArgumentParser(description="TSP GA CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run / resume evolutionary search")
    run_parser.add_argument("--data-root", default="data/tsplib")
    run_parser.add_argument("--generations", type=int, default=5)
    run_parser.add_argument("--population", type=int, default=12)
    run_parser.add_argument("--islands", type=int, default=2)
    run_parser.add_argument("--migration-interval", type=int, default=3)
    run_parser.add_argument("--migrants", type=int, default=2)
    run_parser.add_argument("--samples", type=int, default=3)
    run_parser.add_argument("--max-runtime", type=float, default=2.0)
    run_parser.add_argument("--runtime-weight", type=float, default=0.1)
    run_parser.add_argument("--seed", type=int, default=123)
    run_parser.add_argument(
        "--max-nodes",
        type=int,
        default=500,
        help="Skip instances with more nodes than this (to avoid huge graphs by default).",
    )
    run_parser.add_argument(
        "--max-instances",
        type=int,
        default=100,
        help="Limit number of instances loaded (None for all).",
    )
    run_parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints")
    run_parser.add_argument("--verbose", action="store_true", help="Print extra debug info")
    run_parser.set_defaults(func=run)

    data_parser = subparsers.add_parser("data", help="Inspect current checkpoint")
    data_parser.add_argument("--data-root", default="data/tsplib")
    data_parser.set_defaults(func=data)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
