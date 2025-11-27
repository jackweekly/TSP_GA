import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

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


def _choose_instances(instances):
    instances = sorted(instances, key=lambda inst: (len(inst.graph), inst.name.lower()))
    if not sys.stdin.isatty():
        return [instances[0]]
    print("Select instance(s) by number (comma separated) or press Enter for first:")
    display = instances[:20]
    for idx, inst in enumerate(display):
        print(f"[{idx:02d}] {inst.name:<12} {len(inst.graph):>5} nodes")
    if len(instances) > len(display):
        print(f"... ({len(instances) - len(display)} more not shown)")
    choice = input("Choice: ").strip()
    if not choice:
        return [instances[0]]
    picks = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            i = int(part)
            if 0 <= i < len(instances):
                picks.append(instances[i])
    return picks or [instances[0]]


def run(args) -> None:
    data_root = Path(args.data_root)
    print(f"[info] loading data from {data_root}")
    instances = load_data(data_root)
    if not instances:
        raise RuntimeError(
            f"No TSPLIB instances found in {data_root}. "
            "Place .tsp (and optional .opt.tour) files there before running."
        )
    selected = _choose_instances(instances)
    graphs = [inst.graph for inst in selected]
    optima = [inst.optimum for inst in selected]
    names = ", ".join(inst.name for inst in selected)
    print(f"[info] using instances: {names} (count={len(graphs)})")

    cfg = IslandConfig(
        population_size=30,
        islands=2,
        migration_interval=5,
        migrants=2,
        evaluation_samples=1,
        max_runtime=1.0,
        runtime_weight=0.1,
        random_seed=123,
    )
    model = build_model(graphs, optima, resume=True, cfg=cfg)

    print("[info] running continuously; Ctrl+C to stop.")
    try:
        while True:
            model.step()
            best, score = model.best()
            if best is None:
                print(f"gen {model.generation}: no valid genome scored yet (score={score})")
            else:
                print(f"gen {model.generation}: best ops={best.ops} score={score:.2f}")
            _print_island_tops(model, top_k=3)
            save_checkpoint(model)
    except KeyboardInterrupt:
        print("Interrupted. Checkpoint saved.")


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


def _print_island_tops(model: IslandModel, top_k: int = 3) -> None:
    lines: List[str] = []
    for idx, island in enumerate(model.islands):
        scored = [(island.evaluate_genome(g), g) for g in island.population]
        scored.sort(key=lambda x: x[0])
        best_score, best_genome = scored[0]
        avg_score = sum(s for s, _ in scored) / len(scored)
        top_ops = " ".join(best_genome.ops)
        lines.append(f"[island {idx}] best={best_score:8.2f} avg={avg_score:8.2f} ops={top_ops}")
    print(" | ".join(lines))


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

    run_parser = subparsers.add_parser("run", help="Run / resume evolutionary search (continuous, two islands)")
    run_parser.add_argument("--data-root", default="data/tsplib")
    run_parser.set_defaults(func=run)

    data_parser = subparsers.add_parser("data", help="Inspect current checkpoint")
    data_parser.add_argument("--data-root", default="data/tsplib")
    data_parser.set_defaults(func=data)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
