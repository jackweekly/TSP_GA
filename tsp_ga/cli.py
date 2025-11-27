import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

from tsp_ga.data import load_data, split_by_hash
from tsp_ga.island import IslandConfig, IslandModel


CHECKPOINT_PATH = Path("checkpoints/island_state.json")

PRESETS = [
    {
        "name": "Quick small (berlin52/att48)",
        "desc": "One-shot small instances, 3 gens, pop 8, 1 island",
        "overrides": {
            "allow_list": "berlin52,att48",
            "max_nodes": 100,
            "max_instances": 5,
            "generations": 3,
            "population": 8,
            "islands": 1,
            "samples": 1,
            "max_runtime": 1.0,
        },
    },
    {
        "name": "Small batch (default)",
        "desc": "≤500 nodes, up to 25 instances, 3 gens, pop 8",
        "overrides": {},
    },
    {
        "name": "Medium batch",
        "desc": "≤1000 nodes, up to 100 instances, 5 gens, pop 12",
        "overrides": {
            "max_nodes": 1000,
            "max_instances": 100,
            "generations": 5,
            "population": 12,
            "samples": 2,
            "max_runtime": 2.0,
        },
    },
    {
        "name": "Use CLI args",
        "desc": "Skip overrides; use provided flags",
        "overrides": None,
    },
]


def _select_preset() -> dict:
    if not sys.stdin.isatty():
        return PRESETS[1]
    try:
        import curses
    except Exception:
        return PRESETS[1]

    def _menu(stdscr):
        curses.curs_set(0)
        current = 0
        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, "Select run preset (arrow keys, Enter to confirm)")
            for i, opt in enumerate(PRESETS):
                prefix = "> " if i == current else "  "
                stdscr.addstr(i + 2, 0, f"{prefix}{opt['name']} - {opt['desc']}")
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                current = (current - 1) % len(PRESETS)
            elif key in (curses.KEY_DOWN, ord("j")):
                current = (current + 1) % len(PRESETS)
            elif key in (curses.KEY_ENTER, 10, 13):
                return PRESETS[current]

    try:
        return curses.wrapper(_menu)
    except Exception:
        return PRESETS[1]


def _apply_overrides(args, overrides: dict) -> None:
    if not overrides:
        return
    for k, v in overrides.items():
        setattr(args, k, v)


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
    if not args.no_menu:
        preset = _select_preset()
        if preset["overrides"] is not None:
            _apply_overrides(args, preset["overrides"])
        print(f"[preset] {preset['name']}")
    data_root = Path(args.data_root)
    print(f"[info] loading data from {data_root} (max_nodes={args.max_nodes}, max_instances={args.max_instances})")
    instances = load_data(
        data_root,
        max_nodes=args.max_nodes,
        max_instances=args.max_instances,
        allow_list=args.allow_list,
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
        print(f"Split ratios train/val/test default to 70/15/15 via hash.")
    splits = split_by_hash(instances)
    train = splits["train"] or instances
    graphs = [inst.graph for inst in train]
    optima = [inst.optimum for inst in train]
    print(f"[info] train set size={len(graphs)}")

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
        if args.verbose:
        print(f"[info] generation {model.generation+1} start")
        model.step()
        best, score = model.best()
        if best is None:
            print(f"gen {model.generation}: no valid genome scored yet (score={score})")
        else:
            print(f"gen {model.generation}: best ops={best.ops} score={score:.2f}")
        if args.verbose:
            print(f"  checkpoint -> {CHECKPOINT_PATH}")
        save_checkpoint(model)
        if args.continuous:
            continue
        if g == args.generations - 1:
            break
    # Continuous mode: keep evolving until interrupted.
    if args.continuous:
        gen_counter = args.generations
        try:
            while True:
                gen_counter += 1
                if args.verbose:
                    print(f"[info] generation {model.generation+1} start")
                model.step()
                best, score = model.best()
                if best is None:
                    print(f"gen {model.generation}: no valid genome scored yet (score={score})")
                else:
                    print(f"gen {model.generation}: best ops={best.ops} score={score:.2f}")
                if args.log_top > 0:
                    _print_island_tops(model, top_k=args.log_top)
                save_checkpoint(model)
        except KeyboardInterrupt:
            print("Continuous run interrupted. Checkpoint saved.")


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
    for idx, island in enumerate(model.islands):
        scored = [(island.evaluate_genome(g), g) for g in island.population]
        scored.sort(key=lambda x: x[0])
        tops = ", ".join(f"{s:.1f}:{g.ops}" for s, g in scored[:top_k])
        print(f"  island {idx}: {tops}")


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

    run_parser = subparsers.add_parser("run", help="Run / resume evolutionary search (interactive menu by default)")
    run_parser.add_argument("--data-root", default="data/tsplib")
    run_parser.add_argument("--generations", type=int, default=5)
    run_parser.add_argument("--population", type=int, default=20)
    run_parser.add_argument("--islands", type=int, default=1)
    run_parser.add_argument("--migration-interval", type=int, default=5)
    run_parser.add_argument("--migrants", type=int, default=1)
    run_parser.add_argument("--samples", type=int, default=1, help="Evaluation samples per genome")
    run_parser.add_argument("--max-runtime", type=float, default=1.0)
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
        default=25,
        help="Limit number of instances loaded (None for all).",
    )
    run_parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints")
    run_parser.add_argument("--verbose", action="store_true", help="Print extra debug info")
    run_parser.add_argument(
        "--allow-list",
        type=str,
        default=None,
        help="Comma-separated instance names to allow (e.g., berlin52,att48). If set, only these load.",
    )
    run_parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Skip interactive preset menu and use CLI flags as provided.",
    )
    run_parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run generations indefinitely until interrupted (Ctrl+C).",
    )
    run_parser.add_argument(
        "--log-top",
        type=int,
        default=2,
        help="In continuous mode, log top-k scores per island each generation.",
    )
    run_parser.set_defaults(func=run)

    data_parser = subparsers.add_parser("data", help="Inspect current checkpoint")
    data_parser.add_argument("--data-root", default="data/tsplib")
    data_parser.set_defaults(func=data)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
