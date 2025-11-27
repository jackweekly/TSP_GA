import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import torch

from tsp_ga.data import load_data, split_by_hash
from tsp_ga.island import IslandConfig, IslandModel


CHECKPOINT_PATH = Path("checkpoints/island_state.json")


def save_checkpoint(model: IslandModel, path: Path = CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_state(), indent=2))
    # Save surrogate models per island if available.
    for idx, island in enumerate(model.islands):
        if hasattr(island, "surrogate") and island.surrogate:
            island.surrogate.save(path.parent / f"surrogate_{idx}.pt")


def load_checkpoint(graphs, optima, path: Path = CHECKPOINT_PATH) -> IslandModel:
    state = json.loads(path.read_text())
    return IslandModel.from_state(state, graphs, optima)


def build_model(graphs, optima, resume: bool, cfg: IslandConfig, dist_mats=None, devices=None, node_maps=None) -> IslandModel:
    if resume and CHECKPOINT_PATH.exists():
        print(f"Resuming from {CHECKPOINT_PATH}")
        state = json.loads(CHECKPOINT_PATH.read_text())
        model = IslandModel.from_state(state, graphs, optima, dist_mats=dist_mats, devices=devices, node_maps=node_maps)
    else:
        print("Starting new model")
        model = IslandModel(cfg, graphs, optima, dist_mats=dist_mats, devices=devices, node_maps=node_maps)
    # Load surrogate weights if present.
    for idx, island in enumerate(model.islands):
        ckpt_path = CHECKPOINT_PATH.parent / f"surrogate_{idx}.pt"
        if hasattr(island, "surrogate") and island.surrogate and ckpt_path.exists():
            island.surrogate.load(ckpt_path)
    return model


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _build_dist_mat(graph: nx.Graph, device: torch.device):
    nodes = list(graph.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    edges = list(graph.edges(data="weight", default=1.0))
    if not edges:
        mat = torch.zeros((len(nodes), len(nodes)), device=device)
        return mat, nodes
    rows = []
    cols = []
    vals = []
    for u, v, w in edges:
        rows.append(idx_map[u])
        cols.append(idx_map[v])
        rows.append(idx_map[v])
        cols.append(idx_map[u])
        vals.extend([w, w])
    indices = torch.tensor([rows, cols], device=device)
    values = torch.tensor(vals, device=device, dtype=torch.float32)
    size = (len(nodes), len(nodes))
    sp = torch.sparse_coo_tensor(indices, values, size=size, device=device)
    return sp.to_dense(), nodes


def _load_or_build_dist_mats(instances, devices, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    dist_mats = [None] * len(instances)
    node_orders = [None] * len(instances)

    def worker(idx_inst):
        idx, inst = idx_inst
        device = devices[idx % len(devices)]
        cache_path = cache_dir / f"{inst.name}.pt"
        if cache_path.exists():
            data = torch.load(cache_path, map_location=device)
            if isinstance(data, dict) and "dist" in data and "nodes" in data:
                mat = data["dist"]
                nodes = data["nodes"]
            else:
                mat = data
                nodes = list(inst.graph.nodes())
        else:
            mat, nodes = _build_dist_mat(inst.graph, device=device)
            torch.save({"dist": mat.cpu(), "nodes": nodes}, cache_path)
        if mat.dtype != torch.float16:
            mat = mat.to(torch.float16)
        dist_mats[idx] = mat.to(device)
        node_orders[idx] = {n: i for i, n in enumerate(nodes)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(instances))) as ex:
        list(ex.map(worker, enumerate(instances)))
    return dist_mats, node_orders


def _choose_instances(instances):
    # Use all instances to encourage general algorithms (no user prompt).
    return instances


def run(args) -> None:
    t0 = time.perf_counter()
    data_root = Path(args.data_root)
    log(f"loading data from {data_root}")
    instances = load_data(data_root)
    if not instances:
        raise RuntimeError(
            f"No TSPLIB instances found in {data_root}. "
            "Place .tsp (and optional .opt.tour) files there before running."
        )
    # Deduplicate by name.
    seen = set()
    selected = []
    for inst in instances:
        if inst.name in seen:
            continue
        seen.add(inst.name)
        selected.append(inst)
    t_load = time.perf_counter()
    log(f"loaded {len(selected)} instances in {t_load - t0:.2f}s")
    graphs = [inst.graph for inst in selected]
    optima = [inst.optimum for inst in selected]
    device_count = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(device_count)] if device_count else [torch.device("cpu")]
    dist_cache = data_root / ".cache"
    t_cache_start = time.perf_counter()
    log("building/loading distance matrices...")
    dist_mats, node_maps = _load_or_build_dist_mats(selected, devices, dist_cache)
    t_cache_end = time.perf_counter()
    log(f"distance matrices ready in {t_cache_end - t_cache_start:.2f}s (cache: {dist_cache})")
    names = ", ".join(inst.name for inst in selected)
    log(f"using instances: {names} (count={len(graphs)}), devices={devices}")

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
    model = build_model(graphs, optima, resume=True, cfg=cfg, dist_mats=dist_mats, devices=devices, node_maps=node_maps)

    log("running continuously; Ctrl+C to stop.")
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
