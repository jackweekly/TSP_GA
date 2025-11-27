import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import tsplib95


@dataclass
class Instance:
    name: str
    path: Path
    graph: nx.Graph
    optimum: Optional[float]


def hash_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _solution_candidates(path: Path) -> Iterable[Path]:
    yield path.with_suffix(".opt.tour")
    for ext in (".opt.tour", ".opt", ".tour"):
        yield path.parent / "solutions" / f"{path.stem}{ext}"


def _read_dimension(path: Path) -> Optional[int]:
    try:
        with path.open("r") as f:
            for line in f:
                if "DIMENSION" in line.upper():
                    parts = line.replace(":", " ").split()
                    for token in parts:
                        if token.isdigit():
                            return int(token)
        return None
    except Exception:
        return None


def _load_optimum(problem, path: Path) -> Optional[float]:
    for candidate in _solution_candidates(path):
        if not candidate.exists():
            continue
        try:
            tour_file = tsplib95.parse(candidate.read_text())
            nodes = list(tour_file.tours[0])
            dist = 0.0
            for i in range(len(nodes)):
                a = nodes[i]
                b = nodes[(i + 1) % len(nodes)]
                dist += problem.get_weight(a, b)
            return float(dist)
        except Exception:
            continue
    return None


def load_instance(path: Path) -> Instance:
    problem = tsplib95.load(path)
    graph = problem.get_graph()
    optimum = _load_optimum(problem, path)
    return Instance(name=problem.name, path=path, graph=graph, optimum=optimum)


def load_tsplib_instances(
    root: Path, max_nodes: Optional[int] = None, max_instances: Optional[int] = None
) -> List[Instance]:
    tsp_files = sorted(root.glob("*.tsp"))
    instances: List[Instance] = []
    for p in tsp_files:
        if max_nodes is not None:
            dim = _read_dimension(p)
            if dim is not None and dim > max_nodes:
                continue
        instances.append(load_instance(p))
        if max_instances is not None and len(instances) >= max_instances:
            break
    return instances


def split_by_hash(
    instances: Iterable[Instance], ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Dict[str, List[Instance]]:
    train_r, val_r, test_r = ratios
    total = train_r + val_r + test_r
    if not np.isclose(total, 1.0):
        raise ValueError("Ratios must sum to 1.")
    buckets = {"train": [], "val": [], "test": []}
    for inst in instances:
        h = hash_file(inst.path)
        val = int(h, 16) % 1000 / 1000.0
        if val < train_r:
            buckets["train"].append(inst)
        elif val < train_r + val_r:
            buckets["val"].append(inst)
        else:
            buckets["test"].append(inst)
    return buckets


def cache_manifest(instances: List[Instance], cache_path: Path) -> None:
    cache = [
        {
            "name": inst.name,
            "path": str(inst.path),
            "optimum": inst.optimum,
            "hash": hash_file(inst.path),
        }
        for inst in instances
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def load_manifest(cache_path: Path) -> Optional[List[Dict]]:
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text())


def load_data(
    root: Path,
    use_cache: bool = True,
    max_nodes: Optional[int] = None,
    max_instances: Optional[int] = None,
) -> List[Instance]:
    cache_path = root / "manifest.json"
    if use_cache:
        cached = load_manifest(cache_path)
        if cached:
            instances = []
            for item in cached:
                p = Path(item["path"])
                if not p.exists():
                    continue
                inst = load_instance(p)
                if hash_file(p) != item["hash"]:
                    continue
                if max_nodes is not None:
                    dim = len(inst.graph)
                    if dim > max_nodes:
                        continue
                instances.append(inst)
            if instances:
                return instances
    instances = load_tsplib_instances(root, max_nodes=max_nodes, max_instances=max_instances)
    if use_cache:
        cache_manifest(instances, cache_path)
    return instances
