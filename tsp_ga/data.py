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


def _load_optimum(opt_path: Path) -> Optional[float]:
    if not opt_path.exists():
        return None
    tour_file = tsplib95.parse(opt_path.read_text())
    nodes = list(tour_file.tours[0])
    dist = 0.0
    inst = tsplib95.load(opt_path.with_suffix(".tsp"))
    for i in range(len(nodes)):
        a = nodes[i]
        b = nodes[(i + 1) % len(nodes)]
        dist += inst.get_weight(a, b)
    return float(dist)


def load_instance(path: Path) -> Instance:
    problem = tsplib95.load(path)
    graph = problem.get_graph()
    optimum = _load_optimum(path.with_suffix(".opt.tour"))
    return Instance(name=problem.name, path=path, graph=graph, optimum=optimum)


def load_tsplib_instances(root: Path) -> List[Instance]:
    tsp_files = sorted(root.glob("*.tsp"))
    instances = [load_instance(p) for p in tsp_files]
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


def load_data(root: Path, use_cache: bool = True) -> List[Instance]:
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
                instances.append(inst)
            if instances:
                return instances
    instances = load_tsplib_instances(root)
    if use_cache:
        cache_manifest(instances, cache_path)
    return instances
