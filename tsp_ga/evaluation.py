import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import torch

from .solvers.base import SolveResult, Solver, tour_length


@dataclass
class Fitness:
    length: float
    runtime: float
    gap: float
    score: float
    solver_name: str


def _tour_length_torch(dist: torch.Tensor, tour) -> float:
    idx = torch.tensor(tour, device=dist.device, dtype=torch.long)
    a = idx
    b = idx.roll(-1)
    return dist[a, b].sum().item()


def evaluate_solver(
    solver: Solver,
    graph: nx.Graph,
    optimum: Optional[float],
    max_runtime: float = 5.0,
    runtime_weight: float = 0.1,
    dist_mat: Optional[torch.Tensor] = None,
) -> Fitness:
    start = time.perf_counter()
    tour = solver.solve(graph)
    runtime = time.perf_counter() - start
    if runtime > max_runtime:
        # Penalize slow solvers heavily.
        return Fitness(
            length=float("inf"),
            runtime=runtime,
            gap=float("inf"),
            score=float("inf"),
            solver_name=solver.__class__.__name__,
        )
    if dist_mat is not None and torch.is_tensor(dist_mat):
        length = _tour_length_torch(dist_mat, tour)
    else:
        length = tour_length(graph, tour)
    gap = float("inf") if optimum is None else (length - optimum) / optimum
    score = length + runtime_weight * length * runtime
    return Fitness(
        length=length,
        runtime=runtime,
        gap=gap,
        score=score,
        solver_name=solver.__class__.__name__,
    )


def aggregate_fitness(fitnesses: List[Fitness]) -> Dict[str, float]:
    if not fitnesses:
        return {"score": float("inf"), "gap": float("inf"), "runtime": float("inf")}
    score = sum(f.score for f in fitnesses) / len(fitnesses)
    gap = sum(f.gap for f in fitnesses if f.gap != float("inf")) / max(
        1, sum(1 for f in fitnesses if f.gap != float("inf"))
    )
    runtime = sum(f.runtime for f in fitnesses) / len(fitnesses)
    return {"score": score, "gap": gap, "runtime": runtime}
