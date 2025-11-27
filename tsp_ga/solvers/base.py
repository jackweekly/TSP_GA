import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import networkx as nx


Tour = List[int]


def tour_length(graph: nx.Graph, tour: Sequence[int]) -> float:
    dist = 0.0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        dist += graph[a][b]["weight"]
    return float(dist)


class Solver(ABC):
    name: str = "base"

    @abstractmethod
    def solve(self, graph: nx.Graph) -> Tour:
        raise NotImplementedError


@dataclass
class SolveResult:
    tour: Tour
    length: float
    solver_name: str
    optimum: float

    @property
    def gap(self) -> float:
        if self.optimum is None or math.isclose(self.optimum, 0.0):
            return float("inf")
        return (self.length - self.optimum) / self.optimum
