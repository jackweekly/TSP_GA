import random
from dataclasses import dataclass, field
from typing import List

from .base import Solver
from .heuristics import (
    NearestNeighborSolver,
    RandomInsertionSolver,
    TwoOptLocalSearch,
)


PRIMITIVES = ["nearest_neighbor", "random_insertion", "two_opt"]


def build_solver(ops: List[str]) -> Solver:
    if not ops:
        return NearestNeighborSolver()
    solver: Solver = NearestNeighborSolver()
    for op in ops:
        if op == "nearest_neighbor":
            solver = NearestNeighborSolver()
        elif op == "random_insertion":
            solver = RandomInsertionSolver()
        elif op == "two_opt":
            solver = TwoOptLocalSearch(solver)
        else:
            raise ValueError(f"Unknown operator {op}")
    return solver


@dataclass
class Genome:
    ops: List[str] = field(default_factory=list)

    @staticmethod
    def random(rng: random.Random, length: int = None) -> "Genome":
        if length is None:
            length = rng.randint(1, 4)
        return Genome([rng.choice(PRIMITIVES) for _ in range(length)])

    def mutate(self, rate: float = 0.3) -> "Genome":
        new_ops = self.ops[:]
        # Operator-level mutations: replace, insert, delete, swap.
        for i in range(len(new_ops)):
            if random.random() < rate:
                new_ops[i] = random.choice(PRIMITIVES)
        if random.random() < rate:
            pos = random.randrange(len(new_ops) + 1)
            new_ops.insert(pos, random.choice(PRIMITIVES))
        if new_ops and random.random() < rate:
            new_ops.pop(random.randrange(len(new_ops)))
        if len(new_ops) > 1 and random.random() < rate:
            i, j = random.sample(range(len(new_ops)), 2)
            new_ops[i], new_ops[j] = new_ops[j], new_ops[i]
        return Genome(new_ops)

    def crossover(self, other: "Genome") -> "Genome":
        split = random.randint(0, max(len(self.ops), len(other.ops)))
        child_ops = self.ops[:split] + other.ops[split:]
        return Genome(child_ops)

    def build_solver(self) -> Solver:
        return build_solver(self.ops)
