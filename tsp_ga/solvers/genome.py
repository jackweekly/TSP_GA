import random
from dataclasses import dataclass, field
from typing import List

from .base import Solver
from .heuristics import CompositionSolver


CONSTRUCT_PRIMS = ["nearest_neighbor", "random_insertion", "greedy_cycle", "christofides"]
IMPROVE_PRIMS = ["two_opt", "three_opt"]
DIV_PRIMS = ["double_bridge", "ruin_recreate"]


@dataclass
class Genome:
    construct: str
    improve_ops: List[str] = field(default_factory=list)
    diversify_ops: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.improve_ops, str):
            self.improve_ops = [self.improve_ops]
        if isinstance(self.diversify_ops, str):
            self.diversify_ops = [self.diversify_ops]

    @staticmethod
    def random(rng: random.Random, improve_len: int = None, diversify_len: int = None) -> "Genome":
        if improve_len is None:
            improve_len = rng.randint(1, 3)
        if diversify_len is None:
            diversify_len = rng.randint(0, 2)
        construct = rng.choice(CONSTRUCT_PRIMS)
        improve_ops = [rng.choice(IMPROVE_PRIMS) for _ in range(improve_len)]
        diversify_ops = [rng.choice(DIV_PRIMS) for _ in range(diversify_len)]
        return Genome(construct=construct, improve_ops=improve_ops, diversify_ops=diversify_ops)

    def mutate(self, rate: float = 0.5) -> "Genome":
        construct = self.construct
        improve_ops = self.improve_ops[:]
        diversify_ops = self.diversify_ops[:]
        if random.random() < rate:
            construct = random.choice(CONSTRUCT_PRIMS)
        for ops, pool in [(improve_ops, IMPROVE_PRIMS), (diversify_ops, DIV_PRIMS)]:
            for i in range(len(ops)):
                if random.random() < rate:
                    ops[i] = random.choice(pool)
            if random.random() < rate:
                pos = random.randrange(len(ops) + 1)
                ops.insert(pos, random.choice(pool))
            if ops and random.random() < rate:
                ops.pop(random.randrange(len(ops)))
            if len(ops) > 1 and random.random() < rate:
                i, j = random.sample(range(len(ops)), 2)
                ops[i], ops[j] = ops[j], ops[i]
        return Genome(construct=construct, improve_ops=improve_ops, diversify_ops=diversify_ops)

    def crossover(self, other: "Genome") -> "Genome":
        child_construct = random.choice([self.construct, other.construct])
        mid_improve = random.randint(0, max(len(self.improve_ops), len(other.improve_ops)))
        child_improve = self.improve_ops[:mid_improve] + other.improve_ops[mid_improve:]
        mid_div = random.randint(0, max(len(self.diversify_ops), len(other.diversify_ops)))
        child_div = self.diversify_ops[:mid_div] + other.diversify_ops[mid_div:]
        return Genome(construct=child_construct, improve_ops=child_improve, diversify_ops=child_div)

    def build_solver(self) -> Solver:
        return CompositionSolver(self.construct, self.improve_ops, self.diversify_ops)

    @property
    def signature(self) -> str:
        return f"{self.construct}|{'-'.join(self.improve_ops)}|{'-'.join(self.diversify_ops)}"
