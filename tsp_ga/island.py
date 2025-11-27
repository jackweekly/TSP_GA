import copy
import random
from dataclasses import dataclass
from typing import List

from .evolutionary import EvolutionConfig, EvolutionarySearch
from .solvers.genome import Genome


@dataclass
class IslandConfig(EvolutionConfig):
    islands: int = 2
    migration_interval: int = 5
    migrants: int = 2


class IslandModel:
    def __init__(self, cfg: IslandConfig, graphs, optima):
        self.cfg = cfg
        self.islands: List[EvolutionarySearch] = []
        for i in range(cfg.islands):
            rng = random.Random(cfg.random_seed + i)
            island_cfg = copy.deepcopy(cfg)
            self.islands.append(EvolutionarySearch(island_cfg, graphs, optima, rng=rng))
        self.generation = 0

    def migrate(self) -> None:
        migrants: List[List[Genome]] = []
        for island in self.islands:
            scored = [(island.evaluate_genome(g), g) for g in island.population]
            scored.sort(key=lambda x: x[0])
            migrants.append([g for _, g in scored[: self.cfg.migrants]])
        for i, island in enumerate(self.islands):
            incoming = migrants[(i - 1) % len(self.islands)]
            island.population[-len(incoming) :] = [copy.deepcopy(g) for g in incoming]

    def step(self) -> None:
        for island in self.islands:
            island.step()
        self.generation += 1
        if self.generation % self.cfg.migration_interval == 0:
            self.migrate()

    def best(self):
        best_genome = None
        best_score = float("inf")
        for island in self.islands:
            genome = island.best()
            score = island.evaluate_genome(genome)
            if score < best_score:
                best_genome = genome
                best_score = score
        return best_genome, best_score
