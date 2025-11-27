import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from .evaluation import Fitness, aggregate_fitness, evaluate_solver
from .solvers.genome import Genome


@dataclass
class EvolutionConfig:
    population_size: int = 20
    elite_fraction: float = 0.2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    max_runtime: float = 5.0
    runtime_weight: float = 0.1
    evaluation_samples: int = 3
    random_seed: int = 123


class EvolutionarySearch:
    def __init__(
        self,
        config: EvolutionConfig,
        graphs: Sequence,
        optima: Sequence,
        rng: random.Random = None,
    ):
        self.cfg = config
        self.graphs = graphs
        self.optima = optima
        self.rng = rng or random.Random(config.random_seed)
        random.seed(config.random_seed)
        self.population: List[Genome] = [
            Genome.random(self.rng) if hasattr(Genome, "random") else Genome([])
            for _ in range(config.population_size)
        ]
        # Fallback: if Genome.random not defined, start with simple primitives.
        if not hasattr(Genome, "random"):
            for g in self.population:
                g.ops = [self.rng.choice(["nearest_neighbor", "random_insertion"])]

    def evaluate_genome(self, genome: Genome) -> float:
        solver = genome.build_solver()
        fitnesses: List[Fitness] = []
        sample_idxs = self.rng.sample(
            range(len(self.graphs)), k=min(self.cfg.evaluation_samples, len(self.graphs))
        )
        for idx in sample_idxs:
            graph = self.graphs[idx]
            opt = self.optima[idx]
            fitness = evaluate_solver(
                solver,
                graph,
                opt,
                max_runtime=self.cfg.max_runtime,
                runtime_weight=self.cfg.runtime_weight,
            )
            fitnesses.append(fitness)
        agg = aggregate_fitness(fitnesses)
        return agg["score"]

    def step(self) -> None:
        scored = [(self.evaluate_genome(g), g) for g in self.population]
        scored.sort(key=lambda x: x[0])
        elite_count = max(1, int(self.cfg.elite_fraction * len(scored)))
        elites = [g for _, g in scored[:elite_count]]
        new_pop: List[Genome] = []
        # Keep elites
        new_pop.extend(elites)
        while len(new_pop) < self.cfg.population_size:
            if self.rng.random() < self.cfg.crossover_rate and len(elites) >= 2:
                a, b = self.rng.sample(elites, 2)
                child = a.crossover(b)
            else:
                child = self.rng.choice(elites)
            child = child.mutate(self.cfg.mutation_rate)
            new_pop.append(child)
        self.population = new_pop

    def best(self, graphs=None) -> Genome:
        # Evaluate full set if provided; otherwise reuse current scoring.
        if graphs:
            best_genome = None
            best_score = float("inf")
            for g in self.population:
                score = self.evaluate_genome(g)
                if score < best_score:
                    best_score = score
                    best_genome = g
            return best_genome
        return self.population[0]
