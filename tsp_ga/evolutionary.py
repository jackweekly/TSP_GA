import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from .evaluation import Fitness, aggregate_fitness, evaluate_solver
from .solvers.genome import Genome
from .gnn import SurrogateManager


@dataclass
class EvolutionConfig:
    population_size: int = 40
    elite_fraction: float = 0.1
    mutation_rate: float = 0.7
    crossover_rate: float = 0.8
    max_runtime: float = 2.0
    runtime_weight: float = 0.1
    evaluation_samples: int = 2
    novelty_weight: float = 0.1
    random_seed: int = 123


class EvolutionarySearch:
    def __init__(
        self,
        config: EvolutionConfig,
        graphs: Sequence,
        optima: Sequence,
        dist_mats: Sequence = None,
        rng: random.Random = None,
        device=None,
    ):
        self.cfg = config
        self.graphs = graphs
        self.optima = optima
        self.dist_mats = dist_mats or [None] * len(graphs)
        self.device = device
        self.rng = rng or random.Random(config.random_seed)
        random.seed(config.random_seed)
        self.population: List[Genome] = [
            Genome.random(self.rng) for _ in range(config.population_size)
        ]
        self.surrogate = SurrogateManager(device) if device and str(device).startswith("cuda") else None

    def evaluate_genome(self, genome: Genome) -> float:
        solver = genome.build_solver()
        fitnesses: List[Fitness] = []
        sample_idxs = self.rng.sample(
            range(len(self.graphs)), k=min(self.cfg.evaluation_samples, len(self.graphs))
        )
        for idx in sample_idxs:
            graph = self.graphs[idx]
            opt = self.optima[idx]
            dist = self.dist_mats[idx] if self.dist_mats else None
            fitness = evaluate_solver(
                solver,
                graph,
                opt,
                max_runtime=self.cfg.max_runtime,
                runtime_weight=self.cfg.runtime_weight,
                dist_mat=dist,
            )
            fitnesses.append(fitness)
            if self.surrogate and dist is not None:
                self.surrogate.observe(dist, genome.signature, fitness.score)
        agg = aggregate_fitness(fitnesses)
        if self.surrogate:
            self.surrogate.train_step()
        return agg["score"]

    def step(self) -> None:
        scored = [(self.evaluate_genome(g), g) for g in self.population]
        # Apply novelty pressure based on signature frequency.
        sig_counts = {}
        for _, g in scored:
            sig_counts[g.signature] = sig_counts.get(g.signature, 0) + 1
        adjusted = []
        for score, g in scored:
            rarity = 1.0 / sig_counts[g.signature]
            adj = score * (1 - self.cfg.novelty_weight * rarity)
            adjusted.append((adj, g))
        adjusted.sort(key=lambda x: x[0])
        elite_count = max(1, int(self.cfg.elite_fraction * len(adjusted)))
        elites = [g for _, g in adjusted[:elite_count]]
        new_pop: List[Genome] = []
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
