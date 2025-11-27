import random
from pathlib import Path

from tsp_ga.data import load_data, split_by_hash
from tsp_ga.island import IslandConfig, IslandModel


def main():
    data_root = Path("data/tsplib")
    if not data_root.exists():
        raise FileNotFoundError("Place TSPLIB files in data/tsplib")

    instances = load_data(data_root)
    splits = split_by_hash(instances)
    train = splits["train"] or instances  # fallback if hash buckets empty
    graphs = [inst.graph for inst in train]
    optima = [inst.optimum for inst in train]

    cfg = IslandConfig(
        population_size=12,
        islands=2,
        migration_interval=3,
        migrants=2,
        evaluation_samples=3,
        max_runtime=2.0,
    )
    model = IslandModel(cfg, graphs, optima)
    generations = 5
    for g in range(generations):
        model.step()
        best, score = model.best()
        print(f"gen {g+1}: best ops={best.ops} score={score:.2f}")


if __name__ == "__main__":
    main()
