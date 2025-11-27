import random
from typing import List, Sequence, Tuple

import networkx as nx

from .base import Solver, Tour, tour_length


def nearest_neighbor_tour(graph: nx.Graph, start: int) -> Tour:
    tour = [start]
    unvisited = set(graph.nodes())
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda node: graph[current][node]["weight"])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour


def two_opt(graph: nx.Graph, tour: Tour, max_iter: int = 100) -> Tour:
    best = tour
    best_len = tour_length(graph, best)
    n = len(tour)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if j - i == 1:
                    continue
                new_tour = best[:]
                new_tour[i:j] = reversed(new_tour[i:j])
                new_len = tour_length(graph, new_tour)
                if new_len + 1e-9 < best_len:
                    best = new_tour
                    best_len = new_len
                    improved = True
        if not improved:
            break
    return best


class NearestNeighborSolver(Solver):
    name = "nearest_neighbor"

    def solve(self, graph: nx.Graph) -> Tour:
        start = random.choice(list(graph.nodes()))
        return nearest_neighbor_tour(graph, start)


class RandomInsertionSolver(Solver):
    name = "random_insertion"

    def solve(self, graph: nx.Graph) -> Tour:
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        tour = nodes[:3]
        for node in nodes[3:]:
            best_pos = 0
            best_increase = float("inf")
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                inc = (
                    graph[a][node]["weight"]
                    + graph[node][b]["weight"]
                    - graph[a][b]["weight"]
                )
                if inc < best_increase:
                    best_increase = inc
                    best_pos = i + 1
            tour.insert(best_pos, node)
        return tour


class TwoOptLocalSearch(Solver):
    name = "two_opt_local_search"

    def __init__(self, base: Solver, max_iter: int = 100):
        self.base = base
        self.max_iter = max_iter

    def solve(self, graph: nx.Graph) -> Tour:
        base_tour = self.base.solve(graph)
        return two_opt(graph, base_tour, max_iter=self.max_iter)
