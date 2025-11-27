import random
from typing import List, Sequence, Tuple

import networkx as nx
import torch

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


def greedy_cycle(graph: nx.Graph) -> Tour:
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    tour = nodes[:2]
    remaining = nodes[2:]
    while remaining:
        node = remaining.pop()
        best_pos = 0
        best_inc = float("inf")
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            inc = graph[a][node]["weight"] + graph[node][b]["weight"] - graph[a][b]["weight"]
            if inc < best_inc:
                best_inc = inc
                best_pos = i + 1
        tour.insert(best_pos, node)
    return tour


def christofides_like(graph: nx.Graph) -> Tour:
    # Simple MST + matching approximation (not full Christofides, but gives diversity).
    mst = nx.minimum_spanning_tree(graph)
    odd_nodes = [v for v in mst.degree() if mst.degree()[v] % 2 == 1]
    subgraph = graph.subgraph(odd_nodes)
    matching = nx.algorithms.matching.max_weight_matching(subgraph, maxcardinality=True)
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(matching)
    euler_circuit = list(nx.eulerian_circuit(multigraph, source=odd_nodes[0] if odd_nodes else 1))
    path = []
    visited = set()
    for u, v in euler_circuit:
        if u not in visited:
            path.append(u)
            visited.add(u)
        if v not in visited:
            path.append(v)
            visited.add(v)
    return path


def two_opt(graph: nx.Graph, tour: Tour, max_iter: int = 200) -> Tour:
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


def torch_two_opt(dist: torch.Tensor, tour: torch.Tensor, max_iter: int = 200) -> torch.Tensor:
    # tour: 1D long tensor of node indices on device matching dist
    best = tour.clone()
    best_len = (dist[best, best.roll(-1)]).sum()
    n = best.shape[0]
    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if j - i == 1:
                    continue
                cand = best.clone()
                cand[i:j] = torch.flip(cand[i:j], dims=[0])
                cand_len = (dist[cand, cand.roll(-1)]).sum()
                if cand_len + 1e-9 < best_len:
                    best = cand
                    best_len = cand_len
                    improved = True
        if not improved:
            break
    return best


def torch_three_opt(dist: torch.Tensor, tour: torch.Tensor, max_iter: int = 50) -> torch.Tensor:
    best = tour.clone()
    best_len = (dist[best, best.roll(-1)]).sum()
    n = best.shape[0]
    for _ in range(max_iter):
        i, j, k = torch.sort(torch.randint(0, n, (3,), device=dist.device)).values
        variants = [
            torch.cat([best[:i], torch.flip(best[i:j], [0]), torch.flip(best[j:k], [0]), best[k:]]),
            torch.cat([best[:i], best[j:k], best[i:j], best[k:]]),
            torch.cat([best[:i], torch.flip(best[j:k], [0]), best[i:j], best[k:]]),
        ]
        for cand in variants:
            cand_len = (dist[cand, cand.roll(-1)]).sum()
            if cand_len + 1e-9 < best_len:
                best = cand
                best_len = cand_len
    return best


def three_opt(graph: nx.Graph, tour: Tour, max_iter: int = 50) -> Tour:
    # Simplified 3-opt using random segment swaps.
    best = tour
    best_len = tour_length(graph, best)
    n = len(tour)
    for _ in range(max_iter):
        i, j, k = sorted(random.sample(range(n), 3))
        variants = [
            best[:i] + best[i:j][::-1] + best[j:k][::-1] + best[k:],
            best[:i] + best[j:k] + best[i:j] + best[k:],
            best[:i] + best[j:k][::-1] + best[i:j] + best[k:],
        ]
        for cand in variants:
            cand_len = tour_length(graph, cand)
            if cand_len + 1e-9 < best_len:
                best = cand
                best_len = cand_len
    return best


def double_bridge(tour: Tour) -> Tour:
    n = len(tour)
    if n < 8:
        return tour
    a, b, c = sorted(random.sample(range(1, n - 1), 3))
    return tour[:a] + tour[c:] + tour[b:c] + tour[a:b]


def torch_double_bridge(tour: torch.Tensor) -> torch.Tensor:
    n = tour.shape[0]
    if n < 8:
        return tour
    a, b, c = torch.sort(torch.randint(1, n - 1, (3,), device=tour.device)).values
    return torch.cat([tour[:a], tour[c:], tour[b:c], tour[a:b]])


def ruin_recreate(graph: nx.Graph, tour: Tour, ruin_frac: float = 0.1) -> Tour:
    n = len(tour)
    ruin_count = max(2, int(n * ruin_frac))
    remove_idx = sorted(random.sample(range(n), ruin_count), reverse=True)
    removed = []
    for idx in remove_idx:
        removed.append(tour.pop(idx))
    for node in removed:
        best_pos = 0
        best_inc = float("inf")
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            inc = graph[a][node]["weight"] + graph[node][b]["weight"] - graph[a][b]["weight"]
            if inc < best_inc:
                best_inc = inc
                best_pos = i + 1
        tour.insert(best_pos, node)
    return tour


def torch_ruin_recreate(dist: torch.Tensor, tour: torch.Tensor, ruin_frac: float = 0.1) -> torch.Tensor:
    n = tour.shape[0]
    ruin_count = max(2, int(n * ruin_frac))
    idx = torch.randperm(n, device=dist.device)[:ruin_count]
    keep_mask = torch.ones(n, dtype=torch.bool, device=dist.device)
    keep_mask[idx] = False
    kept = tour[keep_mask].tolist()
    removed = tour[idx].tolist()
    for node in removed:
        best_pos = 0
        best_inc = float("inf")
        for i in range(len(kept)):
            a = kept[i]
            b = kept[(i + 1) % len(kept)]
            inc = dist[a, node] + dist[node, b] - dist[a, b]
            if inc < best_inc:
                best_inc = inc
                best_pos = i + 1
        kept.insert(best_pos, node)
    return torch.tensor(kept, device=dist.device, dtype=torch.long)


class ConstructiveSolver(Solver):
    name = "constructive"

    def __init__(self, strategy: str):
        self.strategy = strategy

    def solve(self, graph: nx.Graph) -> Tour:
        if self.strategy == "nearest_neighbor":
            start = random.choice(list(graph.nodes()))
            return nearest_neighbor_tour(graph, start)
        if self.strategy == "random_insertion":
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
        if self.strategy == "greedy_cycle":
            return greedy_cycle(graph)
        if self.strategy == "christofides":
            return christofides_like(graph)
        return nearest_neighbor_tour(graph, random.choice(list(graph.nodes())))


def apply_improvements(graph: nx.Graph, tour: Tour, ops: List[str]) -> Tour:
    cur = tour
    for op in ops:
        if op == "two_opt":
            cur = two_opt(graph, cur)
        elif op == "three_opt":
            cur = three_opt(graph, cur)
        elif op == "double_bridge":
            cur = double_bridge(cur)
        elif op == "ruin_recreate":
            cur = ruin_recreate(graph, cur)
    return cur


class CompositionSolver(Solver):
    """
    Solver built from phases: construct -> improve -> diversify.
    Can operate on CPU (networkx) or GPU (torch) if dist_mat provided.
    """

    name = "composition"

    def __init__(self, construct: str, improve_ops: List[str], diversify_ops: List[str]):
        self.construct = construct
        self.improve_ops = improve_ops
        self.diversify_ops = diversify_ops
        self.dist_mat = None  # set externally when GPU matrix available
        self.node_map = None   # optional mapping from node label to index

    def solve(self, graph: nx.Graph) -> Tour:
        if self.dist_mat is not None and torch.is_tensor(self.dist_mat):
            # GPU path
            base_nodes = ConstructiveSolver(self.construct).solve(graph)
            if self.node_map:
                base_idx = [self.node_map.get(n, 0) for n in base_nodes]
            else:
                base_idx = base_nodes
            base = torch.tensor(base_idx, device=self.dist_mat.device, dtype=torch.long)
            cur = base
            for op in self.improve_ops:
                if op == "two_opt":
                    cur = torch_two_opt(self.dist_mat, cur)
                elif op == "three_opt":
                    cur = torch_three_opt(self.dist_mat, cur)
            for op in self.diversify_ops:
                if op == "double_bridge":
                    cur = torch_double_bridge(cur)
                elif op == "ruin_recreate":
                    cur = torch_ruin_recreate(self.dist_mat, cur)
                for imp in self.improve_ops:
                    if imp == "two_opt":
                        cur = torch_two_opt(self.dist_mat, cur)
                    elif imp == "three_opt":
                        cur = torch_three_opt(self.dist_mat, cur)
            return cur.tolist()
        # CPU path
        base = ConstructiveSolver(self.construct).solve(graph)
        improved = apply_improvements(graph, base, self.improve_ops)
        for op in self.diversify_ops:
            if op == "double_bridge":
                improved = double_bridge(improved)
            elif op == "ruin_recreate":
                improved = ruin_recreate(graph, improved)
            improved = apply_improvements(graph, improved, self.improve_ops)
        return improved
