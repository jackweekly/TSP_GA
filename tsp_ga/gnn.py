import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.in_mlp = nn.Linear(3, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.readout = nn.Linear(hidden, hidden)

    def forward(self, adj: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # adj: [B,N,N], feats: [B,N,3]
        h = F.relu(self.in_mlp(feats))
        for layer in self.layers:
            agg = torch.matmul(adj, h) / (adj.sum(dim=-1, keepdim=True) + 1e-6)
            h = F.relu(layer(agg))
        g = h.mean(dim=1)
        return self.readout(g)


class SurrogateModel(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 4, ops_dim: int = 32):
        super().__init__()
        self.encoder = GraphEncoder(hidden=hidden, depth=depth)
        self.op_proj = nn.Linear(ops_dim, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, adj: torch.Tensor, feats: torch.Tensor, op_vec: torch.Tensor) -> torch.Tensor:
        g_emb = self.encoder(adj, feats)
        o = F.relu(self.op_proj(op_vec))
        x = F.relu(g_emb + o)
        return self.out(x).squeeze(-1)


def build_op_vec(signature: str, device) -> torch.Tensor:
    ops = signature.split("|")
    tokens = []
    for part in ops:
        tokens.extend(part.split("-") if part else [])
    vocab = [
        "nearest_neighbor",
        "random_insertion",
        "greedy_cycle",
        "christofides",
        "two_opt",
        "three_opt",
        "double_bridge",
        "ruin_recreate",
    ]
    vec = torch.zeros(len(vocab), device=device)
    for t in tokens:
        if t in vocab:
            vec[vocab.index(t)] += 1.0
    if vec.norm() > 0:
        vec = vec / vec.norm()
    return vec


class SurrogateManager:
    def __init__(self, device):
        self.device = device
        self.model = SurrogateModel().to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.buffer = []

    def featurize_graph(self, dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # dist: [N,N]
        adj = dist / (dist.max() + 1e-6)
        deg = adj.sum(dim=-1, keepdim=True)
        mean_w = dist.mean(dim=-1, keepdim=True)
        nfeat = torch.tensor([[dist.shape[0]]], device=dist.device).repeat(dist.shape[0], 1)
        feats = torch.cat([deg, mean_w, nfeat], dim=-1)
        return adj, feats

    def predict(self, dist: torch.Tensor, signature: str) -> float:
        self.model.eval()
        with torch.no_grad():
            adj, feats = self.featurize_graph(dist)
            op_vec = build_op_vec(signature, dist.device)
            op_vec = F.pad(op_vec, (0, 32 - op_vec.numel())) if op_vec.numel() < 32 else op_vec[:32]
            op_vec = op_vec.unsqueeze(0)
            return self.model(adj.unsqueeze(0), feats.unsqueeze(0), op_vec).item()

    def observe(self, dist: torch.Tensor, signature: str, score: float):
        adj, feats = self.featurize_graph(dist)
        op_vec = build_op_vec(signature, dist.device)
        op_vec = F.pad(op_vec, (0, 32 - op_vec.numel())) if op_vec.numel() < 32 else op_vec[:32]
        self.buffer.append((adj, feats, op_vec, torch.tensor([score], device=dist.device)))
        if len(self.buffer) > 256:
            self.buffer = self.buffer[-256:]

    def train_step(self, batch_size: int = 16):
        if len(self.buffer) < batch_size:
            return
        self.model.train()
        batch = random.sample(self.buffer, batch_size)
        adjs, feats, ops, ys = zip(*batch)
        max_n = max(a.shape[-1] for a in adjs)
        def pad(t, target):
            if t.shape[-1] == target:
                return t
            pad_amt = target - t.shape[-1]
            return F.pad(t, (0, pad_amt, 0, pad_amt))
        adjs = torch.stack([pad(a, max_n) for a in adjs])
        feats = torch.stack([F.pad(f, (0, 0, 0, max_n - f.shape[0])) for f in feats])
        ops = torch.stack(ops)
        ys = torch.cat(ys)
        pred = self.model(adjs, feats, ops)
        loss = F.mse_loss(pred, ys)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict()}, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
