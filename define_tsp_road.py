### ✅ define_tsp_road.py
import torch
from torch.utils.data import Dataset
import pickle
from problems.tsp.problem_tsp import TSP

class TSPRoad(TSP):
    def __init__(self):
        super().__init__()

    def get_costs(self, input, pi):
        if torch.is_tensor(input):
            raise ValueError("도로 거리 기반에서는 dict 형태의 입력만 지원합니다.")

        dist = input["dist"]
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)
        pi = pi.long()

        B, L = pi.size()
        cost = torch.zeros(B, device=pi.device)
        idx = torch.arange(B, device=pi.device)

        for step in range(L - 1):
            a = pi[:, step]
            b = pi[:, step + 1]
            a = a.clamp(0, dist.size(1) - 1)
            b = b.clamp(0, dist.size(2) - 1)
            cost += dist[idx, a, b]

        return cost, None

class RoadDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.locs = [torch.tensor(d["loc"], dtype=torch.float) for d in data]
        self.dists = [torch.tensor(d["dist"], dtype=torch.float) for d in data]

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        return {
            "loc": self.locs[idx],
            "dist": self.dists[idx]
        }
