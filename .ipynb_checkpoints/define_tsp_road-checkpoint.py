# define_tsp_road.py
import torch
from torch.utils.data import Dataset
import pickle
from problems.tsp.problem_tsp import TSP

class TSPRoad(TSP):
    def __init__(self):
        super().__init__()

    def get_costs(self, input, pi):
        # (1) 모델 내부에서 호출될 때는 input이 그냥 Tensor
        if torch.is_tensor(input):
            return super().get_costs(input, pi)  # 원래 TSP 거리 계산
    
        # (2) 파인튜닝 시에는 dict로 {"dist": (B, N, N)} 형태
        dist = input["dist"]  # (B, N, N)
    
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)
        pi = pi.long()
    
        B, L = pi.size()
        cost = torch.zeros(B, device=pi.device)
        idx  = torch.arange(B, device=pi.device)
    
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
        self.locs  = [torch.tensor(d["loc"], dtype=torch.float)  for d in data]
        self.dists = [torch.tensor(d["dist"], dtype=torch.float) for d in data]

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        return {
            "loc": self.locs[idx],   # (N, 2)
            "dist": self.dists[idx]  # (N, N)
        }


