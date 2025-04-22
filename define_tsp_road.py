import torch
from torch.utils.data import Dataset
import pickle
from problems.tsp.problem_tsp import TSP

class TSPRoad(TSP):
    def __init__(self):
        super().__init__()
        # 미터 단위를 조정하기 위한 스케일 팩터
        self.scale_factor = 0.001  # km 단위로 변환 (더 작은 숫자로 만듦)
        
    def get_costs(self, input, pi):
        """
        도로 기반 거리 계산 함수
        input: {"dist": Tensor(B, N, N)} 또는 Tensor(B, N, 2)
        pi: 경로 순서 (B, N)
        """
        # (1) 모델 내부에서 호출될 때는 input이 그냥 Tensor - 유클리드 거리 계산
        if torch.is_tensor(input):
            # 원래 TSP 거리 계산을 사용하되, 결과를 미터 단위와 동일한 스케일로 복원
            cost, mask = super().get_costs(input, pi)
            return cost / self.scale_factor, mask  # 스케일 복원 (km -> m)
        
        # (2) 파인튜닝 시에는 dist로 {"dist": (B, N, N)} 형태
        dist = input["dist"]  # (B, N, N)
        
        # 학습과 추론을 위해 거리값을 스케일링 (미터 -> km)
        # 이렇게 하면 모델이 더 작은 숫자로 학습하게 됨
        scaled_dist = dist * self.scale_factor
        
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)
        pi = pi.long()
        
        B, L = pi.size()
        cost = torch.zeros(B, device=pi.device)
        idx = torch.arange(B, device=pi.device)
        
        for step in range(L - 1):
            a = pi[:, step]
            b = pi[:, step + 1]
            a = a.clamp(0, scaled_dist.size(1) - 1)
            b = b.clamp(0, scaled_dist.size(2) - 1)
            cost += scaled_dist[idx, a, b]
        
        return cost, None  # 학습 중에는 스케일된 값으로 반환

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
            "loc": self.locs[idx],   # (N, 2)
            "dist": self.dists[idx]  # (N, N)
        }