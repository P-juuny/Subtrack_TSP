import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from nets.attention_model import AttentionModel
from define_tsp_road import TSPRoad, RoadDataset

# -------------------------
# 2-OPT 로컬 서치 구현
# -------------------------
def two_opt(route, dist_matrix):
    n = len(route)
    best_route = route
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            if j - i == 1:
                continue
            new_route = route[:i] + route[i:j][::-1] + route[j:]
            cost_old = sum(dist_matrix[route[k], route[k+1]] for k in range(n - 1))
            cost_new = sum(dist_matrix[new_route[k], new_route[k+1]] for k in range(n - 1))
            if cost_new < cost_old:
                return new_route, True
    return best_route, False

# -------------------------
# Actor-Critic 모델 정의
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 hidden_dim=128,
                 n_encode_layers=3,
                 n_heads=8,
                 tanh_clipping=10.0):
        super().__init__()
        self.actor = AttentionModel(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            normalization="batch",
            problem=TSPRoad()
        )
        self.value_head = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, loc):
        _, logp, tour = self.actor(loc, return_pi=True)
        pooled = loc.mean(dim=1)
        value = self.value_head(pooled).squeeze(-1)
        return logp, tour, value

# -------------------------
# 하이퍼파라미터
# -------------------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN      = "pretrained/tsp_100/epoch-99.pt"
DATA_PKL      = "data/road_TSP_100_fixed.pkl"
OUT_MODEL     = "pretrained/final_ppo_road.pt"
BEST_MODEL    = "pretrained/best_ppo_road.pt"
BATCH_SIZE    = 64
EPOCHS        = 300
LR_INIT       = 1e-5
ETA_MIN       = 1e-7
PPO_EPOCHS    = 4
CLIP_EPS      = 0.2
ENT_COEF      = 0.005
VALUE_COEF    = 0.5
NUM_SAMPLES   = 8
FALLBACK_DIST = 60000.0
VAL_SPLIT     = 0.1
SEED          = 42

# 재현성
random.seed(SEED)
torch.manual_seed(SEED)

env = TSPRoad()
model = ActorCritic().to(DEVICE)
ckpt = torch.load(PRETRAIN, map_location=DEVICE)
model.actor.load_state_dict(ckpt['model'], strict=False)
model.actor.set_decode_type('sampling')

optimizer = Adam(model.parameters(), lr=LR_INIT)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

# 데이터 로드 및 전처리
dataset = RoadDataset(DATA_PKL)
for i in range(len(dataset)):
    dist_mat = dataset[i]['dist']
    dist_mat[torch.isinf(dist_mat)] = FALLBACK_DIST

n = len(dataset)
indices = list(range(n))
random.shuffle(indices)
split_idx = int(n * VAL_SPLIT)
train_idx, val_idx = indices[split_idx:], indices[:split_idx]
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE)

best_val = float('inf')

# -------------------------
# 학습 루프
# -------------------------
for ep in range(1, EPOCHS + 1):
    model.train()
    for batch in tqdm(train_loader, desc=f"Train Ep {ep}/{EPOCHS}", ncols=80):
        loc  = batch['loc'].to(DEVICE)
        dist = batch['dist'].to(DEVICE)

        rewards, logps, values = [], [], []
        for _ in range(NUM_SAMPLES):
            logp, tour, value = model(loc)
            cost, _ = env.get_costs({'dist': dist}, tour)
            reward = -cost / 1e5
            rewards.append(reward)
            logps.append(logp)
            values.append(value)

        rewards = torch.stack(rewards)  # (S, B)
        logps   = torch.stack(logps)
        values  = torch.stack(values)

        returns    = rewards
        advantages = returns - values
        logp_old   = logps.detach()

        # PPO 업데이트: 여러 스텝 loss 누적 후 한 번 backward
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        for _ in range(PPO_EPOCHS):
            ratio = torch.exp(logps - logp_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            total_policy_loss += -torch.min(surr1, surr2).mean()
            total_value_loss  += (returns - values).pow(2).mean()
            total_entropy     += -(logps.exp() * logps).mean()

        policy_loss = total_policy_loss / PPO_EPOCHS
        value_loss  = total_value_loss / PPO_EPOCHS
        entropy     = total_entropy / PPO_EPOCHS
        loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    # 검증: greedy 샘플링 + 2-opt
    model.eval()
    model.actor.set_decode_type('greedy')
    val_cost = 0.0
    with torch.no_grad():
        for batch in val_loader:
            loc  = batch['loc'].to(DEVICE)
            dist = batch['dist'].to(DEVICE)
            _, tour = model.actor(loc, return_pi=True)
            tour = tour[0].tolist()
            improved, _ = two_opt(tour, dist[0].cpu().numpy())
            cost, _ = env.get_costs({'dist': dist}, torch.tensor([improved], device=DEVICE))
            val_cost += cost.item()
    avg_val = val_cost / len(val_loader)
    print(f"[Ep {ep}] Val Cost: {avg_val:.2f}")

    if avg_val < best_val:
        best_val = avg_val
        torch.save({'actor': model.actor.state_dict(), 'critic': model.value_head.state_dict()}, BEST_MODEL)
        print(f"🌟 New Best: {best_val:.2f}")

os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
torch.save({'actor': model.actor.state_dict(), 'critic': model.value_head.state_dict()}, OUT_MODEL)
print(f"Training complete. Best Val Cost: {best_val:.2f}")

