import os
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from nets.attention_model import AttentionModel
from define_tsp_road import TSPRoad, RoadDataset

# -------------------------
# CLI 인자 파싱
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='이전에 저장된 체크포인트에서 학습 재개')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='체크포인트 파일 경로')
args = parser.parse_args()

# -------------------------
# 2-OPT 로컬 서치 구현
# -------------------------
def two_opt(route, dist_matrix):
    n = len(route)
    best_route = route
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            if j - i == 1: continue
            new_route = route[:i] + route[i:j][::-1] + route[j:]
            cost_old = sum(dist_matrix[route[k], route[k+1]] for k in range(n-1))
            cost_new = sum(dist_matrix[new_route[k], new_route[k+1]] for k in range(n-1))
            if cost_new < cost_old:
                return new_route, True
    return best_route, False

# -------------------------
# Actor-Critic 모델 정의
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=128, n_encode_layers=3, n_heads=8, tanh_clipping=10.0):
        super().__init__()
        self.actor = AttentionModel(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers, n_heads=n_heads,
            tanh_clipping=tanh_clipping, normalization="batch",
            problem=TSPRoad()
        )
        self.value_head = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
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
CHECKPOINT    = args.checkpoint
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

# 모델, 환경, 옵티마이저 초기화
env = TSPRoad()
model = ActorCritic().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR_INIT)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

# 체크포인트에서 재개
start_epoch = 1
best_val = float('inf')
if args.resume and os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    best_val = ckpt.get('best_val', best_val)
    start_epoch = ckpt.get('epoch', 1) + 1
    print(f"🔄 Resume from epoch {start_epoch}")

# 사전학습 가중치 로드
if start_epoch == 1:
    pretrained = torch.load(PRETRAIN, map_location=DEVICE)
    model.actor.load_state_dict(pretrained['model'], strict=False)
    model.actor.set_decode_type('sampling')

# 데이터 로드 및 전처리
dataset = RoadDataset(DATA_PKL)
for i in range(len(dataset)):
    dist_mat = dataset[i]['dist']
    dist_mat[torch.isinf(dist_mat)] = FALLBACK_DIST
n = len(dataset)
indices = list(range(n)); random.shuffle(indices)
split = int(n * VAL_SPLIT)
train_idx, val_idx = indices[split:], indices[:split]
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE)

# -------------------------
# 학습 루프
# -------------------------
for ep in range(start_epoch, EPOCHS+1):
    model.train()
    for batch in tqdm(train_loader, desc=f"Train Ep {ep}/{EPOCHS}", ncols=80):
        loc, dist = batch['loc'].to(DEVICE), batch['dist'].to(DEVICE)
        mask = torch.isinf(dist); dist[mask] = FALLBACK_DIST
        rewards, logps, values = [], [], []
        for _ in range(NUM_SAMPLES):
            logp, tour, value = model(loc)
            cost, _ = env.get_costs({'dist': dist}, tour)
            rewards.append(-cost / 1e5)
            logps.append(logp); values.append(value)
        rewards = torch.stack(rewards); logps = torch.stack(logps); values = torch.stack(values)
        advantages = rewards - values; logp_old = logps.detach()
        total_p, total_v, total_e = 0, 0, 0
        for _ in range(PPO_EPOCHS):
            ratio = torch.exp(logps - logp_old)
            s1, s2 = ratio * advantages, torch.clamp(ratio, 1-CLIP_EPS,1+CLIP_EPS)*advantages
            total_p += -torch.min(s1,s2).mean()
            total_v += (rewards-values).pow(2).mean()
            total_e += -(logps.exp()*logps).mean()
        loss = total_p/PPO_EPOCHS + VALUE_COEF*(total_v/PPO_EPOCHS) - ENT_COEF*(total_e/PPO_EPOCHS)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    scheduler.step()
    # 검증
    model.eval(); model.actor.set_decode_type('greedy'); val_cost=0
    with torch.no_grad():
        for batch in val_loader:
            loc, dist = batch['loc'].to(DEVICE), batch['dist'].to(DEVICE)
            _, _, tour = model.actor(loc, return_pi=True)
            tour = tour[0].tolist(); imp,_=two_opt(tour, dist[0].cpu().numpy())
            cost,_ = env.get_costs({'dist': dist}, torch.tensor([imp],device=DEVICE))
            val_cost += cost.item()
    avg_val = val_cost/len(val_loader)
    print(f"[Ep {ep}] Val Cost: {avg_val:.2f}")
    # 체크포인트 저장
    torch.save({
        'epoch': ep, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'best_val': best_val
    }, CHECKPOINT)
    if avg_val < best_val:
        best_val = avg_val
        torch.save({'actor': model.actor.state_dict(),'critic':model.value_head.state_dict()}, BEST_MODEL)
        print(f"🌟 New Best: {best_val:.2f}")
# 최종 저장
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
torch.save({'actor': model.actor.state_dict(),'critic':model.value_head.state_dict()}, OUT_MODEL)
print(f"Training complete. Best Val Cost: {best_val:.2f}")


