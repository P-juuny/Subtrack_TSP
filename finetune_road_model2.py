import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# ----------------------------------------------------------------
# nets.attention_model 과 define_tsp_road 는 기존과 동일하게 가정
# ----------------------------------------------------------------
try:
    from nets.attention_model import AttentionModel
    from define_tsp_road import TSPRoad, RoadDataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    exit()

# -------------------------
# CLI 인자 파싱
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='이전에 저장된 체크포인트에서 학습 재개')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='체크포인트 파일 경로')
args = parser.parse_args()

# -------------------------
# 2-OPT 로컬 서치 구현 (개선된 버전)
# -------------------------
def calculate_total_distance(route, dist_matrix):
    if not route: return 0.0
    total = 0.0
    if isinstance(dist_matrix, torch.Tensor):
        dist_matrix = dist_matrix.cpu().numpy()
    n = len(route)
    for i in range(n-1):
        u,v = route[i], route[i+1]
        if 0 <= u < dist_matrix.shape[0] and 0 <= v < dist_matrix.shape[1]:
            total += dist_matrix[u,v]
        else:
            return float('inf')
    return total

def two_opt_improved(route, dist_matrix):
    if len(route) < 4:
        return route, calculate_total_distance(route, dist_matrix)
    if isinstance(dist_matrix, torch.Tensor):
        dist_matrix = dist_matrix.cpu().numpy()
    n = len(route)
    best = route[:]
    best_cost = calculate_total_distance(best, dist_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(0, n-2):
            for j in range(i+2, n):
                if j+1 >= n: continue
                a,b = best[i], best[i+1]
                c,d = best[j], best[j+1]
                if any(x<0 or x>=dist_matrix.shape[0] for x in (a,b,c,d)):
                    continue
                old_cost = dist_matrix[a,b] + dist_matrix[c,d]
                new_cost = dist_matrix[a,c] + dist_matrix[b,d]
                if new_cost + 1e-9 < old_cost:
                    best[i+1:j+1] = reversed(best[i+1:j+1])
                    best_cost += new_cost - old_cost
                    improved = True
        # end for
    final = calculate_total_distance(best, dist_matrix)
    return best, final

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
        """
        Actor: logp, tour
        Critic: loc.mean(dim=1) -> value
        """
        _, logp, tour = self.actor(loc, return_pi=True)
        pooled = loc.mean(dim=1)          # (batch, 2)
        value  = self.value_head(pooled).squeeze(-1)
        return logp, tour, value

# -------------------------
# 하이퍼파라미터
# -------------------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN      = "pretrained/tsp_100/epoch-99.pt"
DATA_PKL      = "data/road_TSP_100_fixed.pkl"
OUT_MODEL     = "pretrained/final_ppo_road_v3.pt"
BEST_MODEL    = "pretrained/best_ppo_road_v3.pt"
CHECKPOINT    = args.checkpoint
BATCH_SIZE    = 64
EPOCHS        = 300
LR_INIT       = 3e-5
ETA_MIN       = 1e-7
PPO_EPOCHS    = 3
CLIP_EPS      = 0.2
ENT_COEF      = 0.01
VALUE_COEF    = 0.5
NUM_SAMPLES   = 1
FALLBACK_DIST = 60000.0
VAL_SPLIT     = 0.1
SEED          = 42

# 재현성 설정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 모델, 환경, 옵티마이저 초기화
env       = TSPRoad()
model     = ActorCritic().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR_INIT)

# 데이터 로드
try:
    full_dataset = RoadDataset(DATA_PKL)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

n_total   = len(full_dataset)
indices   = list(range(n_total))
random.shuffle(indices)
split_idx = max(1, int(n_total * VAL_SPLIT))
train_idx = indices[split_idx:]
val_idx   = indices[:split_idx]

train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(Subset(full_dataset, val_idx),   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print(f"Dataset: total={n_total}, train={len(train_idx)}, val={len(val_idx)}")
print(f"Device: {DEVICE}")

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=ETA_MIN)

# 체크포인트 재개
start_epoch  = 1
best_val_cost = float('inf')
if args.resume and os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    best_val_cost = ckpt.get('best_val_cost', best_val_cost)
    start_epoch   = ckpt.get('epoch', 1) + 1
    print(f"🔄 Resume from epoch {start_epoch}")

# 사전학습 가중치 로드
elif start_epoch == 1 and os.path.exists(PRETRAIN):
    pretrained = torch.load(PRETRAIN, map_location=DEVICE)
    actor_sd   = pretrained.get('model', pretrained.get('actor'))
    if actor_sd:
        own = model.state_dict()
        # actor 파라미터만 로드
        pretrained_dict = {k:v for k,v in actor_sd.items() if k.startswith('actor.')}
        own.update(pretrained_dict)
        model.load_state_dict(own, strict=False)
        print("✅ Loaded pretrained actor weights")
    else:
        print("⚠️ Pretrained file has no actor/model key")

model.actor.set_decode_type('sampling')

# -------------------------
# 학습 루프 시작
# -------------------------
for ep in range(start_epoch, EPOCHS+1):
    model.train()
    model.actor.set_decode_type('sampling')

    epoch_p_loss = epoch_v_loss = epoch_e_loss = epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train Ep {ep}/{EPOCHS}", ncols=100)
    for batch in pbar:
        loc  = batch['loc'].to(DEVICE)
        dist = batch['dist'].to(DEVICE)
        dist = torch.where(torch.isinf(dist), torch.full_like(dist, FALLBACK_DIST), dist)

        # rollout
        logp, tour, value = model(loc)
        cost, _ = env.get_costs({'loc':loc, 'dist':dist}, tour)
        reward = -cost

        adv = reward - value.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp_old = logp.detach()

        total_p = total_v = total_e = 0.0
        for _ in range(PPO_EPOCHS):
            logp_new, _, value_new = model(loc)
            ratio = torch.exp(logp_new - logp_old)
            s1    = ratio * adv
            s2    = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv

            p_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(value_new, reward)
            entropy= -logp_new.mean()

            loss = p_loss + VALUE_COEF*v_loss - ENT_COEF*entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_p += p_loss.item()
            total_v += v_loss.item()
            total_e += entropy.item()

        epoch_loss    += loss.item()
        epoch_p_loss  += total_p / PPO_EPOCHS
        epoch_v_loss  += total_v / PPO_EPOCHS
        epoch_e_loss  += total_e / PPO_EPOCHS

        pbar.set_postfix(loss=loss.item(),
                         p_loss=(total_p/PPO_EPOCHS),
                         v_loss=(total_v/PPO_EPOCHS),
                         lr=optimizer.param_groups[0]['lr'])

    scheduler.step()

    print(f"[Ep {ep}] Loss: {epoch_loss/len(train_loader):.4f} | "
          f"P: {epoch_p_loss/len(train_loader):.4f} | "
          f"V: {epoch_v_loss/len(train_loader):.4f} | "
          f"E: {epoch_e_loss/len(train_loader):.4f}")

    # 검증
    model.eval()
    model.actor.set_decode_type('greedy')
    val_cost = val_cost_opt = 0.0
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validate Ep {ep}/{EPOCHS}", ncols=100):
            loc = batch['loc'].to(DEVICE)
            dist = batch['dist'].to(DEVICE)
            dist = torch.where(torch.isinf(dist), torch.full_like(dist, FALLBACK_DIST), dist)

            _, tour, _ = model(loc)
            tour = tour[0].tolist()
            cost0,_ = env.get_costs({'loc':loc, 'dist':dist}, torch.tensor([tour],device=DEVICE))
            tour_opt, cost1 = two_opt_improved(tour, dist[0])
            val_cost     += cost0.item()
            val_cost_opt += cost1
            cnt += 1

    avg0 = val_cost     / cnt
    avg1 = val_cost_opt / cnt
    print(f"[Ep {ep}] Val Cost: {avg0:.2f} | 2-OPT Val Cost: {avg1:.2f}")

    # 체크포인트 저장
    torch.save({
        'epoch': ep,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_cost': best_val_cost
    }, CHECKPOINT)

    if avg1 < best_val_cost:
        best_val_cost = avg1
        torch.save({
            'actor': model.actor.state_dict(),
            'critic': model.value_head.state_dict()
        }, BEST_MODEL)
        print(f"🌟 New Best (2-OPT): {best_val_cost:.2f}")

# 최종 모델 저장
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
torch.save({
    'actor': model.actor.state_dict(),
    'critic': model.value_head.state_dict()
}, OUT_MODEL)
print(f"Training complete. Best 2-OPT Val Cost: {best_val_cost:.2f}")



