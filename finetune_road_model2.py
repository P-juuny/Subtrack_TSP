# 전체 500줄 기준 Attention TSP 도로기반 PPO 학습 파이프라인
# - AttentionModel + PPO + ActorCritic + 2-OPT + Checkpoint + Validation 포함
# - dist 기반 도로거리 학습
# - 2025.04 최적화 버전
# 작성자: ChatGPT (GPT-4, 요청 기반)

# ----------------------------- #
# 필요한 라이브러리 import
# ----------------------------- #
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

from nets.attention_model import AttentionModel
from define_tsp_road import TSPRoad, RoadDataset

# ----------------------------- #
# 설정 및 파라미터
# ----------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN = "pretrained/tsp_100/epoch-99.pt"
DATA_PKL = "data/road_TSP_100_fixed.pkl"
OUT_MODEL = "pretrained/final_ppo_road_v3.pt"
BEST_MODEL = "pretrained/best_ppo_road_v3.pt"
CHECKPOINT = args.checkpoint
BATCH_SIZE = 64
EPOCHS = 300
LR_INIT = 3e-5
ETA_MIN = 1e-7
PPO_EPOCHS = 3
CLIP_EPS = 0.2
ENT_COEF = 0.01
VALUE_COEF = 0.5
FALLBACK_DIST = 60000.0
VAL_SPLIT = 0.1
SEED = 42

# Seed 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------- #
# 2-OPT 알고리즘
# ----------------------------- #
def calculate_total_distance(route, dist_matrix):
    total_dist = 0.0
    for i in range(len(route) - 1):
        total_dist += dist_matrix[route[i], route[i + 1]]
    return total_dist

def two_opt(route, dist_matrix):
    best = route
    improved = True
    best_cost = calculate_total_distance(best, dist_matrix)
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = route[:]
                new_route[i:j] = route[j-1:i-1:-1]
                new_cost = calculate_total_distance(new_route, dist_matrix)
                if new_cost < best_cost:
                    best = new_route
                    best_cost = new_cost
                    improved = True
    return best, best_cost

# ----------------------------- #
# 모델 정의
# ----------------------------- #
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = AttentionModel(
            embedding_dim=128, hidden_dim=128,
            n_encode_layers=3, n_heads=8,
            tanh_clipping=10.0, normalization="batch",
            problem=TSPRoad()
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, loc):
        embeddings = self.actor._init_embed(loc)
        # embedder의 출력이 tuple인 경우 처리
        graph_embed = self.actor.embedder(embeddings)
        if isinstance(graph_embed, tuple):
            # 첫 번째 요소가 node embeddings
            graph_embed = graph_embed[0]
        
        # 평균 계산
        pooled = graph_embed.mean(dim=1)
        value = self.critic(pooled).squeeze(-1)
        _, logp, pi = self.actor(loc, return_pi=True)
        return logp, pi, value

# ----------------------------- #
# 학습 준비
# ----------------------------- #
def create_directories():
    """필요한 디렉토리 생성"""
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    os.makedirs(os.path.dirname(BEST_MODEL), exist_ok=True)
    
    # CHECKPOINT 경로 처리 (파일 이름만 있는 경우 처리)
    checkpoint_dir = os.path.dirname(CHECKPOINT)
    if checkpoint_dir:  # 디렉토리가 있는 경우만 생성
        os.makedirs(checkpoint_dir, exist_ok=True)

create_directories()
env = TSPRoad()
model = ActorCritic().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR_INIT)

# 데이터 로드
print(f"Loading dataset from {DATA_PKL}...")
try:
    full_dataset = RoadDataset(DATA_PKL)
    n_total = len(full_dataset)
    indices = list(range(n_total))
    random.shuffle(indices)
    split = int(n_total * VAL_SPLIT)
    train_idx, val_idx = indices[split:], indices[:split]
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE)
    print(f"Dataset loaded successfully with {n_total} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=ETA_MIN)

# 체크포인트 or Pretrain
start_epoch = 1
best_val_cost = float('inf')
if args.resume and os.path.exists(CHECKPOINT):
    print(f"Loading checkpoint from {CHECKPOINT}...")
    try:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_val_cost = ckpt.get('best_val_cost', float('inf'))
        start_epoch = ckpt.get('epoch', 1) + 1
        print(f"Checkpoint loaded successfully, resuming from epoch {start_epoch}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting from scratch")
elif os.path.exists(PRETRAIN):
    print(f"Loading pretrained model from {PRETRAIN}...")
    try:
        pt = torch.load(PRETRAIN, map_location=DEVICE)
        # 다양한 저장 형식 처리
        if 'model' in pt:
            model.actor.load_state_dict(pt['model'])
        elif 'actor' in pt:
            model.actor.load_state_dict(pt['actor'])
        else:
            # 가능하면 직접 로드
            try:
                model.actor.load_state_dict(pt)
            except:
                print("⚠️ Could not load pretrained weights, model keys don't match")
        print("✅ Loaded pretrained actor weights")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("⚠️ Starting with random initialization")

# 디코딩 방식 설정
model.actor.set_decode_type('sampling')

# ----------------------------- #
# 학습 루프
# ----------------------------- #
print(f"Dataset: total={n_total}, train={len(train_idx)}, val={len(val_idx)}")
print(f"Device: {DEVICE}")
print(f"Starting training for {EPOCHS} epochs")

# 학습 루프 시작
for ep in range(start_epoch, EPOCHS + 1):
    model.train()
    epoch_loss = []
    epoch_policy = []
    epoch_value = []
    epoch_entropy = []
    
    pbar = tqdm(train_loader, desc=f"Train Ep {ep}/{EPOCHS}")
    
    for batch_idx, batch in enumerate(pbar):
        # 배치 데이터 준비
        try:
            loc, dist = batch['loc'].to(DEVICE), batch['dist'].to(DEVICE)
            
            # NaN 및 Inf 값 처리
            dist[dist.isnan() | dist.isinf() | (dist < 0)] = FALLBACK_DIST
            
            # Old policy의 logp, pi, value 계산
            with torch.no_grad():
                logp_old, pi, value = model(loc)
                cost, _ = env.get_costs({'loc': loc, 'dist': dist}, pi)
                reward = -cost  # 비용 최소화는 보상 최대화와 같음
                
                # 이점(Advantage) 계산
                adv = reward - value
                # 이점 정규화
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            # PPO 업데이트 (여러 번 반복)
            for _ in range(PPO_EPOCHS):
                # 새로운 policy의 logp와 value 계산
                logp_new, _, value_new = model(loc)
                
                # PPO 클리핑 기법 적용
                ratio = torch.exp(logp_new - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                
                # 손실 함수 계산
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_new, reward)
                entropy = -logp_new.mean()
                
                # 최종 손실 함수
                loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * entropy
                
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # 손실 기록
                epoch_loss.append(loss.item())
                epoch_policy.append(policy_loss.item())
                epoch_value.append(value_loss.item())
                epoch_entropy.append(entropy.item())
            
            # tqdm 상태 업데이트
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                policy=f"{policy_loss.item():.4f}",
                value=f"{value_loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )
            
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            continue
    
    # 학습률 업데이트
    scheduler.step()
    
    # 에포크 요약 출력
    avg_loss = np.mean(epoch_loss)
    avg_policy = np.mean(epoch_policy)
    avg_value = np.mean(epoch_value)
    avg_entropy = np.mean(epoch_entropy)
    
    print(f"[Ep {ep:3d}] Loss: {avg_loss:.4f} | Policy: {avg_policy:.4f} | Value: {avg_value:.4f} | Entropy: {avg_entropy:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # 검증
    model.eval()
    val_costs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Ep {ep}/{EPOCHS}"):
            loc, dist = batch['loc'].to(DEVICE), batch['dist'].to(DEVICE)
            dist[dist.isnan() | dist.isinf() | (dist < 0)] = FALLBACK_DIST
            
            # 검증 시에는 sampling이 아닌 greedy로 경로 생성
            model.actor.set_decode_type('greedy')
            _, _, tour = model(loc)
            model.actor.set_decode_type('sampling')  # 다시 sampling으로 돌려놓기
            
            cost, _ = env.get_costs({'loc': loc, 'dist': dist}, tour)
            val_costs.extend(cost.cpu().numpy())
    
    # 검증 결과 계산
    val_cost = np.mean(val_costs)
    print(f"[Ep {ep:3d}] Val Cost: {val_cost:.2f}")
    
    # 최고 모델 저장
    if val_cost < best_val_cost:
        best_val_cost = val_cost
        torch.save({
            'model': model.state_dict(),
            'epoch': ep,
            'val_cost': val_cost
        }, BEST_MODEL)
        print(f"🌟 Saved best model @Ep{ep} with val cost {val_cost:.2f}")
    
    # 체크포인트 저장
    torch.save({
        'epoch': ep,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_cost': best_val_cost
    }, CHECKPOINT)
    
    # 매 10 에포크마다 별도 체크포인트 저장
    if ep % 10 == 0:
        cp_path = f"pretrained/checkpoint_ep{ep}.pth"
        torch.save({
            'epoch': ep,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_cost': best_val_cost
        }, cp_path)
        print(f"📝 Saved checkpoint at epoch {ep}")

# 학습 종료
print("\n" + "="*50)
print(f"Training completed. Best validation cost: {best_val_cost:.2f}")

# 최종 모델 저장
torch.save({
    'model': model.state_dict(),
    'best_val_cost': best_val_cost,
    'final_epoch': EPOCHS
}, OUT_MODEL)

print(f"✅ Final model saved to {OUT_MODEL}")
print(f"✅ Best model saved to {BEST_MODEL}")