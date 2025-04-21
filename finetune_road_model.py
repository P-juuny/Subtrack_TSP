import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
from nets.attention_model import AttentionModel
from define_tsp_road import TSPRoad, RoadDataset

# ── 설정 ──
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN      = "pretrained/tsp_100/epoch-99.pt"
DATA_PKL      = "data/road_TSP_100_fixed.pkl"
OUT_MODEL     = "pretrained/tsp_100_road_finetuned.pt"
BEST_MODEL    = "pretrained/best_road_finetuned.pt"
BATCH_SIZE    = 64
EPOCHS        = 500
LR            = 1e-5      # 학습률 낮춤
SCALE         = 1e5       # 거리 단위 축소 (100km 단위)
NUM_SAMPLES   = 16        # 샘플링 횟수 증가
ENT_COEF      = 0.01      # 엔트로피 정규화 계수
FALLBACK_DIST = 60000.0   # OSRM 실패 시 대체 거리
VAL_SPLIT     = 0.1       # 검증셋 비율
SEED          = 42
EARLY_STOPPING_PATIENCE = 50  # 개선 없을 때 중단

# 재현성을 위해 시드 고정
random.seed(SEED)
torch.manual_seed(SEED)

# 모델 초기화 및 사전학습 가중치 로드
model = AttentionModel(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=3,
    n_heads=8,
    tanh_clipping=10.0,
    normalization="batch",  # LayerNorm 키 오류 방지를 위해 batchnorm 사용,  # LayerNorm 적용
    problem=TSPRoad()
).to(DEVICE)
ckpt = torch.load(PRETRAIN, map_location=DEVICE)
model.load_state_dict(ckpt["model"], strict=False)
model.set_decode_type("sampling")

# 옵티마이저 및 LR 스케줄러
opt = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)

# Best 모델 및 Early Stopping 변수
best_val_cost = float('inf')
no_improve_epochs = 0

# EMA 베이스라인 초기화
ema_baseline = None

# 데이터셋 로드 및 분할
dataset = RoadDataset(DATA_PKL)
n = len(dataset)
idxs = list(range(n))
random.shuffle(idxs)
split = int(n * VAL_SPLIT)
val_idxs, train_idxs = idxs[:split], idxs[split:]
train_ds = Subset(dataset, train_idxs)
val_ds   = Subset(dataset, val_idxs)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# 디버그: 데이터셋 크기 및 배치 수 확인
print(f"전체 데이터: {n}, 학습셋: {len(train_idxs)}, 검증셋: {len(val_idxs)}")
print(f"학습용 배치 수: {len(train_loader)}, 검증용 배치 수: {len(val_loader)}")

val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

for ep in range(1, EPOCHS+1):
    # ── TRAIN ──
    model.train()
    total_loss = 0.0
    total_cost = 0.0
    train_pbar = tqdm(train_loader, desc=f"Train Epoch {ep}/{EPOCHS}", ncols=100)
    for batch in train_pbar:
        loc_t  = batch["loc"].to(DEVICE)
        dist_t = batch["dist"].to(DEVICE)
        # fallback 처리
        mask = torch.isinf(dist_t)
        if mask.any():
            dist_t[mask] = FALLBACK_DIST

        # multi-sample REINFORCE
        log_liks = []
        rewards  = []
        for _ in range(NUM_SAMPLES):
            _, log_likelihood, pi = model(loc_t, return_pi=True)
            cost_real, _ = TSPRoad().get_costs({"dist": dist_t}, pi)
            # 비용을 음수 보상으로 사용
            rewards.append((-cost_real) / SCALE)
            log_liks.append(log_likelihood)
        log_liks = torch.stack(log_liks, dim=0)   # (S, B)
        rewards  = torch.stack(rewards, dim=0)    # (S, B)

        # EMA 베이스라인 업데이트
        batch_baseline = rewards.mean(dim=0, keepdim=True)
        ema_baseline   = batch_baseline if ema_baseline is None else 0.9 * ema_baseline + 0.1 * batch_baseline

        # 어드밴티지 계산 및 정규화
        advantages = rewards - ema_baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # 손실 계산
        loss_rl = -(advantages.detach() * log_liks).mean()
        ent     = -(log_liks.exp() * log_liks).mean()
        loss    = loss_rl - ENT_COEF * ent

        # 역전파 및 그래디언트 클리핑
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        batch_loss = loss.item()
        batch_cost = cost_real.mean().item()
        total_loss += batch_loss
        total_cost += batch_cost

        train_pbar.set_postfix(batch_loss=f"{batch_loss:.4f}",
                               batch_cost=f"{batch_cost:.2f}")

    avg_train_loss = total_loss / len(train_loader)
    avg_train_cost = total_cost / len(train_loader)

    # ── VALIDATION ──
    model.eval()
    model.set_decode_type("greedy")
    val_cost = 0.0
    with torch.no_grad():
        for batch in val_loader:
            loc_t  = batch["loc"].to(DEVICE)
            dist_t = batch["dist"].to(DEVICE)
            mask = torch.isinf(dist_t)
            if mask.any():
                dist_t[mask] = FALLBACK_DIST
            _, _, pi = model(loc_t, return_pi=True)
            cost_real, _ = TSPRoad().get_costs({"dist": dist_t}, pi)
            val_cost += cost_real.mean().item()
    avg_val_cost = val_cost / len(val_loader)
    model.set_decode_type("sampling")

    # ── 에폭 요약 로그 ──
    print(f"[Epoch {ep}] "
          f"train_loss: {avg_train_loss:.4f} | "
          f"train_cost: {avg_train_cost:.2f} m | "
          f"val_cost:   {avg_val_cost:.2f} m")

    # ── Best 모델 체크포인트 ──
    if avg_val_cost < best_val_cost:
        best_val_cost = avg_val_cost
        torch.save({"model": model.state_dict()}, BEST_MODEL)
        print(f"🌟 New best model at epoch {ep}, val_cost = {best_val_cost:.2f}")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    # ── LR 스케줄러 및 Early-Stopping ──
    scheduler.step(avg_val_cost)
    if no_improve_epochs > EARLY_STOPPING_PATIENCE:
        print(f"🔔 Early stopping at epoch {ep} (no improvement for {no_improve_epochs} epochs)")
        break

# ── 최종 모델 저장 ──
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
torch.save({"model": model.state_dict()}, OUT_MODEL)
print(f"✅ Saved final model to {OUT_MODEL}")
print(f"✅ Best validation model saved to {BEST_MODEL}")

