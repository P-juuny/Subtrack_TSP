#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_road_model2.py
학습: PPO + 2‑OPT 로컬서치
검증: 빠른 greedy 평가만 (샘플 제한)
"""

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

# 사용자 정의 모듈
from nets.attention_model import AttentionModel
from define_tsp_road import TSPRoad, RoadDataset

# -------------------------
# CLI 인자
# -------------------------
parser = argparse.ArgumentParser(description="PPO + 2-OPT 로컬서치 학습 스크립트")
parser.add_argument('--resume', action='store_true', help='이전에 저장된 체크포인트에서 재개')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='체크포인트 경로')
args = parser.parse_args()

# -------------------------
# 하이퍼파라미터
# -------------------------
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN    = "pretrained/tsp_100/epoch-99.pt"
DATA_PKL    = "data/road_TSP_100_fixed.pkl"
OUT_MODEL   = "pretrained/final_ppo_road_v2.pt"
BEST_MODEL  = "pretrained/best_ppo_road_v2.pt"
CHECKPOINT  = args.checkpoint

BATCH_SIZE  = 64
EPOCHS      = 300
LR_INIT     = 3e-5
ETA_MIN     = 1e-7
PPO_EPOCHS  = 3
CLIP_EPS    = 0.2
ENT_COEF    = 0.01
VALUE_COEF  = 0.5
VAL_SPLIT   = 0.1
SEED        = 42

VAL_LIMIT   = 100  # 검증에 사용할 최대 샘플 수

# 재현성 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# Actor-Critic 정의
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
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, loc):
        # loc: (batch, nodes, 2)
        _, logp, tour = self.actor(loc, return_pi=True)
        # Critic: actor 인코더 출력의 평균 풀링
        # 내부 구현 의존적이므로 간단히 loc 평균 사용
        pooled = loc.mean(dim=1)
        value = self.value_head(pooled).squeeze(-1)
        return logp, tour, value

# -------------------------
# 데이터 로드 및 분할
# -------------------------
dataset = RoadDataset(DATA_PKL)
n = len(dataset)
indices = list(range(n))
random.shuffle(indices)
split = int(n * VAL_SPLIT)
train_idx, val_idx = indices[split:], indices[:split]
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset: total={n}, train={len(train_idx)}, val={len(val_idx)}")
print(f"Device: {DEVICE}")

# -------------------------
# 모델 / 옵티마이저 / 스케줄러
# -------------------------
env       = TSPRoad()
model     = ActorCritic().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR_INIT)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

# 체크포인트 로드(재개)
start_epoch = 1
best_val_cost = float('inf')
if args.resume and os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    best_val_cost = ckpt.get('best_val_cost', best_val_cost)
    start_epoch = ckpt.get('epoch', 1) + 1
    print(f"🔄 Resume from epoch {start_epoch}")

# 사전학습 가중치 로드
elif start_epoch == 1 and os.path.exists(PRETRAIN):
    pre = torch.load(PRETRAIN, map_location=DEVICE)
    # actor 부분만 로드
    model.actor.load_state_dict(pre.get('model', pre.get('actor')), strict=False)
    print("✅ Loaded pretrained actor weights")

# -------------------------
# 학습 + 검증 루프
# -------------------------
for ep in range(start_epoch, EPOCHS+1):
    model.train()
    model.actor.set_decode_type('sampling')
    total_p_loss = total_v_loss = total_e_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Train Ep {ep}/{EPOCHS}", ncols=80)
    for batch in pbar:
        loc  = batch['loc'].to(DEVICE)
        dist = batch['dist'].to(DEVICE)
        # inf→큰 값 치환
        dist = torch.where(torch.isinf(dist), torch.full_like(dist, 1e6), dist)

        # rollout
        logp, tour, value = model(loc)
        cost, _ = env.get_costs({'dist': dist}, tour)
        reward = -cost

        # advantage
        adv = reward - value.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO 업데이트
        old_logp = logp.detach()
        policy_loss = value_loss = entropy = None
        for _ in range(PPO_EPOCHS):
            logp_new, _, value_new = model(loc)
            ratio = torch.exp(logp_new - old_logp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
            policy_loss = -torch.min(s1, s2).mean()
            value_loss  = F.mse_loss(value_new, reward)
            entropy     = -logp_new.mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_p_loss += policy_loss.item()
        total_v_loss += value_loss.item()
        total_e_loss += entropy.item()

    scheduler.step()
    print(f"[Ep {ep}] P_loss: {total_p_loss/len(train_loader):.4f}, V_loss: {total_v_loss/len(train_loader):.4f}, Ent: {total_e_loss/len(train_loader):.4f}")

    # ── 빠른 검증 ──
    model.eval()
    model.actor.set_decode_type('greedy')
    val_cost = 0.0
    cnt = 0

    with torch.no_grad():
        for batch in val_loader:
            if cnt >= VAL_LIMIT: break
            loc, dist = batch['loc'].to(DEVICE), batch['dist'].to(DEVICE)
            _, tour, _ = model(loc, return_pi=True)
            cost, _ = env.get_costs({'dist': dist}, tour)
            val_cost += cost.item()
            cnt += loc.size(0)

    avg_val = val_cost / cnt if cnt>0 else float('nan')
    print(f"[Ep {ep}] 빠른검증 Val Cost({cnt}샘플): {avg_val:.2f}")

    # 체크포인트 저장
    torch.save({
        'epoch': ep,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_cost': best_val_cost
    }, CHECKPOINT)

    # 최고 모델 갱신
    if avg_val < best_val_cost:
        best_val_cost = avg_val
        torch.save(model.state_dict(), BEST_MODEL)
        print(f"🌟 New Best Val Cost: {best_val_cost:.2f}")

# 최종 모델 저장
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
torch.save(model.state_dict(), OUT_MODEL)
print(f"✅ Training complete. Best Val Cost: {best_val_cost:.2f}")




