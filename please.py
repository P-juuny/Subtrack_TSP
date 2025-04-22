#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.serialization

from define_tsp_road import TSPRoad, RoadDataset
from nets.attention_model import AttentionModel

# 설정
DATA_PKL   = "data/road_TSP_100_nozero.pkl"
MODEL_PATH = "pretrained/best_tsp_road.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

# 데이터 로드
dataset = RoadDataset(DATA_PKL)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 생성
env = TSPRoad()
model = AttentionModel(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=3,
    n_heads=8,
    tanh_clipping=10.0,
    normalization="batch",
    problem=env
).to(DEVICE)

# =============================
# 안전하게 체크포인트 언피클
# =============================
# numpy.core.multiarray.scalar 을 allow‐list에 추가
from numpy.core.multiarray import scalar as np_scalar

with torch.serialization.safe_globals([np_scalar]):
    ckpt = torch.load(MODEL_PATH,
                      map_location=DEVICE,
                      weights_only=False)  # 반드시 False

# state_dict 꺼내기
state = ckpt.get("model", ckpt)
model.load_state_dict(state)
model.eval()

# 추론
pred_kms = []
with torch.no_grad():
    for idx, batch in enumerate(loader):
        loc  = batch["loc"].to(DEVICE)
        dist = batch["dist"].to(DEVICE)

        model.set_decode_type("greedy")
        cost, ll, pi = model({"loc":loc, "dist":dist}, return_pi=True)
        km = cost.item() / 1000.0
        pred_kms.append(km)
        print(f"Instance {idx:3d}: {km:.2f} km")

arr = np.array(pred_kms)
print("\n▶ 평균: {:.2f} km".format(arr.mean()))
print("▶ 최소: {:.2f} km".format(arr.min()))
print("▶ 최대: {:.2f} km".format(arr.max()))
