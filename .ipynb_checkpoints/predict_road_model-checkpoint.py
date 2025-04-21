import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from define_tsp_road import TSPRoad, RoadDataset
from nets.attention_model import AttentionModel

# ── 환경 설정 ──
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH   = "pretrained/best_road_finetuned.pt"
DATA_PKL     = "data/road_TSP_100_fixed.pkl"
BATCH_SIZE   = 1      # 한 번에 한 인스턴스씩
NUM_SAMPLES  = 50     # best-of-50 샘플링

# ── 모델 로드 ──
model = AttentionModel(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=3,
    n_heads=8,
    tanh_clipping=10.0,
    normalization="batch",
    problem=TSPRoad()
).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.set_decode_type("sampling")

# ── 데이터 로드 ──
dataset = RoadDataset(DATA_PKL)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

preds = []
# wrap the outer loop in tqdm
for batch in tqdm(loader, desc="Predicting routes", ncols=100):
    loc  = batch["loc"].to(DEVICE)
    dist = batch["dist"].to(DEVICE)

    best_cost = float('inf')
    best_pi   = None

    # K번 샘플링해서 가장 낮은 cost 선택
    for _ in range(NUM_SAMPLES):
        _, logp, pi = model(loc, return_pi=True)
        cost_real, _ = TSPRoad().get_costs({"dist": dist}, pi)
        c = cost_real.mean().item()
        if c < best_cost:
            best_cost = c
            best_pi   = pi.cpu().tolist()

    preds.append({
        "loc":  loc.cpu().tolist()[0],
        "tour": best_pi[0],
        "cost": best_cost
    })

# ── 결과 저장 ──
with open("predictions.json", "w") as f:
    json.dump(preds, f)

avg = sum(p["cost"] for p in preds) / len(preds)
print(f"\nSaved best‑of‑{NUM_SAMPLES} predictions → avg cost = {avg:.2f} m")


