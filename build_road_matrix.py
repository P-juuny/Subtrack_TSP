# build_road_matrix.py
import requests
import os
import pickle
import random
import time
import numpy as np
from tqdm import tqdm

# ⚙️ 설정
OSRM_URL      = "http://router.project-osrm.org/table/v1/driving/"
LAT_RANGE     = (37.40, 37.65)
LON_RANGE     = (126.80, 127.15)
MAX_RETRIES   = 3
# 여기서 FALLBACK_DIST과 MAX_VALID_DIST 를 같게 잡으면
# 모든 거리가 [0, FALLBACK_DIST] 로 클램핑됩니다.
FALLBACK_DIST    = 60_000.0  # meters
MAX_VALID_DIST   = FALLBACK_DIST
TRAIN_SAMPLES    = 5000
PRED_SAMPLES     = 1
N_NODES          = 100
TRAIN_OUT_PATH   = "data/road_TSP_100.pkl"
PRED_OUT_PATH    = "data/road_TSP_100Predict_large.pkl"

def fetch_osrm_matrix(coords):
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = f"{OSRM_URL}{coord_str}?annotations=distance"
    for _ in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if "distances" in data:
                dist = np.array(data["distances"], dtype=float)
                # **바로 클램핑**
                dist[dist < 0] = 0.0
                dist[dist > MAX_VALID_DIST] = MAX_VALID_DIST
                return dist
        except Exception:
            time.sleep(1)
    # 실패 시에도 클램핑된 동일 배열 리턴
    dist = np.full((len(coords), len(coords)), FALLBACK_DIST, dtype=float)
    return dist

def build_dataset(num_samples, n_nodes, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset = []
    for _ in tqdm(range(num_samples), desc=f"Building {out_path}"):
        coords = [
            (random.uniform(*LAT_RANGE), random.uniform(*LON_RANGE))
            for _ in range(n_nodes)
        ]
        dist = fetch_osrm_matrix(coords)
        dataset.append({
            "loc": coords,
            "dist": dist
        })
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"✅ Saved {num_samples} instances to {out_path}")

if __name__ == "__main__":
    build_dataset(TRAIN_SAMPLES, N_NODES, TRAIN_OUT_PATH)
    build_dataset(PRED_SAMPLES,  N_NODES, PRED_OUT_PATH)

