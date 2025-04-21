# check_over_threshold.py

import pickle
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# 1) 위도·경도 범위 (서울/경기 남부 박스)
LAT_RANGE = (37.40, 37.65)
LON_RANGE = (126.80, 127.15)

# 2) Haversine 함수 (지구 반지름=6371km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# 3) 이론 최대 거리 & Threshold (2×)
theoretical_max = haversine(LAT_RANGE[0], LON_RANGE[0],
                            LAT_RANGE[1], LON_RANGE[1])
threshold = theoretical_max * 2

# 4) 데이터 로드
with open("data/road_TSP_100_fixed.pkl", "rb") as f:
    dataset = pickle.load(f)

# 5) 계산
total_pairs_exceed = 0
instances_exceed = []
for idx, sample in enumerate(dataset):
    dist_matrix = np.array(sample["dist"], dtype=float)
    mask = dist_matrix > threshold
    cnt = mask.sum()
    if cnt > 0:
        instances_exceed.append((idx, int(cnt)))
        total_pairs_exceed += cnt

print(f"Theoretical max distance: {theoretical_max:.2f} m")
print(f"Threshold (2×): {threshold:.2f} m\n")

print(f"전체 인스턴스 수: {len(dataset)}")
print(f"> Threshold 초과 인스턴스: {len(instances_exceed)} 개")
print(f"총 초과 엔트리 (i,j) 수: {total_pairs_exceed}\n")

print("초과 인스턴스 예시 (최초 10개):")
for idx, cnt in instances_exceed[:10]:
    print(f"  샘플 #{idx}: 초과 엔트리 {cnt} 개")

