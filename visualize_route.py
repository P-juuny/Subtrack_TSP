import json
import matplotlib
matplotlib.use("Agg")   # 화면 대신 파일에 저장하도록
import matplotlib.pyplot as plt

# ── 데이터 불러오기 ──
with open("predictions.json") as f:
    rl_preds = json.load(f)
with open("or_tools.json") as f:
    or_preds = json.load(f)

# 첫 번째 샘플만 비교
rl = rl_preds[0]
or_ = or_preds[0]

# 경로 따라 좌표 뽑기
rl_pts = [rl["loc"][i] for i in rl["tour"]]
or_pts = [or_["loc"][i] for i in or_["tour"]]
lons_rl, lats_rl = zip(*rl_pts)
lons_or, lats_or = zip(*or_pts)

# 그리기
plt.figure(figsize=(6,6))
plt.plot(lons_rl, lats_rl,  "o-", label=f"RL cost={rl['cost']:.2f}")
plt.plot(lons_or, lats_or, "x--", label=f"OR cost={or_['cost']:.2f}")
plt.legend()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("RL vs OR‑Tools Route Comparison")
plt.tight_layout()
plt.savefig("route_comparison.png")
plt.close()
print("Saved ▶ route_comparison.png")
