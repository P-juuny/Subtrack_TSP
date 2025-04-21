# visualize_predictions.py

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_predictions(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_route(ax, locs, tour, label=None, color='C0'):
    """
    locs: list of [x,y]
    tour: list of indices (e.g. [0,3,1,2,...])
    """
    pts = np.array(locs)
    tour_pts = pts[tour + [tour[0]]]  # 돌아오는 마지막 엣지 포함
    ax.plot(tour_pts[:,0], tour_pts[:,1], '-', lw=1.5, color=color, label=label)
    ax.scatter(pts[:,0], pts[:,1], s=20, color=color)
    # 시작점 강조
    ax.scatter(pts[tour[0],0], pts[tour[0],1], s=60, color=color, marker='*')

def main(args):
    preds = load_predictions(args.pred)
    idx = args.index
    if idx < 0 or idx >= len(preds):
        print(f"인덱스 {idx} 범위는 0~{len(preds)-1} 입니다.")
        return

    data = preds[idx]
    loc = data['loc']
    tour = data['tour']
    cost = data['cost']

    fig, ax = plt.subplots(figsize=(6,6))
    plot_route(ax, loc, tour, label=f"Model (cost={cost:.1f}m)", color='C0')

    ax.set_title(f"Route #{idx} – avg cost {cost:.1f} m")
    ax.axis('off')
    ax.legend(loc='upper right')

    # 출력 또는 저장
    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize TSPModel predictions")
    parser.add_argument('pred', help="predictions.json 경로")
    parser.add_argument('--index', type=int, default=0, help="시각화할 항목 인덱스 (기본: 0)")
    parser.add_argument('--out',   type=str, default=None, help="저장할 파일명 (png). 지정 안하면 화면에 표시")
    args = parser.parse_args()
    main(args)
