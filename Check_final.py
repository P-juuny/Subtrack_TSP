#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from define_tsp_road import TSPRoad, RoadDataset
from nets.attention_model import AttentionModel
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def load_model(model_path: str, device: torch.device) -> AttentionModel:
    """
    파인튜닝된 TSP-Road 모델을 로드합니다.
    """
    # TSPRoad()는 get_costs와 make_state를 모두 제공
    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        n_encode_layers=3,
        n_heads=8,
        tanh_clipping=10.0,
        normalization="batch",
        problem=TSPRoad()
    ).to(device)
    # 디코딩 전략 기본값 설정
    model.set_decode_type('greedy')

    # PyTorch 2.6+ 에서 weights_only=True 기본; False로 지정해 전체 체크포인트 언피클링
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def solve_with_ortools(dist_np: np.ndarray, time_limit: int = 1) -> float:
    """
    OR-Tools를 사용해 주어진 거리 행렬에 대한 최단 투어 비용을 반환합니다.
    """
    n = dist_np.shape[0]
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def callback(a, b, dm=dist_np, m=mgr):
        return int(dm[m.IndexToNode(a)][m.IndexToNode(b)])
    transit_idx = routing.RegisterTransitCallback(callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = time_limit
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    return sol.ObjectiveValue() if sol else float('nan')


def main():
    parser = argparse.ArgumentParser(description="Evaluate TSP-Road model performance")
    parser.add_argument('--model', type=str, default='pretrained/final_tsp_road.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='data/road_TSP_100_nozero.pkl',
                        help='Path to data pickle')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--decode', choices=['greedy','sampling'], default='greedy',
                        help='Decoding strategy')
    parser.add_argument('--evaluate_gap', action='store_true',
                        help='Compute gap vs OR-Tools for first --gap_instances')
    parser.add_argument('--gap_instances', type=int, default=100,
                        help='Number of instances for gap evaluation')
    parser.add_argument('--time_limit', type=int, default=1,
                        help='OR-Tools time limit per instance')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"■ Device: {device}")

    # 1) 모델 로드
    model = load_model(args.model, device)
    print(f"■ Loaded model from {args.model}")

    # 2) 테스트 데이터 로드
    dataset = RoadDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"■ Loaded {len(dataset)} instances from {args.data}")

    total_cost_m = 0.0
    total_count = 0
    gaps = []
    first_tours = []

    # 3) 추론 및 (선택) OR-Tools 비교
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            loc = batch['loc'].to(device)
            dist = batch['dist'].to(device)

            # NaN/Inf/음수 거리 처리
            mask = torch.isnan(dist) | torch.isinf(dist) | (dist < 0)
            dist = torch.where(mask, torch.full_like(dist, 10000.0), dist)

            # 디코딩 전략 설정
            model.set_decode_type(args.decode)
            cost, _, pi = model({'loc': loc, 'dist': dist}, return_pi=True)

            total_cost_m += cost.sum().item()
            batch_size = loc.size(0)
            total_count += batch_size

            # 첫 5개 투어 저장
            if len(first_tours) < 5:
                first_tours.extend(pi.cpu().numpy().tolist())

            # OR-Tools 갭 평가
            if args.evaluate_gap:
                for i in range(batch_size):
                    idx_global = batch_idx * args.batch_size + i
                    if idx_global >= args.gap_instances:
                        break
                    dist_np = dist[i].cpu().numpy().astype(np.float32)
                    seq = pi[i].cpu().numpy()
                    model_cost = float(np.sum([
                        dist_np[seq[j], seq[(j+1) % len(seq)]] for j in range(len(seq))
                    ]))
                    or_cost = solve_with_ortools(dist_np, time_limit=args.time_limit)
                    if not np.isnan(or_cost):
                        gaps.append((model_cost - or_cost) / or_cost * 100)
                if idx_global >= args.gap_instances:
                    break

    # 4) 결과 출력
    avg_cost_km = total_cost_m / total_count / 1000.0
    print(f"\n■ Average predicted cost: {avg_cost_km:.2f} km over {total_count} instances")

    print("\n■ First 5 tour examples:")
    for i, tour in enumerate(first_tours[:5]):
        print(f"  Sample {i}: {tour}")

    if args.evaluate_gap and gaps:
        gaps_arr = np.array(gaps)
        print(f"\n■ Gap vs OR-Tools on first {len(gaps_arr)} instances:")
        print(f"   Mean: {gaps_arr.mean():.2f}%")
        print(f"   Median: {np.median(gaps_arr):.2f}%")
        print(f"   Max: {gaps_arr.max():.2f}%")
        print(f"   Min: {gaps_arr.min():.2f}%")

if __name__ == '__main__':
    main()
