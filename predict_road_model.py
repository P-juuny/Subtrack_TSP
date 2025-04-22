import os
import argparse
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.font_manager as fm
import matplotlib as mpl

from nets.attention_model import AttentionModel
from define_tsp_road import TSPRoad, RoadDataset

# 한글 폰트 설정 (맥OS 기준)
# 폰트가 없으면 에러가 발생하므로 try-except로 처리
try:
    plt.rcParams['font.family'] = 'AppleGothic'
except:
    print("AppleGothic 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    plt.rcParams['font.family'] = 'sans-serif'

# ----------------------------- #
# 설정 및 파라미터
# ----------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--data_pkl', type=str, default='data/road_TSP_100_fixed.pkl')
parser.add_argument('--model', type=str, default='pretrained/best_tsp_road.pt')
parser.add_argument('--num_samples', type=int, default=30, help='테스트할 샘플 수')
parser.add_argument('--time_limit', type=int, default=5, help='시간 제한(초)')
args = parser.parse_args()

# ----------------------------- #
# 미터 단위 변환 함수
# ----------------------------- #
def convert_to_meters(cost):
    """모델 비용을 미터 단위로 변환"""
    return cost * 1000  # km -> m

# ----------------------------- #
# OR-Tools TSP 솔버
# ----------------------------- #
def create_distance_callback(dist_matrix):
    """거리 콜백 함수 생성"""
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]
    return distance_callback

def solve_tsp_ortools(dist_matrix, time_limit=5):
    """OR-Tools를 사용하여 TSP 해결"""
    # 문제 크기
    size = len(dist_matrix)
    
    # 라우팅 인덱스 관리자 생성
    global manager
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    
    # 라우팅 모델 생성
    routing = pywrapcp.RoutingModel(manager)
    
    # 거리 콜백 등록
    dist_callback_index = routing.RegisterTransitCallback(
        create_distance_callback(dist_matrix)
    )
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_index)
    
    # 검색 파라미터 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # 시간 제한 설정 (밀리초 단위)
    search_parameters.time_limit.seconds = time_limit
    
    # 솔루션 찾기
    start_time = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    solve_time = time.time() - start_time
    
    # 결과 처리
    tour = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            tour.append(node_index)
            index = solution.Value(routing.NextVar(index))
        
        # 시작점으로 돌아가기
        tour.append(0)
        
        # 경로 비용 계산
        cost = 0
        for i in range(len(tour) - 1):
            cost += dist_matrix[tour[i]][tour[i+1]]
        
        return tour, cost, solve_time, True
    else:
        # 솔루션을 찾지 못한 경우
        return [], 0, solve_time, False

# ----------------------------- #
# 메인 함수
# ----------------------------- #
def main():
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")
    
    # 데이터셋 로드
    try:
        dataset = RoadDataset(args.data_pkl)
        print(f"데이터셋 로드 완료: 총 {len(dataset)}개")
    except Exception as e:
        print(f"데이터셋 로드 오류: {e}")
        return
    
    # 모델 로드
    try:
        print(f"모델 로딩 중: {args.model}")
        env = TSPRoad()
        model = AttentionModel(
            embedding_dim=128, 
            hidden_dim=128,
            n_encode_layers=3, 
            n_heads=8,
            tanh_clipping=10.0, 
            normalization="batch",
            problem=env
        ).to(device)
        
        # 모델 가중치 로드 (PyTorch 2.6+ 호환성)
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.set_decode_type('greedy')
        print("✅ 모델 로드 및 설정 완료")
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # 테스트할 샘플 선택
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    
    # 결과 저장 변수
    model_costs = []
    ortools_costs = []
    model_tours = []
    ortools_tours = []
    
    # 샘플 테스트
    print(f"샘플 테스트 중: ", end="")
    with torch.no_grad():
        for idx in tqdm(indices):
            # 샘플 데이터 가져오기
            data = dataset[idx]
            loc = data['loc'].unsqueeze(0).to(device)  # 배치 차원 추가
            dist = data['dist'].unsqueeze(0).to(device)
            
            # 학습된 모델로 경로 생성
            start_time = time.time()
            cost, _, pi = model(loc, return_pi=True)
            model_time = time.time() - start_time
            
            # 미터 단위로 변환
            model_cost_meters = convert_to_meters(cost.item())
            model_costs.append(model_cost_meters)
            model_tours.append(pi[0].cpu().numpy())
            
            # OR-Tools로 경로 생성
            dist_matrix = data['dist'].numpy()
            tour, cost, ortools_time, success = solve_tsp_ortools(dist_matrix, args.time_limit)
            
            if success:
                ortools_costs.append(cost)
                ortools_tours.append(tour)
            else:
                # OR-Tools가 실패하면 모델 결과로 대체
                ortools_costs.append(model_cost_meters)
                ortools_tours.append(pi[0].cpu().numpy())
    
    # 평균 계산
    avg_model_cost = np.mean(model_costs)
    avg_ortools_cost = np.mean(ortools_costs)
    
    # 비율 계산 (모델 / OR-Tools)
    cost_ratios = []
    for m_cost, o_cost in zip(model_costs, ortools_costs):
        if o_cost > 0:  # 0으로 나누는 것 방지
            ratio = m_cost / o_cost
            cost_ratios.append(ratio)
    
    avg_ratio = np.mean(cost_ratios) if cost_ratios else 0
    
    # 시각화 (히스토그램)
    plt.figure(figsize=(10, 6))
    plt.hist(cost_ratios, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='모델 = OR-Tools')
    plt.axvline(x=avg_ratio, color='green', linestyle='-', linewidth=2, 
                label=f'평균 비율: {avg_ratio:.4f}')
    
    plt.title('학습된 모델 vs OR-Tools 비용 비율 분포')
    plt.xlabel('비용 비율 (모델 / OR-Tools)')
    plt.ylabel('샘플 수')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cost_ratio_histogram.png', dpi=300, bbox_inches='tight')
    print("✅ 시각화 저장 완료: cost_ratio_histogram.png")
    
    # 결과 출력
    print("\n--- 요약 ---")
    print(f"모델 평균 거리: {avg_model_cost:.2f} m")
    print(f"OR-Tools 평균 거리: {avg_ortools_cost:.2f} m")
    print(f"평균 비율 (모델 / OR-Tools): {avg_ratio:.4f}")

if __name__ == "__main__":
    main()