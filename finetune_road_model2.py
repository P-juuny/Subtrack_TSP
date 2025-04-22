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

from nets.attention_model import AttentionModel
from define_tsp_road import TSPRoad, RoadDataset

# ----------------------------- #
# 설정 및 파라미터
# ----------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
parser.add_argument('--data_pkl', type=str, default='data/road_TSP_100_nozero.pkl')
parser.add_argument('--pretrain', type=str, default='pretrained/tsp_100/epoch-99.pt')
parser.add_argument('--out_model', type=str, default='pretrained/final_tsp_road.pt')
parser.add_argument('--best_model', type=str, default='pretrained/best_tsp_road.pt')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--clip_grad', type=float, default=1.0)
args = parser.parse_args()

# 기본 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN = args.pretrain
DATA_PKL = args.data_pkl
OUT_MODEL = args.out_model
BEST_MODEL = args.best_model
CHECKPOINT = args.checkpoint
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR_INIT = args.lr
ETA_MIN = 1e-7
FALLBACK_DIST = 10000.0  # 최대 거리 클리핑 값 (미터 단위) - OSRM은 미터 단위로 반환
VAL_SPLIT = 0.1
SEED = 42
MAX_GRAD_NORM = args.clip_grad  # 그래디언트 클리핑 값
WARMUP_EPOCHS = args.warmup_epochs  # 웜업 에포크

# Seed 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------- #
# 유틸리티 함수
# ----------------------------- #
def create_directory(path):
    """디렉토리 생성"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def fix_bad_values(tensor):
    """문제가 있는 텐서 값 수정"""
    return torch.where(
        torch.isnan(tensor) | torch.isinf(tensor), 
        torch.zeros_like(tensor), 
        tensor
    )

# 단위 환산 함수
def convert_to_km(meters):
    """미터 단위 비용을 킬로미터로 변환"""
    return meters / 1000.0

# ----------------------------- #
# Learning Rate 스케줄러
# ----------------------------- #
class WarmupScheduler:
    """
    웜업 후 코사인 스케줄러 적용
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.warmup_epochs = warmup_epochs
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs-warmup_epochs, eta_min=eta_min)
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.eta_min + factor * (self.initial_lr - self.eta_min)
        else:
            # Cosine annealing
            self.cosine_scheduler.step()
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# ----------------------------- #
# 메인 함수
# ----------------------------- #
def main():
    # 1. 디렉토리 생성
    create_directory(OUT_MODEL)
    create_directory(BEST_MODEL)
    create_directory(CHECKPOINT)
    
    # 2. 환경 및 모델 생성
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
    
    # 3. 데이터 로드
    print(f"데이터셋 로딩 중: {DATA_PKL}")
    try:
        full_dataset = RoadDataset(DATA_PKL)
        n_total = len(full_dataset)
        
        # 학습/검증 세트 분할
        indices = list(range(n_total))
        random.shuffle(indices)
        split = int(n_total * VAL_SPLIT)
        train_idx, val_idx = indices[split:], indices[:split]
        
        # 데이터 전처리 - 평균과 표준편차 계산
        dist_values = []
        for idx in range(min(100, n_total)):  # 100개 샘플로 통계 계산
            sample = full_dataset[idx]
            dist = sample['dist']
            dist_values.extend(dist.flatten().tolist())
        
        dist_values = np.array(dist_values)
        dist_values = dist_values[~np.isnan(dist_values) & ~np.isinf(dist_values) & (dist_values > 0)]
        
        # 통계 출력
        mean_dist = np.mean(dist_values)
        median_dist = np.median(dist_values)
        min_dist = np.min(dist_values)
        max_dist = np.max(dist_values)
        
        print(f"거리 통계 (미터 단위):")
        print(f"  - 평균: {mean_dist:.2f}m ({convert_to_km(mean_dist):.2f}km)")
        print(f"  - 중앙값: {median_dist:.2f}m ({convert_to_km(median_dist):.2f}km)")
        print(f"  - 최소: {min_dist:.2f}m ({convert_to_km(min_dist):.2f}km)")
        print(f"  - 최대: {max_dist:.2f}m ({convert_to_km(max_dist):.2f}km)")
        
        # 데이터 로더
        train_loader = DataLoader(
            Subset(full_dataset, train_idx), 
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx), 
            batch_size=BATCH_SIZE
        )
        print(f"데이터셋 로드 완료: 총 {n_total}개 (학습 {len(train_idx)}, 검증 {len(val_idx)})")
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return
    
    # 4. 옵티마이저 & 스케줄러
    optimizer = Adam(model.parameters(), lr=LR_INIT)
    scheduler = WarmupScheduler(
        optimizer, 
        warmup_epochs=WARMUP_EPOCHS,
        max_epochs=EPOCHS,
        eta_min=ETA_MIN
    )
    
    # 5. 모델 초기화/로드
    start_epoch = 1
    best_val_cost = float('inf')
    
    # 체크포인트에서 복원
    if args.resume and os.path.exists(CHECKPOINT):
        print(f"체크포인트 로딩: {CHECKPOINT}")
        try:
            ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            best_val_cost = ckpt.get('best_val_cost', float('inf'))
            start_epoch = ckpt.get('epoch', 1) + 1
            
            # 스케줄러 복원 (웜업 스케줄러는 상태를 저장하지 않으므로 현재 에포크만 설정)
            scheduler.current_epoch = start_epoch - 1
            print(f"에포크 {start_epoch}부터 학습 재개")
        except Exception as e:
            print(f"체크포인트 로드 오류: {e}")
            print("처음부터 학습을 시작합니다")
    
    # 사전 학습 모델 로드
    elif os.path.exists(PRETRAIN):
        print(f"사전학습 모델 로딩: {PRETRAIN}")
        try:
            pt = torch.load(PRETRAIN, map_location=DEVICE)
            if 'model' in pt:
                model.load_state_dict(pt['model'])
            else:
                model.load_state_dict(pt)
            print("✅ 사전학습 가중치 로드 완료")
        except Exception as e:
            print(f"사전학습 모델 로드 오류: {e}")
            print("⚠️ 랜덤 초기화로 시작합니다")
    
    # 6. 학습 시작
    print(f"환경: {DEVICE}")
    print(f"총 {EPOCHS}개 에포크 학습 시작 (웜업 {WARMUP_EPOCHS}에포크)")
    
    # 학습 진행 기록
    train_costs_history = []
    val_costs_history = []
    
    for epoch in range(start_epoch, EPOCHS + 1):
        # 학습 모드
        model.train()
        
        # 에포크 통계
        epoch_loss = 0
        epoch_reward = 0
        processed_batches = 0
        
        # 학습 루프
        train_pbar = tqdm(train_loader, desc=f"학습 {epoch}/{EPOCHS}")
        for batch_idx, batch in enumerate(train_pbar):
            try:
                # 배치 데이터 준비
                loc, dist = batch['loc'].to(DEVICE), batch['dist'].to(DEVICE)
                
                # 비정상 값 제거
                dist = torch.where(
                    torch.isnan(dist) | torch.isinf(dist) | (dist < 0),
                    torch.tensor(FALLBACK_DIST, device=DEVICE),
                    dist
                )
                
                # 모델 입력 준비
                batch_input = {'loc': loc, 'dist': dist}
                
                # 중요: AttentionModel의 디코딩 타입 명시적 설정
                model.set_decode_type('sampling')
                
                # cost, ll을 직접 얻기 위해 모델 호출 (return_pi=False로 설정)
                cost, ll = model(batch_input)
                
                # 손실 계산 (log_likelihood를 최대화하는 것이 목표)
                loss = -ll.mean()
                
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                # 통계 업데이트
                epoch_loss += loss.item()
                # costs는 이미 미터 단위
                cost_m = cost.mean().item()
                epoch_reward += -cost_m  # 메트릭 단위로 저장
                processed_batches += 1
                
                # 진행률 업데이트
                train_pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg_loss=f"{epoch_loss/processed_batches:.4f}",
                    reward=f"{-cost_m:.2f}m ({convert_to_km(-cost_m):.2f}km)"  # 킬로미터 및 미터 단위로 표시
                )
            except Exception as e:
                print(f"\n배치 {batch_idx} 학습 중 오류: {e}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # 학습률 업데이트
        scheduler.step()
        
        # 에포크 통계 출력
        if processed_batches > 0:
            avg_loss = epoch_loss / processed_batches
            avg_reward = epoch_reward / processed_batches
            train_costs_history.append(-avg_reward)  # 비용은 음수 보상
            
            print(f"[에포크 {epoch:3d}] 학습 손실: {avg_loss:.4f} | " 
                  f"비용: {-avg_reward:.2f}m ({convert_to_km(-avg_reward):.2f}km) | "
                  f"학습률: {scheduler.get_lr():.2e}")
        else:
            print(f"[에포크 {epoch:3d}] 경고: 유효한 배치가 없습니다")
        
        # 검증
        model.eval()
        model.set_decode_type('greedy')  # 검증에는 greedy 사용
        
        val_costs = []
        val_pbar = tqdm(val_loader, desc=f"검증 {epoch}/{EPOCHS}")
        
        with torch.no_grad():
            for batch in val_pbar:
                try:
                    # 배치 데이터 준비
                    loc, dist = batch['loc'].to(DEVICE), batch['dist'].to(DEVICE)
                    
                    # 비정상 값 제거
                    dist = torch.where(
                        torch.isnan(dist) | torch.isinf(dist) | (dist < 0),
                        torch.tensor(FALLBACK_DIST, device=DEVICE),
                        dist
                    )
                    
                    # 모델 입력
                    batch_input = {'loc': loc, 'dist': dist}
                    
                    # 검증을 위한 비용 계산
                    # 검증에서는 return_pi=True로 설정하여 경로도 얻음
                    cost, _, pi = model(batch_input, return_pi=True)
                    
                    # 유효한 비용만 저장 (이미 미터 단위)
                    valid_costs = cost[~torch.isnan(cost) & ~torch.isinf(cost)]
                    if len(valid_costs) > 0:
                        val_costs.extend(valid_costs.cpu().numpy())
                
                except Exception as e:
                    print(f"검증 배치 처리 중 오류: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue
        
        # 검증 결과 계산
        if val_costs:
            val_cost = np.mean(val_costs)
            val_costs_history.append(val_cost)
            print(f"[에포크 {epoch:3d}] 검증 비용: {val_cost:.2f}m ({convert_to_km(val_cost):.2f}km)")
            
            # 최고 모델 저장
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'val_cost': val_cost,
                    'val_cost_km': convert_to_km(val_cost)
                }, BEST_MODEL)
                print(f"🌟 최고 모델 저장 @에포크{epoch} (검증 비용: {val_cost:.2f}m, {convert_to_km(val_cost):.2f}km)")
        else:
            print(f"[에포크 {epoch:3d}] 경고: 유효한 검증 결과가 없습니다")
        
        # 에포크 진행 그래프 (10 에포크마다)
        if epoch % 10 == 0 and len(train_costs_history) > 0:
            print(f"\n===== 학습 진행 상황 (에포크 {epoch}) =====")
            print(f"최근 10 에포크 평균 학습 비용: {np.mean(train_costs_history[-10:]):.2f}m ({convert_to_km(np.mean(train_costs_history[-10:])):.2f}km)")
            if len(val_costs_history) > 0:
                print(f"최근 10 에포크 평균 검증 비용: {np.mean(val_costs_history[-10:]):.2f}m ({convert_to_km(np.mean(val_costs_history[-10:])):.2f}km)")
                print(f"현재 최고 검증 비용: {best_val_cost:.2f}m ({convert_to_km(best_val_cost):.2f}km)")
            print("=" * 40)
        
        # 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_cost': best_val_cost,
            'best_val_cost_km': convert_to_km(best_val_cost),
            'train_costs_history': train_costs_history,
            'val_costs_history': val_costs_history
        }, CHECKPOINT)
        
        # 주기적 체크포인트 저장
        if epoch % 10 == 0:
            cp_dir = os.path.dirname(CHECKPOINT)
            if cp_dir:
                cp_path = os.path.join(cp_dir, f"checkpoint_ep{epoch}.pth")
            else:
                cp_path = f"checkpoint_ep{epoch}.pth"
                
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_cost': best_val_cost,
                'best_val_cost_km': convert_to_km(best_val_cost),
                'train_costs_history': train_costs_history,
                'val_costs_history': val_costs_history
            }, cp_path)
            print(f"📝 에포크 {epoch} 체크포인트 저장")
    
    # 학습 완료
    print("\n" + "="*50)
    print(f"학습 완료. 최고 검증 비용: {best_val_cost:.2f}m ({convert_to_km(best_val_cost):.2f}km)")
    
    # 최종 모델 저장
    torch.save({
        'model': model.state_dict(),
        'best_val_cost': best_val_cost,
        'best_val_cost_km': convert_to_km(best_val_cost),
        'final_epoch': EPOCHS,
        'train_costs_history': train_costs_history,
        'val_costs_history': val_costs_history
    }, OUT_MODEL)
    
    print(f"✅ 최종 모델 저장 완료: {OUT_MODEL}")
    print(f"✅ 최고 모델 저장 완료: {BEST_MODEL}")

if __name__ == "__main__":
    main()