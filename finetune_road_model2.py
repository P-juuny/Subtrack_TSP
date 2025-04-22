import os
import random
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

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
parser.add_argument('--lr', type=float, default=1e-5)  # 학습률 감소
parser.add_argument('--weight_decay', type=float, default=1e-5)
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
WEIGHT_DECAY = args.weight_decay
WARMUP_EPOCHS = args.warmup_epochs
MAX_GRAD_NORM = args.clip_grad
ETA_MIN = 1e-7
FALLBACK_DIST = 10000.0  # 최대 거리 클리핑 값 (미터 단위)
VAL_SPLIT = 0.1
SEED = 42

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

def convert_to_km(meters):
    """미터 단위 비용을 킬로미터로 변환"""
    return meters / 1000.0

# ----------------------------- #
# Learning Rate 스케줄러 (Warmup + Cosine)
# ----------------------------- #
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0):
        self.warmup_epochs = warmup_epochs
        self.cosine = CosineAnnealingLR(optimizer, T_max=total_epochs-warmup_epochs, eta_min=eta_min)
        self.optimizer = optimizer
        self.init_lr = optimizer.param_groups[0]['lr']
        self.eta_min = eta_min
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.eta_min + factor * (self.init_lr - self.eta_min)
        else:
            self.cosine.step()

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

                # 전체 데이터셋 거리 통계 (100개 샘플만)  ⚡️
        all_dists = []
        for idx in range(min(100, n_total)):
            d = full_dataset[idx]['dist']
            all_dists.extend(d.flatten())
        all_arr = np.array(all_dists)
        all_arr = all_arr[(~np.isnan(all_arr)) & (~np.isinf(all_arr)) & (all_arr > 0)]
        print("거리 통계 (100개 샘플, 미터):",
              f"평균={all_arr.mean():.2f}, 중앙값={np.median(all_arr):.2f},",
              f"최소={all_arr.min():.2f}, 최대={all_arr.max():.2f}")

        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=BATCH_SIZE
        )
        print(f"데이터 로드 완료: 총 {n_total}개 (학습 {len(train_idx)}, 검증 {len(val_idx)})")
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return

    # 4. 옵티마이저 & 스케줄러
    optimizer = Adam(
        model.parameters(),
        lr=LR_INIT,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        eta_min=ETA_MIN
    )

    # 5. 모델 초기화/로드
    start_epoch = 1
    best_val_cost = float('inf')
    train_history, val_history = [], []

    if args.resume and os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.current_epoch = ckpt.get('epoch', 1)
        best_val_cost = ckpt.get('best_val_cost', best_val_cost)
        train_history = ckpt.get('train_history', [])
        val_history = ckpt.get('val_history', [])
        start_epoch = ckpt.get('epoch', 1) + 1
        print(f"체크포인트 로드: epoch={start_epoch}")
    elif os.path.exists(PRETRAIN):
        pt = torch.load(PRETRAIN, map_location=DEVICE)
        model.load_state_dict(pt.get('model', pt))
        print("사전학습 가중치 로드 완료")

    # EMA baseline 설정
    baseline_ema = None
    ema_alpha = 0.9

    # 6. 학습 및 검증 루프
    for epoch in range(start_epoch, EPOCHS+1):
        # 학습
        model.train()
        total_loss, total_cost = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"학습 {epoch}/{EPOCHS}", dynamic_ncols=True, mininterval=0.5, ascii=True, smoothing=0):
            loc = batch['loc'].to(DEVICE)
            dist = batch['dist'].to(DEVICE)
            dist = torch.where(
                torch.isnan(dist) | torch.isinf(dist) | (dist<0),
                torch.full_like(dist, FALLBACK_DIST),
                dist
            )
            model.set_decode_type('sampling')
            cost, ll = model({'loc':loc, 'dist':dist})
            # 단위 변환 및 REINFORCE
            B, N, _ = dist.size()
            cost_km = cost / 1000.0
            unit_cost = cost_km / N
            if baseline_ema is None:
                baseline_ema = unit_cost.mean().item()
            baseline_ema = ema_alpha * baseline_ema + (1-ema_alpha) * unit_cost.mean().item()
            adv = (unit_cost - baseline_ema)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
            loss = (adv * ll).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item()
            total_cost += unit_cost.mean().item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_cost = total_cost / len(train_loader)
        train_history.append(avg_cost)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}, cost={avg_cost*1000:.2f}m")

        # 검증
        model.eval()
        val_costs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"검증 {epoch}/{EPOCHS}", dynamic_ncols=True, mininterval=0.5, ascii=True, smoothing=0):
                loc = batch['loc'].to(DEVICE)
                dist = batch['dist'].to(DEVICE)
                dist = torch.where(
                    torch.isnan(dist) | torch.isinf(dist) | (dist<0),
                    torch.full_like(dist, FALLBACK_DIST),
                    dist
                )
                model.set_decode_type('greedy')
                cost, _, _ = model({'loc':loc, 'dist':dist}, return_pi=True)
                val_costs.append(cost.mean().item())
        val_arr = np.array(val_costs) / 1000.0
        val_mean_km = val_arr.mean()
        val_history.append(val_mean_km)
        print(f"[Epoch {epoch}] 검증 비용: {val_mean_km*1000:.2f}m ({val_mean_km:.2f}km)")

        # 최고 모델 저장
        if val_mean_km < best_val_cost:
            best_val_cost = val_mean_km
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_val_cost': best_val_cost,
                'train_history': train_history,
                'val_history': val_history
            }, BEST_MODEL)
            print(f"🌟 최고 모델 저장 @epoch{epoch} (Val {best_val_cost*1000:.2f}m)")

        # 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_cost': best_val_cost,
            'train_history': train_history,
            'val_history': val_history
        }, CHECKPOINT)

    print(f"학습 완료. 최고 검증 비용: {best_val_cost*1000:.2f}m ({best_val_cost:.2f}km)")
    torch.save({'model': model.state_dict(), 'best_val_cost': best_val_cost}, OUT_MODEL)
    print(f"✅ 최종 모델 저장: {OUT_MODEL}")

if __name__ == "__main__":
    main()
