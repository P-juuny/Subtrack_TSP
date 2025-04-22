import os
import random
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.cuda.amp as amp
from tqdm import tqdm

from nets.attention_model import AttentionModel
from define_tsp_road import TSPRoad, RoadDataset

# ----------------------------- #
# 파라미터 정의
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
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--clip_grad', type=float, default=1.0)
parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
args = parser.parse_args()

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
VAL_SPLIT    = 0.1

# ----------------------------- #
# 재현성 설정
# ----------------------------- #
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------- #
# 폴더 생성 헬퍼
# ----------------------------- #
def create_dir_if_needed(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ----------------------------- #
# Warmup + Cosine LR 스케줄러
# ----------------------------- #
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0):
        self.warmup_epochs = warmup_epochs
        self.cosine = CosineAnnealingLR(optimizer,
                                        T_max=total_epochs-warmup_epochs,
                                        eta_min=eta_min)
        self.optimizer = optimizer
        self.init_lr   = optimizer.param_groups[0]['lr']
        self.eta_min   = eta_min
        self.epoch     = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            factor = self.epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.eta_min + factor*(self.init_lr - self.eta_min)
        else:
            self.cosine.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# ----------------------------- #
# 모델 로드 함수
# ----------------------------- #
def load_model(path, device):
    problem = TSPRoad()  # 지도 기반 거리 계산 내장
    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        n_encode_layers=3,
        n_heads=8,
        tanh_clipping=10.0,
        normalization="batch",
        problem=problem
    ).to(device)
    model.set_decode_type('greedy')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model

# ----------------------------- #
# OR-Tools 비교용 (optional)
# ----------------------------- #
def solve_with_ortools(dist_np, time_limit=1):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = dist_np.shape[0]
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def cb(a, b, dm=dist_np, m=mgr):
        return int(dm[m.IndexToNode(a)][m.IndexToNode(b)])
    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = time_limit
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    return sol.ObjectiveValue() if sol else float('nan')

# ----------------------------- #
# 메인
# ----------------------------- #
def main():
    # 경로 준비
    create_dir_if_needed(args.out_model)
    create_dir_if_needed(args.best_model)
    create_dir_if_needed(args.checkpoint)

    # 모델·옵티마이저·스케줄러·스케일러
    model     = load_model(args.pretrain, DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupScheduler(optimizer, args.warmup_epochs, args.epochs, eta_min=1e-7)
    scaler    = amp.GradScaler()

    # 데이터셋 로드
    full_ds = RoadDataset(args.data_pkl)
    n_total = len(full_ds)
    idxs    = list(range(n_total))
    random.shuffle(idxs)
    split   = int(n_total * VAL_SPLIT)
    train_idx, val_idx = idxs[split:], idxs[:split]

    train_loader = DataLoader(Subset(full_ds, train_idx),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(full_ds, val_idx),
                              batch_size=args.batch_size)

    # 체크포인트 또는 사전학습 가중치 로드
    start_epoch    = 1
    best_val_cost  = float('inf')
    no_improve_cnt = 0
    if args.resume and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.epoch   = ckpt.get('epoch', 1)
        best_val_cost     = ckpt.get('best_val_cost', best_val_cost)
        start_epoch       = scheduler.epoch + 1
        print(f"▶ Resuming from epoch {start_epoch}")
    else:
        print("▶ Training from scratch or loading pretrained weights above")

    # 학습 루프
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, total_reward = 0.0, 0.0

        for batch in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ascii=True):
            locs = batch['loc'].to(DEVICE)    # [B, N, 2], lat/lon 그대로
            dists = batch['dist'].to(DEVICE)  # [B, N, N], 도로 거리(m)
            # 결측/음수 처리
            dists = torch.where(
                torch.isnan(dists) | torch.isinf(dists) | (dists < 0),
                torch.full_like(dists, 1e6),
                dists
            )

            model.set_decode_type('sampling')
            optimizer.zero_grad()
            with amp.autocast():
                cost, logp = model({'loc': locs, 'dist': dists})
                # cost: [B] in meters (TSPRoad.get_costs가 km→m 복원)
                B, N, _ = dists.size()
                reward = - cost / N    # 노드당 단위 보상 (낮은 cost일수록 보상 높음)
                # baseline EMA
                if epoch == start_epoch and total_reward == 0:
                    baseline = reward.mean().item()
                baseline = 0.9 * baseline + 0.1 * reward.mean().item()
                advantage = reward - baseline
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
                loss = -(advantage * logp).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_reward += reward.mean().item()

        scheduler.step()
        avg_train_rt = total_reward / len(train_loader)

        # 검증
        model.eval()
        val_costs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", ascii=True):
                locs = batch['loc'].to(DEVICE)
                dists = batch['dist'].to(DEVICE)
                dists = torch.where(
                    torch.isnan(dists) | torch.isinf(dists) | (dists < 0),
                    torch.full_like(dists, 1e6),
                    dists
                )
                model.set_decode_type('greedy')
                c, _, _ = model({'loc': locs, 'dist': dists}, return_pi=True)
                val_costs.append(c.mean().item())

        val_mean = np.mean(val_costs)  # meters per tour
        print(f"[Epoch {epoch}] Train reward={avg_train_rt:.1f} | Val cost={val_mean:.1f}m | LR={scheduler.get_lr():.2e}")

        # 베스트 저장 및 Early Stopping
        if val_mean < best_val_cost:
            best_val_cost  = val_mean
            no_improve_cnt = 0
            torch.save({'model': model.state_dict(), 'best_val_cost': best_val_cost, 'epoch': epoch},
                       args.best_model)
            print(f"  ☑️ New best saved @epoch{epoch} (Val {best_val_cost:.1f}m)")
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= args.patience:
                print(f"🔔 Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        # 체크포인트
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_cost': best_val_cost,
                    'epoch': epoch},
                   args.checkpoint)

    # 최종 모델 저장
    torch.save({'model': model.state_dict(), 'best_val_cost': best_val_cost},
               args.out_model)
    print(f"✅ Training done. Best val cost: {best_val_cost:.1f}m")

if __name__ == "__main__":
    main()
