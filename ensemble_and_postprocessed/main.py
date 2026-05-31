import os
import sys
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2

from data import build_loaders


# ── 재현성 / 로깅 유틸 ─────────────────────────────────────────
def set_seed(seed: int):
    """random / numpy / torch 전역 RNG 를 동일 seed 로 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """콘솔과 파일에 동시에 기록하는 로거를 구성합니다.

    같은 이름으로 재호출되어도 핸들러가 중복되지 않도록 매번 초기화합니다.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    fmt = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


# ── 모델 ──────────────────────────────────────────────────────
class UNet(nn.Module):
    """UNet++ with EfficientNet-B4 encoder (ImageNet pretrained).
    decoder_dropout=0.3 으로 overfitting 억제.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            decoder_dropout=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── 손실 함수 ─────────────────────────────────────────────────
class SegmentationLoss(nn.Module):
    """Combo Loss: DiceLoss(0.5) + CrossEntropyLoss(0.5).
    얇은 boundary 윤곽은 픽셀 단위 CE에 더 민감하므로 CE 비중을 0.5로 유지.
    CE weight=[1,1,2.0]로 boundary 클래스를 강조해 macro mIoU 병목을 완화.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        weight    = torch.tensor([1.0, 1.0, 2.0], device=device)
        self.ce   = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : (B, C, H, W)  targets : (B, H, W) LongTensor
        return 0.5 * self.dice(logits, targets) + 0.5 * self.ce(logits, targets)


# ── Trainer ───────────────────────────────────────────────────
class Trainer:
    """AMP + gradient clipping 이 적용된 train / validate 루프 래퍼."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device    = device
        self.scaler    = torch.cuda.amp.GradScaler()

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = self.model(images)          # (B, C, H, W)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader, num_classes: int = 3):
        """배치 평균 loss 와 macro-averaged mIoU 를 반환합니다."""
        self.model.eval()
        total_loss = 0.0
        # 클래스별 intersection / union 누적
        inter = torch.zeros(num_classes, device=self.device)
        union = torch.zeros(num_classes, device=self.device)

        for images, masks in loader:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            with torch.cuda.amp.autocast():
                logits = self.model(images)
                loss   = self.criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)          # (B, H, W)
            for c in range(num_classes):
                pred_c = preds == c
                mask_c = masks == c
                inter[c] += (pred_c & mask_c).sum()
                union[c] += (pred_c | mask_c).sum()

        iou_per_class = (inter / (union + 1e-6))
        miou = iou_per_class.mean().item()
        return total_loss / len(loader), miou, iou_per_class.cpu().tolist()


# ── 추론 ──────────────────────────────────────────────────────
# 테스트 이미지에 적용할 전처리 (augmentation 없음)
_infer_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 멀티스케일 TTA에 사용할 입력 해상도 (학습 해상도 384 + 확대 스케일 512)
_TTA_SCALES = (384, 512)


@torch.no_grad()
def predict_ensemble(models, img_path: Path, device: torch.device) -> np.ndarray:
    """
    여러 모델의 softmax 확률을 평균해 원본 해상도의 uint8 mask (값: 0,1,2) 를 반환합니다.

    - 멀티스케일 TTA: {384, 512} 각 스케일에서 (원본 + 수평 반전) softmax 확률 추론
    - 모델별 / 스케일별 확률을 모두 원본 H×W 로 bilinear 업샘플 후 누적 평균
    - 평균 확률에 argmax 하여 최종 class ID 산출 (앙상블)
    """
    img_pil  = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img_pil.size                        # PIL: (W, H)

    prob_accum = None                                    # (1, C, orig_h, orig_w) 누적 확률

    with torch.cuda.amp.autocast():
        for size in _TTA_SCALES:
            resized  = img_pil.resize((size, size), Image.BILINEAR)
            inp      = _infer_transform(resized).unsqueeze(0).to(device)  # (1,3,size,size)
            inp_flip = torch.flip(inp, dims=[-1])

            for model in models:
                prob_orig     = torch.softmax(model(inp), dim=1)
                prob_flip_raw = torch.softmax(model(inp_flip), dim=1)
                # 반전 공간의 확률을 원본 방향으로 되돌린 뒤 flip 쌍 평균
                probs         = (prob_orig + torch.flip(prob_flip_raw, dims=[-1])) / 2

                # 원본 해상도로 bilinear 업샘플하여 스케일/모델 간 정렬 후 누적
                probs_full = torch.nn.functional.interpolate(
                    probs, size=(orig_h, orig_w), mode='bilinear', align_corners=False
                )
                prob_accum = probs_full if prob_accum is None else prob_accum + probs_full

    # 스케일 수 × 모델 수 로 평균 (argmax 결과에는 영향 없지만 의미를 명확히 함)
    prob_accum = prob_accum / (len(_TTA_SCALES) * len(models))

    # argmax를 통해 최종 class ID (0: FG, 1: BG, 2: Boundary) 추출
    pred = prob_accum.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred                                          # (H, W), values in {0,1,2}


def run_inference(ckpt_paths, test_dir: str, pred_dir: str, device: torch.device):
    """여러 checkpoint 를 불러와 softmax 확률 평균 앙상블로 test 전체를 추론, pred_dir 에 .npy 저장."""
    models = []
    for ckpt_path in ckpt_paths:
        if not Path(ckpt_path).exists():
            print(f"[경고] checkpoint 없음, 건너뜀: {ckpt_path}")
            continue
        model = UNet(in_channels=3, num_classes=3).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt['epoch']}, "
              f"val_loss={ckpt['val_loss']:.4f}, val_mIoU={ckpt.get('val_miou', float('nan')):.4f})")

    if not models:
        raise RuntimeError("추론에 사용할 수 있는 checkpoint 가 하나도 없습니다.")

    print(f"앙상블 모델 수: {len(models)}")

    pred_path = Path(pred_dir)
    pred_path.mkdir(parents=True, exist_ok=True)

    test_images = sorted(Path(test_dir).glob('*.jpg'))
    print(f"Predicting {len(test_images)} test images → {pred_dir}/")

    for img_path in test_images:
        mask = predict_ensemble(models, img_path, device)
        out_file = pred_path / f"{img_path.stem}.npy"
        np.save(out_file, mask)

    print("Done. Unique label values in last prediction:", np.unique(mask))


# ── 단일 시드 학습 ────────────────────────────────────────────
def train_one_seed(
    seed: int,
    ckpt_path: str,
    log_file: str,
    device: torch.device,
    *,
    num_epochs: int = 40,
    learning_rate: float = 2e-4,
    num_classes: int = 3,
    warmup_epochs: int = 5,
    es_patience: int = 10,
    batch_size: int = 16,
) -> float:
    """단일 seed 로 학습하고 best checkpoint 를 ckpt_path 에 저장합니다.

    학습 도중 KeyboardInterrupt(조기종료 신호)가 발생하면 그때까지 저장된
    best checkpoint 를 그대로 유지한 채 함수를 정상 반환하여, 호출부에서
    다음 seed 학습으로 이어갈 수 있게 합니다.
    """
    logger = setup_logger(f"seed{seed}", log_file)
    set_seed(seed)

    train_loader, val_loader, n_train, n_val = build_loaders(seed=seed, batch_size=batch_size)
    logger.info(f"==== Seed {seed} 학습 시작 (log: {log_file}) ====")
    logger.info(f"Train set size: {n_train}  Val set size: {n_val}")
    logger.info(f"Train batches : {len(train_loader)}  Val batches : {len(val_loader)}")

    model     = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = SegmentationLoss(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # linear warmup → cosine decay
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    trainer       = Trainer(model, criterion, optimizer, device)
    best_val_miou = 0.0
    es_counter    = 0
    interrupted   = False

    try:
        for epoch in range(1, num_epochs + 1):
            train_loss                    = trainer.train_one_epoch(train_loader)
            val_loss, val_miou, val_ious  = trainer.validate(val_loader, num_classes=num_classes)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(f"[Seed {seed}][Epoch {epoch:02d}/{num_epochs}] "
                        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                        f"val_mIoU={val_miou:.4f}  lr={current_lr:.2e}")
            # 클래스별 IoU (0: FG, 1: BG, 2: Boundary) — boundary 병목 진단용
            logger.info(f"           IoU  FG={val_ious[0]:.4f}  BG={val_ious[1]:.4f}  "
                        f"Boundary={val_ious[2]:.4f}")

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                es_counter    = 0
                torch.save({
                    'epoch':                epoch,
                    'seed':                 seed,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss':             val_loss,
                    'val_miou':             best_val_miou,
                }, ckpt_path)
                logger.info(f"  → Best checkpoint saved (val_mIoU={best_val_miou:.4f}) → {ckpt_path}")
            else:
                es_counter += 1
                if es_counter >= es_patience:
                    logger.info(f"  → Early stopping triggered (no improvement for {es_patience} epochs)")
                    break

    except KeyboardInterrupt:
        interrupted = True
        if best_val_miou > 0.0:
            logger.info(f"\n[조기종료 신호 감지] Seed {seed} 학습을 중단합니다. "
                        f"best checkpoint(val_mIoU={best_val_miou:.4f})는 {ckpt_path} 에 유지됩니다.")
        else:
            logger.info(f"\n[조기종료 신호 감지] Seed {seed} 학습을 중단합니다. "
                        f"아직 저장된 checkpoint 가 없어 {ckpt_path} 가 생성되지 않았습니다 "
                        f"(최소 1 epoch 검증 완료 후 중단을 권장).")

    status = "중단됨" if interrupted else "정상 완료"
    logger.info(f"==== Seed {seed} 학습 {status}. Best val_mIoU={best_val_miou:.4f} → {ckpt_path} ====\n")

    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)

    return best_val_miou


# ── main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train multi-seed U-Net ensemble or run inference")
    parser.add_argument('--infer', action='store_true',
                        help='학습 없이 앙상블 추론만 실행 (checkpoint 들 필요)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123],
                        help='앙상블 멤버를 학습할 seed 목록 (기본: 42 123)')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--ckpt_prefix', default='best_model_seed',
                        help='시드별 checkpoint 파일명 prefix → {prefix}{seed}.pth')
    parser.add_argument('--ckpts', type=str, nargs='+', default=None,
                        help='추론에 사용할 checkpoint 목록 (미지정 시 seeds 로부터 자동 생성)')
    parser.add_argument('--test_dir',  default='test_images')
    parser.add_argument('--pred_dir',  default='predictions')
    parser.add_argument('--sample',  default='sample_submission.csv',
                        help='make_submission.py 에 전달할 sample CSV 경로 (안내용)')
    parser.add_argument('--out_csv', default='submission.csv',
                        help='make_submission.py 에 전달할 출력 CSV 경로 (안내용)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 추론에 사용할 checkpoint 경로 (명시 안하면 seeds 로부터 생성)
    ckpt_paths = args.ckpts if args.ckpts else [f"{args.ckpt_prefix}{s}.pth" for s in args.seeds]

    # ── 학습 모드: 시드별 순차 학습 ───────────────────────────
    if not args.infer:
        for seed in args.seeds:
            ckpt_path = f"{args.ckpt_prefix}{seed}.pth"
            log_file  = f"train_seed{seed}.log"
            # 조기종료 신호(KeyboardInterrupt)는 train_one_seed 내부에서 처리되어
            # best checkpoint 를 유지한 채 반환 → 자동으로 다음 seed 학습으로 연결됩니다.
            train_one_seed(
                seed, ckpt_path, log_file, device,
                num_epochs=args.epochs,
            )
        print("\n==== 모든 seed 학습 종료 → 앙상블 추론으로 진행 ====")

    # ── 앙상블 추론 + 제출 ─────────────────────────────────────
    run_inference(ckpt_paths, args.test_dir, args.pred_dir, device)

    import subprocess
    subprocess.run([
        sys.executable, "make_submission.py",
        "--pred_dir", args.pred_dir,
        "--sample",   args.sample,
        "--out",      args.out_csv,
    ], check=True)


if __name__ == "__main__":
    main()
