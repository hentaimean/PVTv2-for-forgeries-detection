import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os
from datetime import datetime

from splits.splits import split_dataset_by_groups
from model.pvtv2 import PVTv2B5ForForgerySegmentation
from tools.dataclass import *
from tools.loss import BinaryCrossEntropyLoss, DiceLoss, FocalLoss
from tools.optimizer import create_optimizer
from tools.scheduler import create_scheduler
from tools.metrics import BinarySegmentationMetrics
from tools.visualize import *

# --- Dataset ---
IMAGE_DIR = 'images'
MASKS_DIR = 'masks'
SPLIT_PATH = "splits/grouped_indices.pt"

# --- DataLoader ---
BATCH_SIZE = 2
NUM_WORKERS = 4
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
PIN_MEMORY = True  # ускоряет передачу на GPU
DROP_LAST = True  # для стабильности batch-norm при малых батчах

# --- Configuration ---
MAX_ITERS = 320000
VAL_INTERVAL = 10
SAVE_INTERVAL = 5000
LOG_INTERVAL = 50
VISUALIZE_EVERY = 1

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass

def main():

    # Обучающий датасет с аугментациями и foreground-aware кропами
    train_dataset_full = ForgerySegmentationDataset(
        images_dir=IMAGE_DIR,
        masks_dir=MASKS_DIR,
        transform=get_training_augmentation(),
        fg_crop_prob=0.7,           # ← кропы с подделками
        crop_size=(512, 512),
        use_albumentations=True
    )

    # Валидационный датасет БЕЗ аугментаций, НО С кропами (фиксированный размер)
    eval_dataset_full = ForgerySegmentationDataset(
        images_dir=IMAGE_DIR,
        masks_dir=MASKS_DIR,
        transform=get_validation_augmentation(),
        fg_crop_prob=0.0,
        crop_size=(512, 512),
        use_albumentations=True
    )

    # # Разбитие датасета
    #
    # # Получаем индексы один раз (на основе имён файлов — одинаковы в обоих датасетах)
    # train_idx, val_idx, test_idx = split_dataset_by_groups(
    #     dataset=train_dataset_full,
    #     save_path=SPLIT_PATH
    # )

    # Загружаем сохранённые индексы
    split_data = torch.load(SPLIT_PATH)

    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']
    test_indices = split_data['test_indices']
    seed = split_data.get('seed', 'unknown')

    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Test:  {len(test_indices)}\n")

    # Создаём подвыборки
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(eval_dataset_full, val_indices)
    test_dataset = Subset(eval_dataset_full, test_indices)

    model = PVTv2B5ForForgerySegmentation(img_size=512)
    model = model.float()

    # Загружаем чекпоинт
    checkpoint_path = "model/pvt_v2_b5.pth"
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Удаляем классификационную голову (не нужна для сегментации)
    keys_to_remove = [k for k in state_dict.keys() if k.startswith('head')]
    for k in keys_to_remove:
        del state_dict[k]

    # Загружаем в backbone
    missing_keys, unexpected_keys = model.backbone.load_state_dict(state_dict, strict=False)

    # Проверяем, что всё ок
    if len(unexpected_keys) == 0 and all('head' not in k for k in missing_keys):
        print("Предобученные веса PVTv2-B5 успешно загружены!\n")
        if missing_keys:
            print(f"Не загружены ключи (ожидаемо для head): {missing_keys}")
    else:
        print("Ошибка при загрузке весов:")
        print("Unexpected keys:", unexpected_keys)
        print("Missing keys:", missing_keys)

    bce_loss_fn = BinaryCrossEntropyLoss(loss_weight=1.0, avg_non_ignore=True)
    dice_loss_fn = DiceLoss(loss_weight=1.0, use_sigmoid=True)
    #focal_loss_fn = FocalLoss(loss_weight=1.0)  # опционально

    optimizer = create_optimizer(
        model,
        lr=6e-5,
        weight_decay=0.01,
        head_lr_mult=10.0
    )

    # Планировщик
    scheduler = create_scheduler(
        optimizer,
        warmup_iters=1500,
        total_iters=320000,
        min_lr=0.0,
        power=1.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=SHUFFLE_VAL,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False  # на валидации лучше сохранять все примеры
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/forgery_pvtv2_b5_{timestamp}"
    checkpoint_dir = f"{run_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_iter = iter(train_loader)
    metrics_val = BinarySegmentationMetrics(threshold=0.5)

    best_iou = 0.0

    pbar = tqdm(range(1, MAX_ITERS + 1), desc="Training", mininterval=1.0)

    for iter_idx in pbar:

        # --- Обучение ---
        model.train()
        optimizer.zero_grad()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # МАСКА: float32 + канал → [B, 1, H, W]
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True).float().unsqueeze(1)

        assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
        assert masks.dtype == torch.float32, f"Expected float32, got {masks.dtype}"

        pred = model(images)

        # Считаем каждую потерю отдельно
        loss_bce = bce_loss_fn(pred, masks)
        loss_dice = dice_loss_fn(pred, masks)
        #loss_focal = focal_loss_fn(pred, masks)
        total_loss = loss_bce + loss_dice

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"NaN/Inf in loss at iter {iter_idx}")
            print(f"  BCE={loss_bce.item()}, Dice={loss_dice.item()}")
            print(f"  Pred min/max: {pred.min().item():.4f} / {pred.max().item():.4f}")
            print(f"  Target unique: {torch.unique(masks)}")
            raise ValueError("Stopping due to NaN loss")

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Логирование в TensorBoard ---
        if iter_idx % LOG_INTERVAL == 0:
            writer.add_scalar('Loss/BCE', loss_bce.item(), iter_idx)
            writer.add_scalar('Loss/Dice', loss_dice.item(), iter_idx)
            #writer.add_scalar('Loss/Focal', loss_focal.item(), iter_idx)
            writer.add_scalar('Loss/Total', total_loss.item(), iter_idx)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], iter_idx)

        # --- Обновление tqdm (каждые 100 итераций) ---
        if iter_idx % 100 == 0:
            pbar.set_postfix({
                'BCE': f"{loss_bce.item():.3f}",
                'Dice': f"{loss_dice.item():.3f}",
                #'Focal': f"{loss_focal.item():.3f}",
                'Total': f"{total_loss.item():.3f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        # --- Валидация ---
        if iter_idx % VAL_INTERVAL == 0:
            val_metrics, val_loss = validate_epoch(
                model,
                val_dataset,
                metrics_val,
                device,
                bce_loss_fn,
                dice_loss_fn,
                writer=writer,
                global_step=iter_idx,
                val_sample_size=1000,
                seed=42
            )

            for name, value in val_metrics.items():
                writer.add_scalar(f'Val/{name}', value, iter_idx)

            print(f"\n[Iter {iter_idx}] Val — " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))

            # --- Сохранение ЛУЧШЕЙ модели ---
            current_iou = val_metrics['IoU_forgery']
            if current_iou > best_iou:
                best_iou = current_iou
                best_path = f"{checkpoint_dir}/best_model_iou_{current_iou:.4f}_iter_{iter_idx}.pth"
                torch.save(model.state_dict(), best_path)
                print(f"Новая лучшая модель сохранена: {os.path.basename(best_path)}")

        # --- Сохранение ПОСЛЕДНЕЙ модели (каждые 10k) ---
        if iter_idx % SAVE_INTERVAL == 0:
            last_path = f"{checkpoint_dir}/last_model_iter_{iter_idx}.pth"
            torch.save(model.state_dict(), last_path)

    # --- Финальное сохранение последней модели ---
    final_path = f"{checkpoint_dir}/last_model_final.pth"
    torch.save(model.state_dict(), final_path)
    writer.close()

if __name__ == '__main__':
    main()
