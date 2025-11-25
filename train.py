# train.py — обучение модели для сегментации подделок на изображении

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
from tools.visualize import validate_epoch, visualize_prediction
from tools.forgery_balancer import ForgeryBalancedBatchSampler

# ==============================
# КОНСТАНТЫ ПРОЕКТА
# ==============================

# Пути к данным
IMAGE_DIR = 'images'
MASKS_DIR = 'masks'
SPLIT_PATH = "splits/grouped_indices.pt"
CHECKPOINT_PATH = "model/pvt_v2_b5.pth"

# Параметры DataLoader
BATCH_SIZE = 1
NUM_WORKERS = 0
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
PIN_MEMORY = False  # ускоряет передачу данных на GPU
DROP_LAST = True    # для стабильности batch-norm при малых батчах
SEED = 42

# Параметры обучения
MAX_ITERS = 320_000
VAL_INTERVAL = 5000      # интервал валидации (итераций)
VAL_SAMPLE_SIZE = 4000    # размер валидационной выборки
SAVE_INTERVAL = 5000     # интервал сохранения чекпоинтов
LOG_INTERVAL = 50         # интервал логирования в TensorBoard

# Параметры потерь
BCE_POS_WEIGHT = 10.0
BCE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 1.0
FOCAL_GAMMA = 2.0
FOCAL_LOSS_WEIGHT = 1.0

# Параметры оптимизатора
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
HEAD_LR_MULT = 10.0

# Параметры планировщика
WARMUP_ITERS = 3000
MIN_LR = 0.0
SCHEDULER_POWER = 1.0

# Параметры визуализации
VISUALIZE_TRAIN_INTERVAL = 1000  # каждые N итераций — визуализация в обучении
FG_CROP_PROB = 1.0   # вероятность кропа с подделкой

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


def main():
    """
    Основная функция обучения модели с периодической валидацией и логированием.
    Автоматически создаёт разбиение на выборки, если оно ещё не сохранено.
    """
    # --- Инициализация датасетов ---

    # Обучающий датасет: с аугментациями и кропами, содержащими подделки
    train_dataset_full = ForgerySegmentationDataset(
        images_dir=IMAGE_DIR,
        masks_dir=MASKS_DIR,
        transform=get_training_augmentation(),
        fg_crop_prob=FG_CROP_PROB,
        crop_size=(512, 512),
        use_albumentations=True
    )

    # Валидационный датасет: без аугментаций, с фиксированным кропом
    eval_dataset_full = ForgerySegmentationDataset(
        images_dir=IMAGE_DIR,
        masks_dir=MASKS_DIR,
        transform=get_validation_augmentation(),
        fg_crop_prob=0.0,
        crop_size=(512, 512),
        use_albumentations=True
    )

    # --- Разбиение датасета или загрузка существующего ---

    if not os.path.exists(SPLIT_PATH):
        print(f"Файл разбиения не найден: {SPLIT_PATH}. Генерация нового...")
        train_idx, val_idx, test_idx = split_dataset_by_groups(
            dataset=train_dataset_full,
            save_path=SPLIT_PATH,
            seed=SEED
        )
        split_data = torch.load(SPLIT_PATH)
    else:
        print(f"Загрузка существующего разбиения: {SPLIT_PATH}")
        split_data = torch.load(SPLIT_PATH)

    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']
    test_indices = split_data['test_indices']
    seed = split_data.get('seed', 'unknown')

    print(f"Размеры выборок (seed={seed}):")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Test:  {len(test_indices)}\n")

    # Создание подвыборок с помощью Subset
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(eval_dataset_full, val_indices)
    test_dataset = Subset(eval_dataset_full, test_indices)

    # --- Инициализация модели ---

    model = PVTv2B5ForForgerySegmentation(img_size=512)
    model = model.float()

    # Загрузка предобученных весов PVTv2-B5 (без классификационной головы)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # Удаляем веса head-слоёв (они не нужны для сегментации)
    keys_to_remove = [k for k in state_dict.keys() if k.startswith('head')]
    for k in keys_to_remove:
        del state_dict[k]

    # Загружаем веса в backbone
    missing_keys, unexpected_keys = model.backbone.load_state_dict(state_dict, strict=False)

    # Проверка корректности загрузки
    if len(unexpected_keys) == 0 and all('head' not in k for k in missing_keys):
        print("Предобученные веса PVTv2-B5 успешно загружены!\n")
        if missing_keys:
            print(f"Не загружены (ожидаемо для head): {missing_keys}")
    else:
        print("Ошибка при загрузке весов:")
        print("Unexpected keys:", unexpected_keys)
        print("Missing keys:", missing_keys)

    # --- Инициализация функций потерь ---

    bce_loss_fn = BinaryCrossEntropyLoss(
        pos_weight=BCE_POS_WEIGHT,
        loss_weight=BCE_LOSS_WEIGHT
    )
    dice_loss_fn = DiceLoss(
        loss_weight=DICE_LOSS_WEIGHT,
        use_sigmoid=True
    )
    focal_loss_fn = FocalLoss(
        gamma=FOCAL_GAMMA,
        loss_weight=FOCAL_LOSS_WEIGHT,
        use_sigmoid=True
    )

    # --- Инициализация оптимизатора и планировщика ---

    optimizer = create_optimizer(
        model,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        head_lr_mult=HEAD_LR_MULT
    )

    scheduler = create_scheduler(
        optimizer,
        warmup_iters=WARMUP_ITERS,
        total_iters=MAX_ITERS,
        min_lr=MIN_LR,
        power=SCHEDULER_POWER
    )

    # # --- Активация ForgeryBalancedBatchSampler ---
    # train_sampler = ForgeryBalancedBatchSampler(
    #     full_dataset=train_dataset_full,
    #     allowed_indices=train_indices,
    #     batch_size=BATCH_SIZE,
    #     fg_ratio=0.5,  # 50% подделок в батче
    #     shuffle=SHUFFLE_TRAIN,
    #     seed=SEED
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=SHUFFLE_TRAIN,
        drop_last=DROP_LAST
    )

    # --- Настройка логирования и сохранения ---

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/forgery_pvtv2_b5_{timestamp}"
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # --- Подготовка к обучению ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_iter = iter(train_loader)
    metrics_val = BinarySegmentationMetrics(threshold=0.5)

    # --- Проверка модели ---
    print("Проверка модели перед обучением...")
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_image = sample_batch['image'].to(device)
        sample_pred = model(sample_image)
        print(f"Минимальный логит: {sample_pred.min().item():.4f}")
        print(f"Максимальный логит: {sample_pred.max().item():.4f}")
        print(f"Средний логит: {sample_pred.mean().item():.4f}")

        # Только backbone
        feats = model.backbone(sample_image)
        print("Backbone output shapes:")
        for i, f in enumerate(feats):
            print(f"  Level {i}: {f.shape}, mean={f.mean().item():.4f}, std={f.std().item():.4f}")

    best_iou = 0.0
    pbar = tqdm(range(1, MAX_ITERS + 1), desc="Training", mininterval=1.0)

    # --- Цикл обучения ---

    for iter_idx in pbar:

        # === Шаг обучения ===
        model.train()
        optimizer.zero_grad()

        # Получаем следующий батч (с циклическим повтором при необходимости)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Перенос данных на устройство и приведение маски к нужному формату
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True).float().unsqueeze(1)

        # Проверка типов данных
        assert images.dtype == torch.float32, f"Ожидался float32 для изображений, получен {images.dtype}"
        assert masks.dtype == torch.float32, f"Ожидался float32 для масок, получен {masks.dtype}"

        # Прямой проход
        pred = model(images)

        # Подсчёт потерь
        loss_bce = bce_loss_fn(pred, masks)
        loss_dice = dice_loss_fn(pred, masks)
        loss_focal = focal_loss_fn(pred, masks)
        total_loss = loss_bce + loss_dice + loss_focal

        # Проверка на NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Обнаружено NaN/Inf в итерации {iter_idx}")
            print(f"  BCE={loss_bce.item():.4f}, Dice={loss_dice.item():.4f}, Focal={loss_focal.item():.4f}")
            print(f"  Pred min/max: {pred.min().item():.4f} / {pred.max().item():.4f}")
            print(f"  Уникальные значения маски: {torch.unique(masks)}")
            raise ValueError("Остановка из-за NaN в функции потерь")

        # Обратное распространение и шаг оптимизатора
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # === Логирование в TensorBoard ===
        if iter_idx % LOG_INTERVAL == 0:
            writer.add_scalar('Loss/BCE', loss_bce.item(), iter_idx)
            writer.add_scalar('Loss/Dice', loss_dice.item(), iter_idx)
            writer.add_scalar('Loss/Focal', loss_focal.item(), iter_idx)
            writer.add_scalar('Loss/Total', total_loss.item(), iter_idx)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], iter_idx)

        # Обновление прогресс-бара
        if iter_idx % LOG_INTERVAL == 0:
            pbar.set_postfix({
                'BCE': f"{loss_bce.item():.3f}",
                'Dice': f"{loss_dice.item():.3f}",
                'Focal': f"{loss_focal.item():.3f}",
                'Total': f"{total_loss.item():.3f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        # === Визуализация во время обучения ===
        if iter_idx % VISUALIZE_TRAIN_INTERVAL == 0:
            with torch.no_grad():
                pred_viz = model(images[:1])
                fig = visualize_prediction(images[:1], masks[:1], pred_viz[:1])
                writer.add_figure('Train/Prediction', fig, global_step=iter_idx)

        # === Валидация ===
        if iter_idx % VAL_INTERVAL == 0:
            val_metrics, val_loss = validate_epoch(
                model=model,
                val_dataset=val_dataset,
                metrics_obj=metrics_val,
                device=device,
                bce_loss_fn=bce_loss_fn,
                dice_loss_fn=dice_loss_fn,
                focal_loss_fn=focal_loss_fn,
                writer=writer,
                global_step=iter_idx,
                val_sample_size=VAL_SAMPLE_SIZE,
                seed=SEED
            )

            # Логирование метрик
            for name, value in val_metrics.items():
                writer.add_scalar(f'Val/{name}', value, iter_idx)

            print(f"\n[Итерация {iter_idx}] Валидация — " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))

            # Сохранение лучшей модели по IoU
            current_iou = val_metrics['IoU_forgery']
            if current_iou > best_iou:
                best_iou = current_iou
                best_path = os.path.join(
                    checkpoint_dir,
                    f"best_model_iou_{current_iou:.4f}_iter_{iter_idx}.pth"
                )
                torch.save(model.state_dict(), best_path)
                print(f"Новая лучшая модель сохранена: {os.path.basename(best_path)}")

        # === Сохранение последней модели (периодически) ===
        if iter_idx % SAVE_INTERVAL == 0:
            last_path = os.path.join(checkpoint_dir, f"last_model_iter_{iter_idx}.pth")
            torch.save(model.state_dict(), last_path)

    # === Финальное сохранение ===
    final_path = os.path.join(checkpoint_dir, "last_model_final.pth")
    torch.save(model.state_dict(), final_path)
    writer.close()
    print(f"\nОбучение завершено. Финальная модель сохранена: {final_path}")


if __name__ == '__main__':
    main()