# tools/visualize.py — визуализация предсказаний и валидация модели

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# ==============================
# КОНСТАНТЫ ВИЗУАЛИЗАЦИИ И ВАЛИДАЦИИ
# ==============================

# Параметры денормализации (ImageNet)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, .225])

# Параметры визуализации
DEFAULT_VISUALIZATION_THRESHOLD = 0.5
DEFAULT_MAX_IMAGES_PER_FIGURE = 4
DEFAULT_MAX_VAL_VISUALIZATION_IMAGES = 16

# Параметры валидации
DEFAULT_VAL_SAMPLE_SIZE = 1000
DEFAULT_VISUALIZE_EVERY_NTH_VALIDATION = 1  # визуализировать каждую N-ю валидацию
VAL_INTERVAL = 5000


# ==============================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ==============================

def visualize_prediction(image, target_mask, pred_logits, threshold=DEFAULT_VISUALIZATION_THRESHOLD,
                         max_images=DEFAULT_MAX_IMAGES_PER_FIGURE):
    """
    Создаёт фигуру matplotlib для сравнения изображения, маски и предсказания.

    Аргументы:
        image: [B, 3, H, W] — нормализованные изображения (ImageNet)
        target_mask: [B, 1, H, W] — истинные бинарные маски (0/1)
        pred_logits: [B, 1, H, W] — сырые логиты модели
        threshold (float): порог для бинаризации предсказания
        max_images (int): максимальное число изображений в фигуре

    Возвращает:
        matplotlib.figure.Figure — готовая фигура для TensorBoard.
    """
    batch_size = min(image.shape[0], max_images)
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    # Обработка случая с одним изображением
    if batch_size == 1:
        axes = axes[np.newaxis, :]

    for i in range(batch_size):
        # Денормализация изображения
        img = image[i].cpu().numpy().transpose(1, 2, 0)
        img = IMAGE_STD * img + IMAGE_MEAN
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Изображение")
        axes[i, 0].axis('off')

        # Истинная маска
        mask_true = target_mask[i].cpu().numpy().squeeze()
        axes[i, 1].imshow(mask_true, cmap='gray')
        axes[i, 1].set_title("Истинная маска")
        axes[i, 1].axis('off')

        # Предсказанная маска
        pred = (pred_logits[i].sigmoid() > threshold).float().cpu().numpy().squeeze()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title("Предсказание")
        axes[i, 2].axis('off')

    plt.tight_layout()
    return fig


# ==============================
# ФУНКЦИЯ ВАЛИДАЦИИ
# ==============================

def validate_epoch(
        model,
        val_dataset,
        metrics_obj,
        device,
        bce_loss_fn,
        dice_loss_fn,
        focal_loss_fn,
        writer=None,
        global_step=0,
        val_sample_size=DEFAULT_VAL_SAMPLE_SIZE,
        seed=None,
        visualize_every=DEFAULT_VISUALIZE_EVERY_NTH_VALIDATION,
        max_images=DEFAULT_MAX_VAL_VISUALIZATION_IMAGES
):
    """
    Выполняет валидацию модели на подвыборке данных.

    Особенности:
      - Использует случайную подвыборку для ускорения.
      - Поддерживает визуализацию предсказаний в TensorBoard.
      - Сбрасывает и обновляет переданный объект метрик.

    Аргументы:
        model: обучаемая модель (в eval-режиме).
        val_dataset: валидационный датасет.
        metrics_obj: объект BinarySegmentationMetrics.
        device: устройство (CPU/GPU).
        bce_loss_fn, dice_loss_fn: функции потерь.
        writer: SummaryWriter для логирования (опционально).
        global_step: текущая итерация (для логирования).
        val_sample_size: размер случайной подвыборки.
        seed: для воспроизводимости.
        visualize_every: визуализировать каждую N-ю валидацию.
        max_images: макс. число изображений в визуализации.

    Возвращает:
        tuple: (словарь метрик, среднее значение total_loss)
    """
    model.eval()
    metrics_obj.reset()

    # --- 1. Выбор случайной подвыборки ---
    if seed is not None:
        random.seed(seed + global_step)

    total_size = len(val_dataset)
    if val_sample_size >= total_size:
        sampled_indices = list(range(total_size))
    else:
        sampled_indices = random.sample(range(total_size), val_sample_size)

    # --- 2. Создание DataLoader для подвыборки ---
    sampled_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=torch.utils.data.SubsetRandomSampler(sampled_indices),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )

    # --- 3. Проход по подвыборке ---
    total_bce = total_dice = total_loss = total_focal = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in sampled_loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True).float().unsqueeze(1)

            pred = model(images)

            loss_bce = bce_loss_fn(pred, masks)
            loss_dice = dice_loss_fn(pred, masks)
            loss_focal = focal_loss_fn(pred, masks)
            loss_total = loss_bce + loss_dice + loss_focal

            total_bce += loss_bce.item()
            total_dice += loss_dice.item()
            total_focal += loss_focal.item()
            total_loss += loss_total.item()
            num_batches += 1

            metrics_obj.update(pred, masks)

    # Усреднение
    avg_bce = total_bce / num_batches
    avg_dice = total_dice / num_batches
    avg_focal = total_focal / num_batches
    avg_loss = total_loss / num_batches
    metrics = metrics_obj.compute()

    # --- 4. Визуализация (если требуется) ---
    if writer is not None:
        current_val_idx = global_step // VAL_INTERVAL
        should_visualize = (
                current_val_idx % visualize_every == 0 or current_val_idx == 1
        )

        if should_visualize:
            # Сбор батчей для визуализации
            vis_batches = []
            for i, batch in enumerate(sampled_loader):
                if i >= max_images:
                    break
                vis_batches.append(batch)

            if vis_batches:
                images = torch.cat([b['image'][:1] for b in vis_batches], dim=0).to(device)
                masks = torch.cat([b['mask'][:1] for b in vis_batches], dim=0).to(device)
                masks = masks.float().unsqueeze(1)

                with torch.no_grad():
                    preds = model(images)

                fig = visualize_prediction(
                    images, masks, preds,
                    threshold=DEFAULT_VISUALIZATION_THRESHOLD,
                    max_images=len(images)
                )
                writer.add_figure('Val/Predictions', fig, global_step=global_step)
                plt.close(fig)

    # --- 5. Анализ метрик при разных порогах ---
    best_iou_across_thresholds = 0.0
    best_threshold = 0.5
    metrics_by_threshold = {}

    if hasattr(metrics_obj, 'compute_at_thresholds'):
        metrics_by_threshold = metrics_obj.compute_at_thresholds()
        for th, metrics_th in metrics_by_threshold.items():
            if metrics_th['IoU_forgery'] > best_iou_across_thresholds:
                best_iou_across_thresholds = metrics_th['IoU_forgery']
                best_threshold = th

    # Обновляет основные метрики на лучший найденный порог
    if best_iou_across_thresholds > metrics.get('IoU_forgery', 0) + 1e-6:
        # Фиксирует метрики для лучшего порога
        metrics = metrics_by_threshold[best_threshold].copy()
        metrics['best_threshold'] = best_threshold
        print(
            f"\n[Валидация @ {global_step}] Лучший порог: {best_threshold:.1f}, IoU: {best_iou_across_thresholds:.4f}")

    # --- 6. Логирование в TensorBoard ---
    if writer is not None:
        writer.add_scalar('Val/Loss/BCE', avg_bce, global_step)
        writer.add_scalar('Val/Loss/Dice', avg_dice, global_step)
        writer.add_scalar('Val/Loss/Focal', avg_focal, global_step)
        writer.add_scalar('Val/Loss/Total', avg_loss, global_step)

        # Основные метрики (с лучшим порогом)
        for name, value in metrics.items():
            if name != 'best_threshold':
                writer.add_scalar(f'Val/{name}', value, global_step)
        if 'best_threshold' in metrics:
            writer.add_scalar('Val/best_threshold', metrics['best_threshold'], global_step)

        # Дополнительно: логирует IoU для всех порогов
        for th, metrics_th in metrics_by_threshold.items():
            writer.add_scalar(f'Val_Th{th:.1f}/IoU_forgery', metrics_th['IoU_forgery'], global_step)

    # --- 7. Вывод в терминал ---
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'best_threshold')
    losses_str = f"BCE: {avg_bce:.3f}, Dice: {avg_dice:.3f}, Focal: {avg_focal:.3f}, Total: {avg_loss:.3f}"

    if 'best_threshold' in metrics:
        losses_str += f", best_th: {metrics['best_threshold']:.2f}"

    print(f"\n[Валидация @ {global_step}] (выборка: {len(sampled_indices)} изображений)")
    print(f"[Валидация @ {global_step}] Потери — {losses_str}")
    print(f"[Валидация @ {global_step}] Метрики — {metrics_str}")

    return metrics, avg_loss