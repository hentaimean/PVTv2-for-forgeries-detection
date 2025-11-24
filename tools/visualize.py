import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def visualize_prediction(image, target_mask, pred_logits, threshold=0.5, max_images=4):
    """
    Создаёт визуализацию для TensorBoard.

    Args:
        image: [B, 3, H, W], нормализованное под ImageNet
        target_mask: [B, 1, H, W], 0/1
        pred_logits: [B, 1, H, W], сырые logits
        threshold: порог для бинаризации предсказания
        max_images: сколько изображений показывать

    Returns:
        fig: matplotlib.figure.Figure (можно передать в SummaryWriter.add_figure)
    """
    # Денормализация изображения (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    batch_size = min(image.shape[0], max_images)
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = axes[np.newaxis, :]  # приводим к 2D

    for i in range(batch_size):
        # Изображение
        img = image[i].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')

        # Истинная маска
        mask_true = target_mask[i].cpu().numpy().squeeze()
        axes[i, 1].imshow(mask_true, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        # Предсказанная маска
        pred = (pred_logits[i].sigmoid() > threshold).float().cpu().numpy().squeeze()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')

    plt.tight_layout()
    return fig


def validate_epoch(
        model,
        val_dataset,
        metrics_obj,
        device,
        bce_loss_fn,
        dice_loss_fn,
        writer=None,
        global_step=0,
        val_sample_size=1000,
        seed=None,
        visualize_every=1,
        max_images=4
):
    model.eval()
    metrics_obj.reset()

    # --- 1. Выбираем случайные индексы ---
    if seed is not None:
        random.seed(seed + global_step)  # для воспроизводимости
    total_size = len(val_dataset)
    if val_sample_size >= total_size:
        sampled_indices = list(range(total_size))
    else:
        sampled_indices = random.sample(range(total_size), val_sample_size)

    # --- 2. Создаём временный DataLoader только для этих индексов ---
    sampled_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=torch.utils.data.SubsetRandomSampler(sampled_indices),
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    total_bce = total_dice = total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in sampled_loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True).float().unsqueeze(1)

            pred = model(images)

            loss_bce = bce_loss_fn(pred, masks)
            loss_dice = dice_loss_fn(pred, masks)
            # loss_focal = focal_loss_fn(pred, masks)
            loss_total = loss_bce + loss_dice  # + w_focal * loss_focal

            total_bce += loss_bce.item()
            total_dice += loss_dice.item()
            # total_focal += loss_focal.item()
            total_loss += loss_total.item()
            num_batches += 1

            metrics_obj.update(pred, masks)

    # Усреднение
    avg_bce = total_bce / num_batches
    avg_dice = total_dice / num_batches
    # avg_focal = total_focal / num_batches
    avg_loss = total_loss / num_batches
    metrics = metrics_obj.compute()

    if writer is not None:
        current_val_idx = global_step // VAL_INTERVAL  # номер валидации (1, 2, 3, ...)
        if current_val_idx % visualize_every == 0 or current_val_idx == 1:  # первая — всегда

            # Берём первые max_images из sampled_loader
            vis_batches = []
            for i, batch in enumerate(sampled_loader):
                if i >= max_images:
                    break
                vis_batches.append(batch)

            if vis_batches:
                # Собираем тензоры
                images = torch.cat([b['image'][:1] for b in vis_batches], dim=0).to(device)
                masks = torch.cat([b['mask'][:1] for b in vis_batches], dim=0).to(device)
                masks = masks.float().unsqueeze(1)

                with torch.no_grad():
                    preds = model(images)

                # Создаём фигуру
                fig = visualize_prediction(
                    images, masks, preds,
                    threshold=0.5,
                    max_images=len(images)
                )
                writer.add_figure('Val/Predictions', fig, global_step=global_step)
                plt.close(fig)

    # Логирование
    if writer is not None:
        writer.add_scalar('Val/Loss/BCE', avg_bce, global_step)
        writer.add_scalar('Val/Loss/Dice', avg_dice, global_step)
        writer.add_scalar('Val/Loss/Total', avg_loss, global_step)
        for name, value in metrics.items():
            writer.add_scalar(f'Val/Metrics/{name}', value, global_step)

    # Терминал
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    losses_str = f"BCE: {avg_bce:.3f}, Dice: {avg_dice:.3f}, Total: {avg_loss:.3f}"

    print(f"\n[Val @ {global_step}] (sampled {len(sampled_indices)} images)")
    print(f"[Val @ {global_step}] Losses — {losses_str}")
    print(f"[Val @ {global_step}] Metrics — {metrics_str}")

    return metrics, avg_loss