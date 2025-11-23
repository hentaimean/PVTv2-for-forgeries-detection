import matplotlib.pyplot as plt
import numpy as np


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


def validate_epoch(model, val_loader, metrics_obj, device, writer=None, global_step=0):
    model.eval()
    metrics_obj.reset()

    total_bce = total_dice = total_focal = total_loss = 0.0
    num_batches = 0

    # Веса — должны быть теми же, что и при обучении!
    w_bce = 0.5
    w_dice = 1.0
    w_focal = 0.8

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True).float().unsqueeze(1)

            pred = model(images)

            # Считаем каждую потерю
            loss_bce = bce_loss(pred, masks)
            loss_dice = dice_loss_fn(pred, masks)
            loss_focal = focal_loss_fn(pred, masks)
            loss_total = w_bce * loss_bce + w_dice * loss_dice + w_focal * loss_focal

            # Накапливаем
            total_bce += loss_bce.item()
            total_dice += loss_dice.item()
            total_focal += loss_focal.item()
            total_loss += loss_total.item()
            num_batches += 1

            # Обновляем метрики
            metrics_obj.update(pred, masks)

    # Усредняем
    avg_bce = total_bce / num_batches
    avg_dice = total_dice / num_batches
    avg_focal = total_focal / num_batches
    avg_loss = total_loss / num_batches
    metrics = metrics_obj.compute()

    # Логирование в TensorBoard
    if writer is not None:
        writer.add_scalar('Val/Loss/BCE', avg_bce, global_step)
        writer.add_scalar('Val/Loss/Dice', avg_dice, global_step)
        writer.add_scalar('Val/Loss/Focal', avg_focal, global_step)
        writer.add_scalar('Val/Loss/Total', avg_loss, global_step)
        for name, value in metrics.items():
            writer.add_scalar(f'Val/{name}', value, global_step)

    # Формируем строку для терминала
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    losses_str = f"BCE: {avg_bce:.3f}, Dice: {avg_dice:.3f}, Focal: {avg_focal:.3f}, Total: {avg_loss:.3f}"

    print(f"\n[Val @ {global_step}] Losses — {losses_str}")
    print(f"[Val @ {global_step}] Metrics — {metrics_str}")

    return metrics, avg_loss