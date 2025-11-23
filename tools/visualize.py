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


def validate_epoch(model, val_loader, metrics_obj, device, writer=None, epoch=0, visualize_every=5):
    model.eval()
    metrics_obj.reset()

    all_images, all_masks, all_preds = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).float().unsqueeze(1)

            preds = model(images)
            metrics_obj.update(preds, masks)

            # Сохраняем первые несколько примеров для визуализации
            if len(all_images) < 4:
                all_images.append(images[:4])
                all_masks.append(masks[:4])
                all_preds.append(preds[:4])

    # Собираем метрики
    metrics = metrics_obj.compute()

    # Визуализация (раз в N эпох)
    if writer is not None and (epoch + 1) % visualize_every == 0:
        vis_images = torch.cat(all_images, dim=0)[:4]
        vis_masks = torch.cat(all_masks, dim=0)[:4]
        vis_preds = torch.cat(all_preds, dim=0)[:4]

        fig = visualize_prediction(vis_images, vis_masks, vis_preds)
        writer.add_figure('Validation/Predictions', fig, global_step=epoch)
        plt.close(fig)  # освобождаем память

    return metrics