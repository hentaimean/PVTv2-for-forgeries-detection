# test_one.py — скрипт для тестирования модели на одном изображении

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2

from model.pvtv2 import PVTv2B5ForForgerySegmentation

# ==============================
# КОНСТАНТЫ ТЕСТИРОВАНИЯ
# ==============================

# Пути к файлам
MODEL_PATH = "model/best_model_iou_0.3628_iter_180000.pth"
IMAGE_PATH = "test.tif"
OUTPUT_PATH = "prediction.png"

# Параметры обработки
IMG_SIZE = 512
PREDICTION_THRESHOLD = 0.5
OVERLAY_ALPHA = 64  # прозрачность маски (0–255)

# Параметры нормализации (ImageNet)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Цвет маски
MASK_COLOR = [255, 0, 0]  # красный

# ==============================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
# ==============================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = A.Compose([
    A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ToTensorV2()
])

print("Загрузка модели...")
model = PVTv2B5ForForgerySegmentation(img_size=IMG_SIZE)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("Модель загружена.")


# ==============================
# ФУНКЦИИ ОБРАБОТКИ
# ==============================

def load_and_preprocess_image(image_path, target_size=IMG_SIZE):
    """
    Загружает и предобрабатывает изображение для подачи в модель.

    Аргументы:
        image_path (str): путь к изображению
        target_size (int): целевой размер стороны

    Возвращает:
        tuple: (тензор [1, 3, H, W], оригинальные размеры, размеры после ресайза, паддинг)
    """
    image = np.array(Image.open(image_path).convert('RGB'))
    original_h, original_w = image.shape[:2]

    # Сохраняем пропорции при ресайзе
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Дополнение до квадрата target_size x target_size
    pad_h = max(0, target_size - new_h)
    pad_w = max(0, target_size - new_w)
    image_padded = cv2.copyMakeBorder(
        image_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
    )

    augmented = transform(image=image_padded)
    image_tensor = augmented['image']  # [3, H, W]
    return image_tensor.unsqueeze(0), (original_h, original_w), (new_h, new_w), (pad_h, pad_w)


def postprocess_prediction(pred, original_size, resized_size, padding):
    """
    Преобразует предсказание модели в маску оригинального размера.

    Аргументы:
        pred (torch.Tensor): [1, 1, H, W] — логиты
        original_size (tuple): (H_orig, W_orig)
        resized_size (tuple): (H_resized, W_resized)
        padding (tuple): (pad_h, pad_w)

    Возвращает:
        np.ndarray: бинарная маска [H_orig, W_orig]
    """
    pred = pred.sigmoid().squeeze().cpu().numpy()
    pad_h, pad_w = padding

    if pad_h > 0:
        pred = pred[:-pad_h, :]
    if pad_w > 0:
        pred = pred[:, :-pad_w]

    pred_resized = cv2.resize(
        pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST
    )
    return (pred_resized > PREDICTION_THRESHOLD).astype(np.uint8)


def visualize_prediction(original_image, mask, output_path):
    """
    Накладывает полупрозрачную маску на изображение и сохраняет в PNG.

    Аргументы:
        original_image (np.ndarray): [H, W, 3] — оригинальное изображение
        mask (np.ndarray): [H, W] — бинарная маска подделок
        output_path (str): путь для сохранения результата
    """
    overlay = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
    overlay[mask == 1] = MASK_COLOR + [OVERLAY_ALPHA]  # [R, G, B, A]

    original_rgba = np.zeros_like(overlay)
    original_rgba[:, :, :3] = original_image
    original_rgba[:, :, 3] = 255

    # Альфа-смешивание
    result = original_rgba.copy()
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        result[:, :, c] = (1 - alpha) * original_rgba[:, :, c] + alpha * overlay[:, :, c]

    Image.fromarray(result).save(output_path, format='PNG')
    print(f"Результат сохранён: {output_path}")


# ==============================
# ОСНОВНАЯ ФУНКЦИЯ
# ==============================

def main():
    """Выполняет полный цикл: загрузка → предсказание → визуализация."""
    print("Загрузка изображения...")
    image_tensor, original_size, resized_size, padding = load_and_preprocess_image(IMAGE_PATH, IMG_SIZE)

    print("Предсказание...")
    with torch.no_grad():
        pred = model(image_tensor.to(DEVICE))

    print("Постобработка...")
    mask = postprocess_prediction(pred, original_size, resized_size, padding)

    print("Визуализация...")
    original_image = np.array(Image.open(IMAGE_PATH).convert('RGB'))
    visualize_prediction(original_image, mask, OUTPUT_PATH)


if __name__ == '__main__':
    main()