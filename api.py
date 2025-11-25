# api.py — веб-API для предсказания подделок на изображениях

import io
import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, send_file, jsonify

from model.pvtv2 import PVTv2B5ForForgerySegmentation

# ==============================
# КОНСТАНТЫ API
# ==============================

# Путь к предобученной модели
MODEL_PATH = "model/best_model_iou_0.3628_iter_180000.pth"

# Параметры обработки изображений
IMG_SIZE = 512
PREDICTION_THRESHOLD = 0.5
OVERLAY_ALPHA = 64  # прозрачность маски (0–255)

# Параметры нормализации (ImageNet)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Цвет маски в формате [R, G, B]
MASK_COLOR = [255, 0, 0]  # красный

# Настройки Flask
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False

# ==============================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ И ПРЕОБРАЗОВАНИЙ
# ==============================

# Устройство для вычислений
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Трансформации для входного изображения
transform = A.Compose([
    A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ToTensorV2()
])

# Загрузка модели один раз при старте сервера
print("Загрузка модели...")
model = PVTv2B5ForForgerySegmentation(img_size=IMG_SIZE)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("Модель успешно загружена.")


# ==============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def preprocess_image(image_np, target_size=IMG_SIZE):
    """
    Преобразует numpy-изображение в тензор для модели.

    Аргументы:
        image_np (np.ndarray): входное изображение [H, W, 3]
        target_size (int): целевой размер стороны для модели

    Возвращает:
        tuple: (тензор [1, 3, H, W], оригинальные размеры, паддинг)
    """
    h, w = image_np.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = max(0, target_size - new_h)
    pad_w = max(0, target_size - new_w)
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    augmented = transform(image=padded)
    return augmented['image'].unsqueeze(0), (h, w), (pad_h, pad_w)


def postprocess_mask(pred_tensor, original_size, padding):
    """
    Восстанавливает предсказанную маску до исходного размера и бинаризует.

    Аргументы:
        pred_tensor (torch.Tensor): [1, 1, H, W] — логиты модели
        original_size (tuple): (высота, ширина) оригинального изображения
        padding (tuple): (pad_h, pad_w) — добавленный паддинг

    Возвращает:
        np.ndarray: бинарная маска [H_orig, W_orig] (0/1)
    """
    pred = pred_tensor.sigmoid().squeeze().cpu().numpy()
    pad_h, pad_w = padding

    if pad_h > 0:
        pred = pred[:-pad_h, :]
    if pad_w > 0:
        pred = pred[:, :-pad_w]

    pred_resized = cv2.resize(
        pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST
    )
    return (pred_resized > PREDICTION_THRESHOLD).astype(np.uint8)


def overlay_mask_on_image(image_np, mask):
    """
    Накладывает полупрозрачную цветную маску на изображение (в формате RGBA).

    Аргументы:
        image_np (np.ndarray): оригинальное изображение [H, W, 3]
        mask (np.ndarray): бинарная маска подделок [H, W]

    Возвращает:
        np.ndarray: изображение с наложенной маской [H, W, 4] (RGBA)
    """
    overlay = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    overlay[mask == 1] = MASK_COLOR + [OVERLAY_ALPHA]  # [R, G, B, A]

    original_rgba = np.zeros_like(overlay)
    original_rgba[:, :, :3] = image_np
    original_rgba[:, :, 3] = 255  # непрозрачный оригинал

    # Альфа-смешивание
    alpha = overlay[:, :, 3] / 255.0
    result = original_rgba.copy()
    for c in range(3):
        result[:, :, c] = (1 - alpha) * original_rgba[:, :, c] + alpha * overlay[:, :, c]

    return result


# ==============================
# FLASK API
# ==============================

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API-эндпоинт для предсказания подделок.

    Ожидает POST-запрос с файлом изображения в поле 'image'.
    Возвращает PNG-изображение с наложенной маской подделок.
    """
    if 'image' not in request.files:
        return jsonify({"error": "Не предоставлен файл изображения"}), 400

    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)

        # Предобработка → предсказание → постобработка
        tensor, orig_size, padding = preprocess_image(image_np, IMG_SIZE)
        with torch.no_grad():
            pred = model(tensor.to(DEVICE))
        mask = postprocess_mask(pred, orig_size, padding)

        # Наложение маски и отправка результата
        result_rgba = overlay_mask_on_image(image_np, mask)
        result_pil = Image.fromarray(result_rgba)

        img_io = io.BytesIO()
        result_pil.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)