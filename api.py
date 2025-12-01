import io

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, send_file, jsonify

from model.pvtv2 import PVTv2B5ForForgerySegmentation

# --- Конфигурация ---
MODEL_PATH = "model/best_model_iou_0.3628_iter_180000.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512

app = Flask(__name__)

# Инициализация модели один раз при старте
print("Загрузка модели...")
model = PVTv2B5ForForgerySegmentation(img_size=IMG_SIZE)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("Модель загружена.")

# --- Трансформации ---
transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def preprocess_image(image_np, target_size=512):
    """Преобразует numpy-изображение в тензор [1, 3, 512, 512]"""
    h, w = image_np.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = max(0, IMG_SIZE - new_h)
    pad_w = max(0, IMG_SIZE - new_w)
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    augmented = transform(image=padded)

    return augmented['image'].unsqueeze(0), (h, w), (pad_h, pad_w)

def postprocess_mask(pred_tensor, original_size, padding):
    """Восстанавливает маску до исходного размера и бинаризует"""
    pred = pred_tensor.sigmoid().squeeze().cpu().numpy()
    pad_h, pad_w = padding
    if pad_h > 0:
        pred = pred[:-pad_h, :]
    if pad_w > 0:
        pred = pred[:, :-pad_w]
    pred_resized = cv2.resize(pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    return (pred_resized > 0.5).astype(np.uint8)

def overlay_mask_on_image(image_np, mask):
    """Накладывает полупрозрачную красную маску на изображение (RGBA)"""
    overlay = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    overlay[mask == 1] = [255, 0, 0, 64]  # красный с прозрачностью

    original_rgba = np.zeros_like(overlay)
    original_rgba[:, :, :3] = image_np
    original_rgba[:, :, 3] = 255

    alpha = overlay[:, :, 3] / 255.0
    result = original_rgba.copy()
    for c in range(3):
        result[:, :, c] = (1 - alpha) * original_rgba[:, :, c] + alpha * overlay[:, :, c]

    return result

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)

        # Предобработка
        tensor, orig_size, padding = preprocess_image(image_np, IMG_SIZE)

        # Предсказание
        with torch.no_grad():
            pred = model(tensor.to(DEVICE))

        # Постобработка
        mask = postprocess_mask(pred, orig_size, padding)

        # Наложение маски
        result_rgba = overlay_mask_on_image(image_np, mask)

        # Сохранение в байты
        result_pil = Image.fromarray(result_rgba)
        img_io = io.BytesIO()
        result_pil.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)