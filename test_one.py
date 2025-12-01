import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2

from model.pvtv2 import PVTv2B5ForForgerySegmentation

# --- Путь к модели и изображению ---
MODEL_PATH = "model/best_model_iou_0.3628_iter_180000.pth"
IMAGE_PATH = "test.tif"
OUTPUT_PATH = "prediction.png"

# --- Конфигурация ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512  # Размер, на котором обучалась модель

# --- Модель ---
model = PVTv2B5ForForgerySegmentation(img_size=IMG_SIZE)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# --- Нормализация ---
transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def load_and_preprocess_image(image_path, target_size=512):
    """Загружает изображение, приводит к тензору [3, H, W]"""
    image = np.array(Image.open(image_path).convert('RGB'))
    original_h, original_w = image.shape[:2]

    # Сохраняем оригинальные размеры
    # Создаём пустой тензор для батча
    batch = []
    masks = []  # не нужны, но для совместимости

    # Обрабатываем изображение
    # Если размер больше 512x512 — уменьшаем, если меньше — оставляем как есть
    # Но чтобы модель работала стабильно — делаем ресайз до IMG_SIZE, сохраняя пропорции
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Ресайз
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Дополняем до IMG_SIZE
    pad_h = max(0, IMG_SIZE - new_h)
    pad_w = max(0, IMG_SIZE - new_w)
    image_padded = cv2.copyMakeBorder(
        image_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
    )

    # Применяем трансформации
    augmented = transform(image=image_padded)
    image_tensor = augmented['image']  # [3, 512, 512]

    return image_tensor.unsqueeze(0), (original_h, original_w), (new_h, new_w), (pad_h, pad_w)

def postprocess_prediction(pred, original_size, resized_size, padding):
    """Возвращает маску того же размера, что и оригинал"""
    pred = pred.sigmoid().squeeze().cpu().numpy()  # [512, 512]

    # Удаляем паддинг
    pad_h, pad_w = padding
    if pad_h > 0:
        pred = pred[:-pad_h, :]
    if pad_w > 0:
        pred = pred[:, :-pad_w]

    # Ресайз обратно к оригинальному размеру
    pred_resized = cv2.resize(pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    # Бинаризуем по порогу 0.5
    mask = (pred_resized > 0.5).astype(np.uint8)

    return mask

def visualize_prediction(original_image, mask, output_path):
    """Рисует красную область на изображении и сохраняет без осей и белого фона"""
    # Создаём прозрачный фон (RGBA)
    overlay = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)

    # Красный цвет с прозрачностью 0.25
    red_color = [255, 0, 0, 64]  # R, G, B, A

    # Накладываем маску как красный цвет
    overlay[mask == 1] = red_color

    # Накладываем на оригинальное изображение
    # Оригинал — RGB, поэтому делаем его RGBA
    original_rgba = np.zeros_like(overlay)
    original_rgba[:, :, :3] = original_image
    original_rgba[:, :, 3] = 255  # полная непрозрачность

    # Смешиваем: оригинальное изображение + прозрачная красная маска
    result = original_rgba.copy()
    alpha = overlay[:, :, 3] / 255.0  # нормализуем альфа-канал
    for c in range(3):  # R, G, B
        result[:, :, c] = (1 - alpha) * original_rgba[:, :, c] + alpha * overlay[:, :, c]

    # Сохраняем как PNG с прозрачностью
    pil_image = Image.fromarray(result)
    pil_image.save(output_path, format='PNG')
    print(f"Предсказание сохранено: {output_path}")

def main():
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