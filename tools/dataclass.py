# tools/dataclass.py — датасет для бинарной сегментации подделок

import random
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# ==============================
# КОНСТАНТЫ ДАТАСЕТА
# ==============================

# Параметры по умолчанию для датасета
DEFAULT_CROP_SIZE = (512, 512)
DEFAULT_FG_CROP_PROB = 0.8
DEFAULT_IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]

# Параметры аугментаций
TRAIN_AUG_SCALE_MIN = 0.7  # 1 - 0.3
TRAIN_AUG_SCALE_MAX = 1.3  # 1 + 0.3
TRAIN_AUG_SCALE_PROB = 0.5
TRAIN_AUG_HFLIP_PROB = 0.5

# Минимально допустимый размер изображения
MIN_IMAGE_SIZE = 4


# ==============================
# ОСНОВНОЙ КЛАСС ДАТАСЕТА
# ==============================

class ForgerySegmentationDataset(Dataset):
    """
    Датасет для бинарной сегментации подделок на изображениях.

    Аннотации:
      - 0 = реальный (аутентичный) пиксель
      - 1 = поддельный (модифицированный) пиксель

    Особенности:
      - Автоматически сопоставляет изображения и маски по имени файла (без расширения).
      - Поддерживает изображения **любого размера**.
      - При обучении с вероятностью `fg_crop_prob` гарантирует кроп с поддельным пикселем.
      - Совместим с `albumentations`.
    """

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            transform: Optional[Callable] = None,
            fg_crop_prob: float = DEFAULT_FG_CROP_PROB,
            crop_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
            use_albumentations: bool = True,
            preload_forgery_coords: bool = False
    ):
        """
        Инициализация датасета.

        Аргументы:
            images_dir (str): Путь к папке с изображениями.
            masks_dir (str): Путь к папке с бинарными масками (только 0 и 1).
            transform (callable, optional): Трансформации от albumentations.
            fg_crop_prob (float): Вероятность кропа с подделкой (0.0–1.0).
            crop_size (tuple): Целевой размер кропа (H, W).
            use_albumentations (bool): Использовать ли albumentations.
            preload_forgery_coords (bool): Загрузить координаты подделок при старте.
        """
        self.images_dir = Path(images_dir).resolve()
        self.masks_dir = Path(masks_dir).resolve()
        self.fg_crop_prob = fg_crop_prob
        self.crop_h, self.crop_w = crop_size
        self.use_albumentations = use_albumentations
        self._forgery_coords_cache = {}  # кеш: индекс → [(y, x), ...]

        # Сопоставление файлов
        self.file_pairs = self._get_file_pairs(self.images_dir, self.masks_dir)

        # Настройка трансформаций
        if self.use_albumentations:
            if transform is None:
                # Стандартная нормализация под ImageNet
                self.transform = A.Compose([
                    A.Normalize(
                        mean=DEFAULT_IMAGE_NORMALIZE_MEAN,
                        std=DEFAULT_IMAGE_NORMALIZE_STD
                    ),
                    ToTensorV2()
                ])
            else:
                self.transform = transform
        else:
            if transform is not None:
                raise ValueError("Если use_albumentations=False, transform должен быть None.")
            self.transform = None

        # Предзагрузка координат подделок (опционально)
        if preload_forgery_coords:
            for i in range(len(self)):
                _ = self._get_forgery_coords(i)

    def _get_file_pairs(self, img_dir: Path, mask_dir: Path) -> list:
        """Сопоставляет изображения и маски по имени (без расширения)."""
        img_files = {f.stem: f for f in img_dir.iterdir() if f.is_file()}
        mask_files = {f.stem: f for f in mask_dir.iterdir() if f.is_file()}
        pairs = []
        for stem in sorted(img_files.keys()):
            if stem in mask_files:
                pairs.append((img_files[stem], mask_files[stem]))
        return pairs

    def _get_forgery_coords(self, idx: int) -> Optional[List[Tuple[int, int]]]:
        """Возвращает список координат поддельных пикселей или None, если их нет."""
        if idx in self._forgery_coords_cache:
            return self._forgery_coords_cache[idx]

        _, mask_path = self.file_pairs[idx]
        mask = np.array(Image.open(mask_path).convert('L'))

        if mask.sum() == 0:
            self._forgery_coords_cache[idx] = None
            return None

        coords = np.argwhere(mask == 1).tolist()
        coords = [(int(y), int(x)) for y, x in coords]
        self._forgery_coords_cache[idx] = coords
        return coords

    def _random_crop_around_forgery(self, image: np.ndarray, mask: np.ndarray, coords: List[Tuple[int, int]]):
        """
        Случайная обрезка, гарантированно содержащая хотя бы один поддельный пиксель.
        """
        h, w = image.shape[:2]

        # Дополнение нулями, если изображение меньше кропа
        if h < self.crop_h or w < self.crop_w:
            pad_h = max(0, self.crop_h - h)
            pad_w = max(0, self.crop_w - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = image.shape[:2]

        # Выбираем случайный поддельный пиксель
        y, x = random.choice(coords)

        # Определяем допустимый диапазон для верхнего левого угла
        top_min = max(0, y - self.crop_h + 1)
        top_max = min(y, h - self.crop_h)
        left_min = max(0, x - self.crop_w + 1)
        left_max = min(x, w - self.crop_w)

        top = random.randint(top_min, top_max) if top_min <= top_max else 0
        left = random.randint(left_min, left_max) if left_min <= left_max else 0

        return (
            image[top:top + self.crop_h, left:left + self.crop_w],
            mask[top:top + self.crop_h, left:left + self.crop_w]
        )

    def _random_crop_any(self, image: np.ndarray, mask: np.ndarray):
        """Стандартная случайная обрезка (может содержать только фон)."""
        h, w = image.shape[:2]
        if h < self.crop_h or w < self.crop_w:
            pad_h = max(0, self.crop_h - h)
            pad_w = max(0, self.crop_w - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = image.shape[:2]
        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        return (
            image[top:top + self.crop_h, left:left + self.crop_w],
            mask[top:top + self.crop_h, left:left + self.crop_w]
        )

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Возвращает элемент датасета.

        Возвращает:
            dict с ключами:
                - "image": тензор [3, H, W] (нормализованный)
                - "mask": тензор [H, W] (long, 0/1)
                - "img_path", "mask_path": пути к файлам
        """
        img_path, mask_path = self.file_pairs[idx]

        # Загрузка
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Валидация
        if not np.isfinite(image).all():
            raise ValueError(f"Неконечные пиксели в изображении: {img_path}")
        if image.shape[0] < MIN_IMAGE_SIZE or image.shape[1] < MIN_IMAGE_SIZE:
            raise ValueError(f"Изображение слишком маленькое: {image.shape}")
        unique_vals = np.unique(mask)
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(f"Маска {mask_path} содержит недопустимые значения: {unique_vals}")

        # Выбор типа кропа
        if self.fg_crop_prob > 0 and random.random() < self.fg_crop_prob:
            coords = self._get_forgery_coords(idx)
            if coords is not None:
                image, mask = self._random_crop_around_forgery(image, mask, coords)
            else:
                image, mask = self._random_crop_any(image, mask)
        else:
            image, mask = self._random_crop_any(image, mask)

        # Применение трансформаций
        if self.use_albumentations:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # [3, H, W]
            mask = augmented['mask']  # [H, W]
            mask = mask.to(torch.long)
        else:
            image = Image.fromarray(image)
            mask = Image.fromarray(mask.astype(np.uint8))

        return {
            "image": image,
            "mask": mask,
            "img_path": str(img_path),
            "mask_path": str(mask_path)
        }


# ==============================
# ФУНКЦИИ АУГМЕНТАЦИЙ
# ==============================

def get_training_augmentation():
    """
    Пайплайн аугментаций для обучения.

    Примечание:
      - RandomCrop выполняется **внутри датасета**, поэтому здесь не используется.
      - Размер фиксируется в Resize, чтобы соответствовать crop_size.
    """
    return A.Compose([
        A.RandomScale(
            scale_limit=(TRAIN_AUG_SCALE_MIN - 1.0, TRAIN_AUG_SCALE_MAX - 1.0),
            p=TRAIN_AUG_SCALE_PROB
        ),
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=TRAIN_AUG_HFLIP_PROB),
        A.Normalize(
            mean=DEFAULT_IMAGE_NORMALIZE_MEAN,
            std=DEFAULT_IMAGE_NORMALIZE_STD
        ),
        ToTensorV2()
    ])


def get_validation_augmentation():
    """
    Пайплайн для валидации: только нормализация и преобразование в тензор.
    """
    return A.Compose([
        A.Normalize(
            mean=DEFAULT_IMAGE_NORMALIZE_MEAN,
            std=DEFAULT_IMAGE_NORMALIZE_STD
        ),
        ToTensorV2()
    ])