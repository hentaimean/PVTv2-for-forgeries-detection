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


class ForgerySegmentationDataset(Dataset):
    """
    Датасет для бинарной сегментации подделок на изображениях.

    Аннотации:
      - 0 = реальный (аутентичный) пиксель
      - 1 = поддельный (модифицированный) пиксель

    Особенности:
      - Автоматически сопоставляет изображения и маски по имени файла (без учёта расширения).
      - Поддерживает изображения **любого размера** (маленькие, большие, не квадратные).
      - При обучении с вероятностью `fg_crop_prob` гарантирует, что кроп содержит хотя бы один поддельный пиксель.
      - Совместим с `albumentations` для гибких аугментаций.
    """

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            transform: Optional[Callable] = None,
            fg_crop_prob: float = 0.7,
            crop_size: Tuple[int, int] = (512, 512),
            use_albumentations: bool = True,
            preload_forgery_coords: bool = False
    ):
        """
        Инициализация датасета.

        Аргументы:
            images_dir (str): Путь к папке с изображениями.
            masks_dir (str): Путь к папке с бинарными масками (только 0 и 1).
            transform (callable, optional): Трансформации от albumentations.
                Если не задан и use_albumentations=True, применяется только нормализация ImageNet + ToTensor.
            fg_crop_prob (float): Вероятность (от 0 до 1), с которой кроп будет содержать подделку.
                Рекомендуется 0.5–0.8 при обучении.
            crop_size (tuple): Целевой размер кропа (высота, ширина). Пример: (512, 512).
            use_albumentations (bool): Использовать ли albumentations.
            preload_forgery_coords (bool): Загрузить все координаты подделок при старте (может занять много памяти).
        """
        self.images_dir = Path(images_dir).resolve()
        self.masks_dir = Path(masks_dir).resolve()
        self.fg_crop_prob = fg_crop_prob
        self.crop_h, self.crop_w = crop_size
        self.use_albumentations = use_albumentations
        self._forgery_coords_cache = {}  # кеш: индекс → список координат (y, x) подделок

        # Собираем пары изображение-маска по имени без расширения
        self.file_pairs = self._get_file_pairs(self.images_dir, self.masks_dir)

        # Настраиваем трансформации
        if self.use_albumentations:
            if transform is None:
                # Стандартная нормализация под ImageNet (для PVTv2 и других предобученных моделей)
                self.transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = transform
        else:
            if transform is not None:
                raise ValueError("Если use_albumentations=False, transform должен быть None.")
            self.transform = None

        # Опционально: предзагрузка координат подделок
        if preload_forgery_coords:
            for i in range(len(self)):
                _ = self._get_forgery_coords(i)

    def _get_file_pairs(self, img_dir: Path, mask_dir: Path) -> list:
        """Сопоставляет изображения и маски по имени файла (без расширения)."""
        img_files = {f.stem: f for f in img_dir.iterdir() if f.is_file()}
        mask_files = {f.stem: f for f in mask_dir.iterdir() if f.is_file()}
        pairs = []
        for stem in sorted(img_files.keys()):
            if stem in mask_files:
                pairs.append((img_files[stem], mask_files[stem]))
        return pairs

    def _get_forgery_coords(self, idx: int) -> Optional[List[Tuple[int, int]]]:
        """
        Ленивая загрузка координат поддельных пикселей для маски с индексом idx.
        Возвращает список кортежей (y, x) или None, если подделок нет.
        """
        if idx in self._forgery_coords_cache:
            return self._forgery_coords_cache[idx]

        _, mask_path = self.file_pairs[idx]
        mask = np.array(Image.open(mask_path).convert('L'))
        # Находим все пиксели со значением 1
        coords = np.argwhere(mask == 1).tolist()
        coords = [(int(y), int(x)) for y, x in coords]  # приводим к int
        result = coords if coords else None
        self._forgery_coords_cache[idx] = result
        return result

    def _random_crop_around_forgery(self, image: np.ndarray, mask: np.ndarray, coords: List[Tuple[int, int]]):
        """
        Выполняет случайную обрезку так, чтобы в кропе оказался хотя бы один поддельный пиксель.
        """
        h, w = image.shape[:2]

        # Если изображение меньше кропа — дополняем нулями
        if h < self.crop_h or w < self.crop_w:
            pad_h = max(0, self.crop_h - h)
            pad_w = max(0, self.crop_w - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = image.shape[:2]

        # Выбираем случайный поддельный пиксель
        y, x = random.choice(coords)

        # Определяем допустимый диапазон для верхнего левого угла кропа
        top_min = max(0, y - self.crop_h + 1)
        top_max = min(y, h - self.crop_h)
        left_min = max(0, x - self.crop_w + 1)
        left_max = min(x, w - self.crop_w)

        # Защита от вырожденного случая (когда кроп больше изображения, но мы уже дополнили)
        top = random.randint(top_min, top_max) if top_min <= top_max else 0
        left = random.randint(left_min, left_max) if left_min <= left_max else 0

        return image[top:top + self.crop_h, left:left + self.crop_w], mask[top:top + self.crop_h,
                                                                      left:left + self.crop_w]

    def _random_crop_any(self, image: np.ndarray, mask: np.ndarray):
        """
        Стандартная случайная обрезка (может содержать только фон).
        """
        h, w = image.shape[:2]
        if h < self.crop_h or w < self.crop_w:
            pad_h = max(0, self.crop_h - h)
            pad_w = max(0, self.crop_w - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = image.shape[:2]
        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        return image[top:top + self.crop_h, left:left + self.crop_w], mask[top:top + self.crop_h,
                                                                      left:left + self.crop_w]

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Возвращает элемент датасета.

        Возвращает:
            - "image": тензор [3, H, W] (нормализованный под ImageNet)
            - "mask": тензор [H, W] (long, значения 0/1)
            - "img_path", "mask_path": пути к файлам
        """

        img_path, mask_path = self.file_pairs[idx]

        # Загружаем изображение и маску
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite pixel in {img_path}")

        # Проверяем корректность маски
        unique_vals = np.unique(mask)
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(f"Маска {mask_path} содержит недопустимые значения: {unique_vals}. "
                             f"Ожидаются только 0 (реальный) и 1 (поддельный).")

        # Решаем, какой тип кропа использовать
        if self.fg_crop_prob > 0 and random.random() < self.fg_crop_prob:
            coords = self._get_forgery_coords(idx)
            if coords is not None:
                image, mask = self._random_crop_around_forgery(image, mask, coords)
            else:
                # В этом изображении нет подделок → обычный кроп
                image, mask = self._random_crop_any(image, mask)
        else:
            image, mask = self._random_crop_any(image, mask)

        # Применяем трансформации albumentations
        if self.use_albumentations:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # [3, H, W]
            mask = augmented['mask']  # [H, W]
            mask = mask.to(torch.long)  # для CrossEntropyLoss
        else:
            # Возвращаем как PIL (редко используется)
            image = Image.fromarray(image)
            mask = Image.fromarray(mask.astype(np.uint8))

        if image.shape[0] < 4 or image.shape[1] < 4:
            raise ValueError(f"Image too small: {image.shape}")

        return {
            "image": image,
            "mask": mask,
            "img_path": str(img_path),
            "mask_path": str(mask_path)
        }


def get_training_augmentation(
        crop_size=(512, 512),
        fg_crop_prob=0.7
):
    """
    Создаёт пайплайн аугментаций для обучения модели обнаружения подделок.

    Особенности:
      - Не использует Resize до фиксированного размера (сохраняет пропорции).
      - RandomScale имитирует изменение расстояния до объекта.
      - PhotometricDistortion как в mmsegmentation.
      - Нормализация под ImageNet (для PVTv2).
      - RandomCrop выполняется **внутри датасета** (через ForgerySegmentationDataset),
        поэтому здесь он **не нужен**.

    Аргументы:
        crop_size (tuple): Не используется напрямую здесь — передаётся в датасет.
        fg_crop_prob (float): Не используется здесь — управляется в датасете.
        min_scale, max_scale (float): Диапазон случайного масштабирования.

    Возвращает:
        albumentations.Compose: трансформации для применения после кропа.
    """
    return A.Compose([
        # Горизонтальный флип
        A.HorizontalFlip(p=0.5),

        # Фотометрические искажения (как в mmsegmentation)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=10 / 255.0,  # ±10 в шкале [0,1]
                contrast_limit=(0.9, 1.2),  # контраст от 0.9 до 1.2
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,  # ±10 градусов
                sat_shift_limit=(0.8, 1.2),  # насыщенность от 0.8 до 1.2
                val_shift_limit=0,  # яркость не меняем дополнительно
                p=1.0
            )
        ], p=0.8),

        # Нормализация под ImageNet (обязательно для предобученного PVTv2)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # Преобразование в тензор
        ToTensorV2()
    ], p=1.0)


def get_validation_augmentation():
    """
    Пайплайн для валидации: только нормализация и преобразование в тензор.
    Не изменяет размер изображения — предполагается, что вход уже обработан датасетом.

    Возвращает:
        albumentations.Compose: трансформации для валидации.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])