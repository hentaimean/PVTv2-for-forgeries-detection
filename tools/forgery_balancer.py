# tools/forgery_balancer.py — сбалансированный сэмплер для батчей с подделками и оригиналами

import random
from pathlib import Path
from torch.utils.data import Sampler


# ==============================
# КОНСТАНТЫ СЭМПЛЕРА
# ==============================

# Префикс для оригинальных изображений (без подделок)
ORIGINAL_IMAGE_PREFIX = 'orig_'

# Минимальное число подделок в батче (даже при fg_ratio=0)
MIN_FG_PER_BATCH = 1


# ==============================
# ОСНОВНОЙ КЛАСС СЭМПЛЕРА
# ==============================

class ForgeryBalancedBatchSampler(Sampler):
    """
    Генератор батчей, сбалансированных по наличию подделок.

    Особенности:
      - Гарантирует, что в каждом батче есть как подделки, так и оригиналы (если доступны).
      - Работает с подмножеством датасета (allowed_indices).
      - Использует локальные индексы внутри подмножества для корректной работы с Subset.
    """

    def __init__(
        self,
        full_dataset,
        allowed_indices,
        batch_size,
        fg_ratio=0.5,
        shuffle=True,
        seed=42
    ):
        """
        Инициализация сэмплера.

        Аргументы:
            full_dataset: полный ForgerySegmentationDataset.
            allowed_indices: абсолютные индексы разрешённых элементов (например, train_indices).
            batch_size (int): размер батча.
            fg_ratio (float): доля подделок в батче (0.0–1.0).
            shuffle (bool): перемешивать ли индексы.
            seed (int): для воспроизводимости.
        """
        self.full_dataset = full_dataset
        self.allowed_indices = allowed_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Маппинг: абсолютный индекс → локальный индекс в allowed_indices
        self.abs_to_local = {
            abs_idx: local_idx
            for local_idx, abs_idx in enumerate(allowed_indices)
        }

        # Разделение на подделки и оригиналы (по локальным индексам)
        self.fg_local_indices = []
        self.clean_local_indices = []

        for local_idx, abs_idx in enumerate(allowed_indices):
            img_path, _ = full_dataset.file_pairs[abs_idx]
            stem = Path(img_path).stem
            if stem.startswith(ORIGINAL_IMAGE_PREFIX):
                self.clean_local_indices.append(local_idx)
            else:
                self.fg_local_indices.append(local_idx)

        # Настройка соотношения подделок в батче
        if len(self.clean_local_indices) == 0:
            # Если оригиналов нет — используем только подделки
            self.fg_ratio = 1.0
            self.fg_per_batch = self.batch_size
            self.clean_per_batch = 0
        else:
            self.fg_ratio = fg_ratio
            self.fg_per_batch = max(MIN_FG_PER_BATCH, int(self.batch_size * self.fg_ratio))
            self.clean_per_batch = self.batch_size - self.fg_per_batch

        print(f"ForgeryBalancedBatchSampler настроен:")
        print(f"  Подделок: {len(self.fg_local_indices)}")
        print(f"  Оригиналов: {len(self.clean_local_indices)}")
        print(f"  Подделок в батче: {self.fg_per_batch}, Оригиналов: {self.clean_per_batch}\n")

    def __iter__(self):
        """
        Генерирует батчи локальных индексов.
        """
        fg_indices = self.fg_local_indices.copy()
        clean_indices = self.clean_local_indices.copy()

        # Перемешивание при необходимости
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(fg_indices)
            if clean_indices:
                random.shuffle(clean_indices)

        batch = []
        fg_i = clean_i = 0

        # Формирование батчей
        while fg_i < len(fg_indices):
            # Добавляем подделки
            batch.extend(fg_indices[fg_i : fg_i + self.fg_per_batch])
            fg_i += self.fg_per_batch

            # Добавляем оригиналы (с циклическим повтором при нехватке)
            if clean_indices:
                for _ in range(self.clean_per_batch):
                    if clean_i >= len(clean_indices):
                        clean_i = 0
                        if self.shuffle:
                            random.shuffle(clean_indices)
                    batch.append(clean_indices[clean_i])
                    clean_i += 1

            # Выдача полного батча
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = []

        # Выдача остатка (если есть)
        if batch:
            yield batch[:self.batch_size]

    def __len__(self):
        """
        Возвращает приблизительное число батчей за эпоху.
        """
        if self.fg_per_batch == 0:
            return 0
        return max(1, len(self.fg_local_indices) // self.fg_per_batch)