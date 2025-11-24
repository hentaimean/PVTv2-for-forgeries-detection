import random
from pathlib import Path

from torch.utils.data import Sampler


class ForgeryBalancedBatchSampler(Sampler):
    def __init__(self, full_dataset, allowed_indices, batch_size, fg_ratio=0.5, shuffle=True, seed=42):
        self.full_dataset = full_dataset
        self.allowed_indices = allowed_indices  # абсолютные индексы
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Создаём mapping: абсолютный индекс -> локальный индекс
        self.abs_to_local = {abs_idx: local_idx for local_idx, abs_idx in enumerate(allowed_indices)}

        # Разделяем ЛОКАЛЬНЫЕ индексы на подделки и оригиналы
        self.fg_local_indices = []
        self.clean_local_indices = []

        for local_idx, abs_idx in enumerate(allowed_indices):
            img_path, _ = full_dataset.file_pairs[abs_idx]
            stem = Path(img_path).stem
            if stem.startswith('orig_'):
                self.clean_local_indices.append(local_idx)
            else:
                self.fg_local_indices.append(local_idx)

        # ... остальная логика, но работающая с ЛОКАЛЬНЫМИ индексами ...
        if len(self.clean_local_indices) == 0:
            self.fg_ratio = 1.0
        else:
            self.fg_ratio = fg_ratio

        fg_per_batch = max(1, int(self.batch_size * self.fg_ratio))
        self.clean_per_batch = self.batch_size - fg_per_batch

        if len(self.clean_local_indices) == 0:
            self.clean_per_batch = 0
            fg_per_batch = self.batch_size

        self.fg_per_batch = fg_per_batch

        print(f"Подделок: {len(self.fg_local_indices)}, Оригиналов: {len(self.clean_local_indices)}")

    def __iter__(self):
        fg_indices = self.fg_local_indices.copy()
        clean_indices = self.clean_local_indices.copy()

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(fg_indices)
            if clean_indices:
                random.shuffle(clean_indices)

        batch = []
        fg_i = clean_i = 0

        while fg_i < len(fg_indices):
            # Берём локальные индексы подделок
            batch.extend(fg_indices[fg_i : fg_i + self.fg_per_batch])
            fg_i += self.fg_per_batch

            # Берём локальные индексы оригиналов
            if clean_indices:
                for _ in range(self.clean_per_batch):
                    if clean_i >= len(clean_indices):
                        clean_i = 0
                        if self.shuffle:
                            random.shuffle(clean_indices)
                    batch.append(clean_indices[clean_i])
                    clean_i += 1

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch[:self.batch_size]

    def __len__(self):
        return max(1, len(self.fg_local_indices) // self.fg_per_batch)