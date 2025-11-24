import random
from pathlib import Path

from torch.utils.data import Sampler


class ForgeryBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, fg_ratio=0.5, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Разделяем по 'orig_'
        self.fg_indices = []
        self.clean_indices = []
        for i, (img_path, _) in enumerate(dataset.file_pairs):
            stem = Path(img_path).stem
            if stem.startswith('orig_'):
                self.clean_indices.append(i)
            else:
                self.fg_indices.append(i)

        # Защита от отсутствия оригиналов
        if len(self.clean_indices) == 0:
            print("Оригиналов не обнаружено. Используем только подделки.")
            self.fg_ratio = 1.0
        else:
            self.fg_ratio = fg_ratio

        fg_per_batch = max(1, int(self.batch_size * self.fg_ratio))
        self.clean_per_batch = self.batch_size - fg_per_batch

        # Защита: если оригиналов нет, но clean_per_batch > 0
        if len(self.clean_indices) == 0:
            self.clean_per_batch = 0
            fg_per_batch = self.batch_size

        print(f"Подделок: {len(self.fg_indices)}, Оригиналов: {len(self.clean_indices)}")
        print(f"   Батч: {fg_per_batch} подделок + {self.clean_per_batch} оригиналов")

    def __iter__(self):
        fg_indices = self.fg_indices.copy()
        clean_indices = self.clean_indices.copy()

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(fg_indices)
            if clean_indices:
                random.shuffle(clean_indices)

        fg_per_batch = self.batch_size - self.clean_per_batch
        batch = []
        fg_i = clean_i = 0

        while fg_i < len(fg_indices):
            # Подделки
            batch.extend(fg_indices[fg_i : fg_i + fg_per_batch])
            fg_i += fg_per_batch

            # Оригиналы (если есть)
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

        # Остаток (если нужно)
        if batch:
            yield batch[:self.batch_size]

    def __len__(self):
        fg_per_batch = self.batch_size - self.clean_per_batch
        return max(1, len(self.fg_indices) // fg_per_batch)