# splits/splits.py — инструменты для группового разбиения датасета

import random
from collections import defaultdict
from pathlib import Path

import torch

# ==============================
# КОНСТАНТЫ РАЗБИЕНИЯ
# ==============================

# Стандартные пропорции разбиения (можно переопределить при вызове)
DEFAULT_TRAIN_RATIO = 0.90
DEFAULT_VAL_RATIO = 0.05
DEFAULT_TEST_RATIO = 0.05

# Символ-разделитель в имени файла (например, "group_123.jpg" → ключ = "123")
FILENAME_SEPARATOR = '_'


# ==============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def extract_group_key_from_path(path) -> str:
    """
    Извлекает групповой ключ из имени файла.

    Пример:
        "sample_abc_001.png" → "abc"
        "forged_xyz_42.jpg"  → "xyz"

    Аргументы:
        path (str или Path): путь к файлу.

    Возвращает:
        str: групповой ключ (вторая часть имени файла, разделённого '_').
    """
    filename = Path(path).stem  # удаляем расширение
    parts = filename.split(FILENAME_SEPARATOR)
    # Берём вторую часть (индекс 1), если есть хотя бы 2 части
    return parts[1] if len(parts) >= 2 else filename


# ==============================
# ОСНОВНАЯ ФУНКЦИЯ РАЗБИЕНИЯ
# ==============================

def split_dataset_by_groups(
        dataset,
        save_path,
        train_ratio=DEFAULT_TRAIN_RATIO,
        val_ratio=DEFAULT_VAL_RATIO,
        test_ratio=DEFAULT_TEST_RATIO,
        seed=42
):
    """
    Делит датасет на train/val/test с учётом группировки по идентификатору.

    Важно: разбиение происходит **на уровне групп**, а не отдельных изображений.
    Все изображения из одной группы попадают только в одну выборку (train/val/test).

    Аргументы:
        dataset: инициализированный ForgerySegmentationDataset.
        save_path (str): путь для сохранения индексов (в формате .pt).
        train_ratio, val_ratio, test_ratio (float): доли выборок (должны в сумме давать 1.0).
        seed (int): для воспроизводимости случайного перемешивания.

    Возвращает:
        tuple: (train_indices, val_indices, test_indices) — списки индексов.
    """
    # Проверка корректности пропорций
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"Сумма пропорций должна быть 1.0, получено: {total_ratio}"

    # 1. Извлекаем все стемы и их индексы из датасета
    stems = []
    for idx, (img_path, _) in enumerate(dataset.file_pairs):
        stem = Path(img_path).stem
        stems.append((idx, stem))

    print(f"Датасет содержит {len(stems)} элементов.")

    # 2. Перемешиваем пары (индекс, стем) для случайного порядка групп
    random.seed(seed)
    random.shuffle(stems)

    # 3. Группируем индексы по ключу
    groups = defaultdict(list)  # key → список индексов
    for idx, stem in stems:
        key = extract_group_key_from_path(stem)
        groups[key].append(idx)

    group_keys = list(groups.keys())
    print(f"Сформировано {len(group_keys)} групп.")

    # 4. Делим группы на выборки
    n_total = len(group_keys)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val  # гарантирует, что сумма = n_total

    # Защита от ошибок округления
    if n_test < 0:
        raise ValueError(
            f"Округление привело к отрицательному размеру test: "
            f"train={n_train}, val={n_val}, total={n_total}"
        )

    # Формируем множества групп для каждой выборки
    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train: n_train + n_val])
    test_groups = set(group_keys[n_train + n_val: n_train + n_val + n_test])

    # 5. Собираем индексы изображений по группам
    train_indices, val_indices, test_indices = [], [], []
    for key, indices in groups.items():
        if key in train_groups:
            train_indices.extend(indices)
        elif key in val_groups:
            val_indices.extend(indices)
        elif key in test_groups:
            test_indices.extend(indices)

    # 6. Сохраняем результат
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'group_keys_order': group_keys,
        'seed': seed,
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
    }, save_path)

    print(f"Разбиение сохранено: {save_path}")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Test:  {len(test_indices)}")

    return train_indices, val_indices, test_indices