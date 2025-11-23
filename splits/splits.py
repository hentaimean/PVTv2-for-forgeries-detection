import random
from collections import defaultdict
from pathlib import Path

import torch


def extract_group_key_from_path(path) -> str:
    """Извлекает групповой ключ из имени файла (Path или str)."""
    filename = Path(path).stem  # убираем расширение
    parts = filename.split('_')
    return parts[1] if len(parts) >= 2 else filename


def split_dataset_by_groups(
        dataset,
        save_path,
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=42
):
    """
    Делит датасет на train/val/test с учётом группировки по идентификатору.

    Аргументы:
        dataset: уже инициализированный ForgerySegmentationDataset
        train_ratio, val_ratio, test_ratio: доли выборок
        seed: для воспроизводимости
        save_path: куда сохранить индексы

    Возвращает:
        train_indices, val_indices, test_indices — списки индексов для Subset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 1. Извлекаем все стемы и их индексы из датасета
    stems = []
    for idx, (img_path, _) in enumerate(dataset.file_pairs):
        stem = Path(img_path).stem
        stems.append((idx, stem))

    print(f"Датасет содержит {len(stems)} элементов.")

    # 2. Перемешиваем (idx, stem) до группировки!
    random.seed(seed)
    random.shuffle(stems)

    # 3. Группируем по ключу → сохраняем индексы
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
    n_test = n_total - n_train - n_val  # теперь используется!

    # Защита от ошибок округления
    if n_test < 0:
        raise ValueError(f"Округление привело к отрицательному размеру test: "
                         f"train={n_train}, val={n_val}, total={n_total}")

    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train:n_train + n_val])
    test_groups = set(group_keys[n_train + n_val:n_train + n_val + n_test])

    # 5. Собираем индексы
    train_indices, val_indices, test_indices = [], [], []
    for key, indices in groups.items():
        if key in train_groups:
            train_indices.extend(indices)
        elif key in val_groups:
            val_indices.extend(indices)
        elif key in test_groups:
            test_indices.extend(indices)

    # 6. Сохраняем
    Path(save_path).parent.mkdir(exist_ok=True)
    torch.save({
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'group_keys_order': group_keys,
        'seed': seed
    }, save_path)

    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Test:  {len(test_indices)}")

    return train_indices, val_indices, test_indices